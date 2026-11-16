"""The run loop: list open tickets, decide each one, post it once, log it once.

Safe to re-run. The helpdesk's own state is the only memory: we list open tickets, and
re-read each ticket right before deciding. After a crash, a second run simply finds fewer
open tickets, so no customer gets two answers.
"""

import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from agent.config import Settings
from agent.faq import KnowledgeBase
from agent.helpdesk import FatalAPIError, HelpdeskClient, TransientExhausted, ValidationRejected
from agent.llm import LLM, LLMFatalError
from agent.models import Decision, Usage
from agent.pipeline import TicketOutcome, decide_ticket
from agent.runlog import RunLogger, append_spend, utc_now

# Outcomes that leave the ticket open. They get a second try in the same run, then a non-zero exit.
LEFT_OPEN = {"left_open_llm", "failed_transient", "failed_error"}


@dataclass
class RunSummary:
    run_id: str
    outcomes: dict[str, int]
    left_open: int
    usage: Usage
    cost_usd: float
    status: dict


def run(
    settings: Settings,
    helpdesk: HelpdeskClient,
    llm: LLM,
    kb: KnowledgeBase,
    dry_run: bool = False,
    limit: int | None = None,
    log_dir: str | Path = "logs",
) -> RunSummary:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = RunLogger(f"dry-{stamp}" if dry_run else stamp, log_dir)
    helpdesk.status(send_name=True)
    ticket_ids = helpdesk.list_open_ids()[:limit]

    usage = Usage()
    outcomes: Counter[str] = Counter()

    def handle(ticket_id: str) -> dict:
        nonlocal usage
        record = _handle_ticket(ticket_id, settings, helpdesk, llm, kb, dry_run) | {"run_id": log.run_id}
        usage += record.pop("_usage")
        return record

    def finish(record: dict) -> None:
        """Write the ticket's one log line: its final outcome in this run."""
        outcomes[record["outcome"]] += 1
        log.write(record)

    retry_ids = []
    for ticket_id in ticket_ids:
        record = handle(ticket_id)
        if record["outcome"] in LEFT_OPEN:
            retry_ids.append(ticket_id)  # a temporary failure: try once more at the end
        else:
            finish(record)  # logged right away, so a crash loses nothing
    for ticket_id in retry_ids:
        finish(handle(ticket_id))

    summary = RunSummary(
        run_id=log.run_id,
        outcomes=dict(outcomes),
        left_open=sum(outcomes[name] for name in LEFT_OPEN),
        usage=usage,
        cost_usd=usage.cost_usd(settings.llm_model),
        status=helpdesk.status(),
    )
    log.write_summary({
        "run_id": summary.run_id, "model": settings.llm_model, "tickets": len(ticket_ids),
        "outcomes": summary.outcomes, "tokens_in": usage.input_tokens, "tokens_out": usage.output_tokens,
        "cache_read": usage.cache_read_tokens, "cache_write": usage.cache_write_tokens,
        "cost_usd": round(summary.cost_usd, 6), "helpdesk_status": summary.status,
    })
    append_spend("dry-run" if dry_run else "run", log.run_id, usage, settings.llm_model, log_dir)
    return summary


def _handle_ticket(ticket_id, settings, helpdesk, llm, kb, dry_run) -> dict:
    started = time.monotonic()
    attempts_before = helpdesk.attempts
    record = {"ts": utc_now(), "ticket_id": ticket_id, "_usage": Usage()}
    try:
        ticket = helpdesk.get_ticket(ticket_id)
        if ticket is None:
            record["outcome"] = "not_found"
        elif ticket.handled is not None:
            record["outcome"] = "skipped_handled"  # answered since we listed it: never answer twice
        else:
            result = decide_ticket(ticket, kb, llm, settings.llm_model)
            record |= _describe(result, settings.llm_model) | {"received_at": ticket.received_at}
            if result.decision is None:
                record["outcome"] = "left_open_llm"
            elif dry_run:
                _print_decision(ticket_id, result)
                record["outcome"] = "dry_run"
            else:
                record |= _post(helpdesk, ticket_id, result.decision)
    except TransientExhausted:
        record["outcome"] = "failed_transient"
    except (FatalAPIError, LLMFatalError):
        raise  # wrong key or setup: every ticket would fail, so stop
    except Exception as error:  # one odd ticket must not block the rest of the queue
        record |= {"outcome": "failed_error", "error": type(error).__name__}
    record["http_attempts"] = helpdesk.attempts - attempts_before
    record["duration_ms"] = round((time.monotonic() - started) * 1000)
    return record


def _post(helpdesk: HelpdeskClient, ticket_id: str, decision: Decision) -> dict:
    """Post the decision. If the helpdesk rejects its fields (422), hand the ticket to a human instead."""
    try:
        return {"outcome": helpdesk.post_decision(ticket_id, decision).value}
    except ValidationRejected as rejected:
        fallback = Decision(
            action="handoff", language=decision.language, category=decision.category,
            priority=decision.priority, confidence=0.3,
            reason="The helpdesk rejected the automatic decision, so a human must handle this ticket.",
            handoff_note="The agent's decision was rejected by the helpdesk. Please read the original ticket; "
                         "nothing was sent to the customer.",
        )
        outcome = helpdesk.post_decision(ticket_id, fallback).value
        return {"outcome": outcome, "action": "handoff", "validation_problems": rejected.problems}


def _describe(result: TicketOutcome, model: str) -> dict:
    """The log fields for one decided ticket. Codes and labels only: no customer text."""
    fields: dict = {
        "model": model,
        "tokens_in": result.usage.input_tokens + result.usage.cache_read_tokens + result.usage.cache_write_tokens,
        "tokens_out": result.usage.output_tokens,
        "cost_usd": round(result.usage.cost_usd(model), 6),
        "handoff_reasons": result.handoff_reasons,
        "_usage": result.usage,
    }
    if result.decision:
        decision = result.decision
        fields |= {
            "language": decision.language, "category": decision.category, "priority": decision.priority,
            "action": decision.action, "confidence": decision.confidence,
        }
    if result.proposal:
        fields |= {
            "proposed_action": result.proposal.proposed_action,
            "faq_ids": sorted({r.faq_id for r in result.proposal.requests if r.faq_id}),
            "rule_flags": result.proposal.rule_flags,
            "instructions_in_ticket": result.proposal.contains_instructions_to_ai,
        }
    if result.verdict:
        fields["verifier_ok"] = not result.verdict.problems(result.decision.language) and not result.verdict.always_human
    if result.transient_error:
        fields["llm_error"] = result.transient_error[:100]
    return fields


def _print_decision(ticket_id: str, result: TicketOutcome) -> None:
    """Dry run: show what would be posted, with the English translation of a reply."""
    decision = result.decision
    print(f"\n=== {ticket_id}: {decision.action} · {decision.language} · {decision.category} · {decision.priority}")
    print(f"reason: {decision.reason}")
    if decision.draft_reply:
        print(f"draft:\n{decision.draft_reply}")
        if result.verdict:
            print(f"draft in English (verifier):\n{result.verdict.translation_en}")
    if decision.handoff_note:
        print(f"handoff note:\n{decision.handoff_note}")
