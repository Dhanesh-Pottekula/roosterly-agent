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
from agent.runlog import RunLogger, append_spend, ticket_fields, utc_now

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


class Runner:
    def __init__(self, settings: Settings, helpdesk: HelpdeskClient, llm: LLM, kb: KnowledgeBase,
                 dry_run: bool = False, log_dir: str | Path = "logs"):
        self.settings, self.helpdesk, self.llm, self.kb = settings, helpdesk, llm, kb
        self.dry_run = dry_run
        self.log_dir = log_dir
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.log = RunLogger(f"dry-{stamp}" if dry_run else stamp, log_dir)
        self.usage = Usage()

    def run(self, limit: int | None = None) -> RunSummary:
        self.helpdesk.status(send_name=True)
        ticket_ids = self.helpdesk.list_open_ids()[:limit]
        outcomes: Counter[str] = Counter()

        def finish(record: dict) -> None:
            """Write the ticket's one log line: its final outcome in this run."""
            outcomes[record["outcome"]] += 1
            self.log.write(record)

        retry_ids = []
        for ticket_id in ticket_ids:
            record = self._handle(ticket_id)
            if record["outcome"] in LEFT_OPEN:
                retry_ids.append(ticket_id)  # try once more at the end of the run
            else:
                finish(record)  # logged right away, so a crash loses nothing
        for ticket_id in retry_ids:
            finish(self._handle(ticket_id))

        return self._summarise(len(ticket_ids), dict(outcomes))

    def _handle(self, ticket_id: str) -> dict:
        """Take one ticket as far as it can go, and return its log record."""
        started = time.monotonic()
        attempts_before = self.helpdesk.attempts
        record = {"ts": utc_now(), "run_id": self.log.run_id, "ticket_id": ticket_id}
        try:
            ticket = self.helpdesk.get_ticket(ticket_id)
            if ticket is None:
                record["outcome"] = "not_found"
            elif ticket.handled is not None:
                record["outcome"] = "skipped_handled"  # answered since we listed it: never answer twice
            else:
                result = decide_ticket(ticket, self.kb, self.llm, self.settings.llm_model)
                self.usage += result.usage
                record |= {"received_at": ticket.received_at} | ticket_fields(result, self.settings.llm_model)
                record |= self._deliver(ticket_id, result)
        except TransientExhausted:
            record["outcome"] = "failed_transient"
        except (FatalAPIError, LLMFatalError):
            raise  # wrong key or setup: every ticket would fail, so stop
        except Exception as error:  # one odd ticket must not block the rest of the queue
            record |= {"outcome": "failed_error", "error": type(error).__name__}
        record["http_attempts"] = self.helpdesk.attempts - attempts_before
        record["duration_ms"] = round((time.monotonic() - started) * 1000)
        return record

    def _deliver(self, ticket_id: str, result: TicketOutcome) -> dict:
        """Post the decision, unless there is none or this is a dry run."""
        if result.decision is None:
            return {"outcome": "left_open_llm"}
        if self.dry_run:
            print_decision(ticket_id, result)
            return {"outcome": "dry_run"}
        try:
            return {"outcome": self.helpdesk.post_decision(ticket_id, result.decision).value}
        except ValidationRejected as rejected:
            # The helpdesk refused the fields. Do not drop the ticket: give it to a human.
            outcome = self.helpdesk.post_decision(ticket_id, _fallback_handoff(result.decision)).value
            return {"outcome": outcome, "action": "handoff", "validation_problems": rejected.problems}

    def _summarise(self, ticket_count: int, outcomes: dict[str, int]) -> RunSummary:
        model = self.settings.llm_model
        summary = RunSummary(
            run_id=self.log.run_id,
            outcomes=outcomes,
            left_open=sum(count for outcome, count in outcomes.items() if outcome in LEFT_OPEN),
            usage=self.usage,
            cost_usd=self.usage.cost_usd(model),
            status=self.helpdesk.status(),
        )
        self.log.write_summary({
            "run_id": summary.run_id, "model": model, "tickets": ticket_count, "outcomes": outcomes,
            "tokens_in": self.usage.input_tokens, "tokens_out": self.usage.output_tokens,
            "cache_read": self.usage.cache_read_tokens, "cache_write": self.usage.cache_write_tokens,
            "cost_usd": round(summary.cost_usd, 6), "helpdesk_status": summary.status,
        })
        append_spend("dry-run" if self.dry_run else "run", summary.run_id, self.usage, model, self.log_dir)
        return summary


def _fallback_handoff(rejected: Decision) -> Decision:
    return Decision(
        action="handoff", language=rejected.language, category=rejected.category,
        priority=rejected.priority, confidence=0.3,
        reason="The helpdesk rejected the automatic decision, so a human must handle this ticket.",
        handoff_note="The agent's decision was rejected by the helpdesk. Please read the original ticket; "
                     "nothing was sent to the customer.",
    )


def print_decision(ticket_id: str, result: TicketOutcome) -> None:
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
