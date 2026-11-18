"""Run logs: one JSON line per ticket in logs/, and a running total of LLM spend.

A log line holds decisions and codes, never ticket text or drafts. It still passes through
redaction before it is written, so payment details cannot reach a log by accident.
"""

import json
from datetime import datetime, UTC
from pathlib import Path

from agent.models import Usage
from agent.pipeline import TicketOutcome
from agent.redact import redact_payment


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def ticket_fields(result: TicketOutcome, model: str) -> dict:
    """What the log records about one decided ticket. Codes and labels only: no customer text."""
    usage = result.usage
    fields: dict = {
        "model": model,
        "tokens_in": usage.input_tokens + usage.cache_read_tokens,
        "tokens_out": usage.output_tokens,
        "cost_usd": round(usage.cost_usd(model), 6),
        "handoff_reasons": result.handoff_reasons,
    }
    if result.decision:
        fields |= result.decision.model_dump(include={"language", "category", "priority", "action", "confidence"})
    if result.proposal:
        fields |= {
            "proposed_action": result.proposal.proposed_action,
            "faq_ids": sorted({request.faq_id for request in result.proposal.requests if request.faq_id}),
            "rule_flags": result.proposal.rule_flags,
            "instructions_in_ticket": result.proposal.contains_instructions_to_ai,
        }
    if result.verdict:
        fields["verifier_objections"] = result.verdict.problems(result.decision.language) + result.verdict.always_human
    if result.transient_error:
        fields["llm_error"] = result.transient_error[:100]
    return fields


class RunLogger:
    def __init__(self, run_id: str, directory: str | Path = "logs"):
        self.run_id = run_id
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"{run_id}.jsonl"

    def write(self, fields: dict) -> None:
        line, _ = redact_payment(json.dumps(fields, ensure_ascii=False))
        with self.path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")

    def write_summary(self, summary: dict) -> None:
        path = self.directory / f"{self.run_id}.summary.json"
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_spend(kind: str, run_id: str, usage: Usage, model: str, directory: str | Path = "logs") -> float:
    """Record what a run cost and return the total spent so far, in USD."""
    path = Path(directory) / "spend.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": utc_now(),
        "kind": kind,
        "run_id": run_id,
        "model": model,
        "tokens_in": usage.input_tokens,
        "tokens_out": usage.output_tokens,
        "cache_read": usage.cache_read_tokens,
        "cost_usd": round(usage.cost_usd(model), 6),
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")
    return sum(json.loads(line)["cost_usd"] for line in path.read_text(encoding="utf-8").splitlines())
