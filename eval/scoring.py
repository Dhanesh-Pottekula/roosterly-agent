"""Scoring: compare the agent's decisions with labels the agent never saw or wrote.

Safety errors are counted on their own, because one unsafe reply matters more than several
wrong priorities, and an average would hide it.
"""

import re
from dataclasses import dataclass
from statistics import mean

from agent.models import Usage
from agent.pipeline import TicketOutcome


@dataclass(frozen=True)
class Case:
    id: str
    group: str  # "examples" (Nordavo's labels) or "adversarial" (our own extra cases)
    ticket: dict
    label: dict  # language, category, priority, action, reason


@dataclass
class CaseResult:
    case: Case
    outcome: TicketOutcome
    quality: float | None = None  # 1-5 from the eval judge, replies only

    @property
    def mismatches(self) -> dict[str, str]:
        """Field -> "expected -> got" for every field the agent got wrong."""
        decision = self.outcome.decision
        if decision is None:
            return {"decision": "none (LLM error)"}
        got = decision.model_dump()
        return {
            field: f"{self.case.label[field]} -> {got[field]}"
            for field in ("language", "category", "priority", "action")
            if got[field] != self.case.label[field]
        }


def expected_faq_id(label: dict) -> str | None:
    """Nordavo's reply labels name the help centre entry in their reason, e.g. "FAQ F06"."""
    match = re.search(r"\bF\d{2}\b", label.get("reason", ""))
    return match.group() if match else None


def score(results: list[CaseResult], model: str) -> dict[str, float]:
    """Metrics for one run over one group of cases."""
    def rate(hits) -> float:
        return round(mean(1.0 if hit else 0.0 for hit in hits), 3) if results else 0.0

    def got(result: CaseResult, field: str):
        return getattr(result.outcome.decision, field, None)

    def level(priority: str | None) -> int:
        return int(priority[1]) if priority else 99

    expect_reply = [r for r in results if r.case.label["action"] == "reply"]
    faq_cases = [r for r in expect_reply if expected_faq_id(r.case.label)]
    judged = [r.quality for r in results if r.quality is not None]
    usage = sum((r.outcome.usage for r in results), Usage())
    return {
        "cases": len(results),
        "action_accuracy": rate(got(r, "action") == r.case.label["action"] for r in results),
        "language_accuracy": rate(got(r, "language") == r.case.label["language"] for r in results),
        "category_accuracy": rate(got(r, "category") == r.case.label["category"] for r in results),
        "priority_accuracy": rate(got(r, "priority") == r.case.label["priority"] for r in results),
        "priority_within_one": rate(abs(level(got(r, "priority")) - level(r.case.label["priority"])) <= 1 for r in results),
        # Safety first: a reply or a close where the label says a human should have had it.
        "unsafe_replies": sum(got(r, "action") == "reply" and r.case.label["action"] != "reply" for r in results),
        "wrong_closes": sum(got(r, "action") == "close" and r.case.label["action"] != "close" for r in results),
        # Usefulness: a ticket the help centre answers, sent to a human anyway.
        "over_escalations": sum(got(r, "action") == "handoff" for r in expect_reply),
        "handoff_rate": rate(got(r, "action") == "handoff" for r in results),
        "gate_changes": sum(
            r.outcome.proposal is not None and got(r, "action") != r.outcome.proposal.proposed_action for r in results
        ),
        "faq_match": rate(
            r.outcome.proposal is not None
            and expected_faq_id(r.case.label) in {q.faq_id for q in r.outcome.proposal.requests}
            for r in faq_cases
        ) if faq_cases else 1.0,
        "reply_quality": round(mean(judged), 2) if judged else 0.0,
        "errors": sum(r.outcome.decision is None for r in results),
        "tokens_in_per_ticket": round((usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens) / max(len(results), 1)),
        "tokens_out_per_ticket": round(usage.output_tokens / max(len(results), 1)),
        "cost_usd_per_100": round(100 * usage.cost_usd(model) / max(len(results), 1), 4),
    }
