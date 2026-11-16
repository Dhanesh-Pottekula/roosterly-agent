"""One ticket in, one decision out. No helpdesk calls here, so the eval can run this offline.

    redact -> detect language -> propose (LLM) -> check + verify (replies only) -> policy gate -> redact
"""

import re
from dataclasses import dataclass, field

from agent.checks import check_reply
from agent.decide import Proposal, decide
from agent.faq import KnowledgeBase
from agent.language import detect_language
from agent.llm import LLM, LLMDeterministicError, LLMTransientError
from agent.models import Decision, Ticket, Usage
from agent.policy import apply_policy, posted_language
from agent.redact import redact_payment, redact_ticket
from agent.verify import Verdict, verify_reply

_ONLY_REPLY_PREFIXES = re.compile(r"^\s*((re|fw|fwd|aw|antw)\s*:\s*)*$", re.IGNORECASE)


@dataclass
class TicketOutcome:
    decision: Decision | None  # None means: leave the ticket open and try again later
    proposal: Proposal | None = None
    verdict: Verdict | None = None
    handoff_reasons: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    transient_error: str | None = None


def decide_ticket(ticket: Ticket, kb: KnowledgeBase, llm: LLM, model: str) -> TicketOutcome:
    if is_empty(ticket):
        return TicketOutcome(_close_empty())

    rt = redact_ticket(ticket)
    detected = detect_language(f"{rt.subject}\n{rt.body}")
    outcome = TicketOutcome(decision=None)
    try:
        proposal, outcome.usage = decide(rt, kb, llm, model)
        outcome.proposal = proposal
        reply_problems: list[str] = []
        if proposal.proposed_action == "reply":
            language = posted_language(proposal.language, detected) or "en"
            reply_problems = check_reply(proposal.draft_reply or "", language, proposal.requests, rt, kb)
            if not reply_problems:  # the verifier costs a call, so it only sees drafts that passed
                outcome.verdict, usage = verify_reply(rt, proposal.draft_reply, kb, llm, model)
                outcome.usage += usage
    except LLMTransientError as error:
        outcome.transient_error = str(error)
        return outcome
    except LLMDeterministicError as error:
        outcome.handoff_reasons = [f"triage_failed:{error}"]
        outcome.decision = _handoff_after_failure(str(error), detected)
        return outcome

    result = apply_policy(rt, proposal, detected, reply_problems, outcome.verdict)
    outcome.handoff_reasons = result.handoff_reasons
    outcome.decision = _without_payment_details(result.decision)
    return outcome


def is_empty(ticket: Ticket) -> bool:
    """No body, and a subject that is blank or only a reply prefix such as "Re:"."""
    return not ticket.body.strip() and bool(_ONLY_REPLY_PREFIXES.match(ticket.subject))


def _close_empty() -> Decision:
    # Nothing to detect a language from; most customers write Dutch.
    return Decision(
        action="close", language="nl", category="other", priority="P4",
        reason="Empty ticket: no subject and no body to act on.", confidence=0.95,
    )


def _handoff_after_failure(failure: str, detected: str) -> Decision:
    """The model refused or returned nothing usable. A person reads the ticket instead."""
    return Decision(
        action="handoff", language=detected if detected != "unknown" else "nl",
        category="other", priority="P3", confidence=0.3,
        reason=f"Automatic triage failed ({failure}), so a human must read this ticket. Priority not assessed.",
        handoff_note=f"Automatic triage failed ({failure}). Please read the original ticket; nothing was sent to the customer.",
    )


def _without_payment_details(decision: Decision) -> Decision:
    """Last line of defence: nothing we post may repeat a card number or IBAN."""
    def clean(text: str | None) -> str | None:
        return redact_payment(text)[0] if text else text

    return decision.model_copy(update={
        "reason": clean(decision.reason),
        "draft_reply": clean(decision.draft_reply),
        "handoff_note": clean(decision.handoff_note),
    })
