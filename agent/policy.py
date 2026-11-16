"""Step 3: the policy gate. The LLM proposes, this code decides.

Plain, deterministic rules turn a proposal into the decision we post. A rule can only move
a ticket towards a human: nothing here turns a handoff into a reply. To follow a decision,
read `choose_action` from top to bottom; the first rule that matches wins.
"""

from dataclasses import dataclass

from agent.checks import check_note
from agent.decide import Proposal
from agent.models import Action, Decision, Language, Priority, RedactedTicket
from agent.rules import ALWAYS_HUMAN, RuleFlag
from agent.verify import Verdict

CONFIDENCE_CLOSE = 0.95
CONFIDENCE_REPLY = 0.9
CONFIDENCE_RULE_HANDOFF = 0.9  # a client rule or a hard fact decided it
CONFIDENCE_UNSURE_HANDOFF = 0.6  # handed off because we could not be sure

# Why a human, in words, for each handoff reason that is not a client rule.
EXPLANATIONS = {
    "unsupported_language": "The ticket is not clearly in Dutch, English or German.",
    "payment_details_in_ticket": "The ticket contains payment details (redacted), so a human handles it.",
    "possible_manipulation": "The ticket contains instructions aimed at the AI, so its urgency is not trusted.",
    "urgent_or_blocking": "An urgent problem or a blocking bug needs a person, not a help centre answer.",
    "language_disagreement": "The language of the ticket is unclear.",
    "not_answered_by_help_centre": "The help centre does not fully answer this ticket.",
}


@dataclass(frozen=True)
class Facts:
    """What we know about a ticket before choosing an action. Settled first, so no rule can change it."""

    flags: list[RuleFlag]  # always-human rules, from the proposer or the verifier
    language: Language | None  # None: not clearly Dutch, English or German
    priority: Priority
    manipulated: bool  # text aimed at the AI plus a high priority that no rule explains


@dataclass(frozen=True)
class PolicyResult:
    decision: Decision
    handoff_reasons: list[str]


def apply_policy(
    rt: RedactedTicket,
    p: Proposal,
    detected_language: str,
    reply_problems: list[str],
    verdict: Verdict | None,
) -> PolicyResult:
    flags = sorted(set(p.rule_flags) | set(verdict.always_human if verdict else []))
    manipulated = p.contains_instructions_to_ai and not flags and p.priority in ("P1", "P2")
    facts = Facts(
        flags=flags,
        language=posted_language(p.language, detected_language),
        priority=posted_priority(p, flags, manipulated, rt.had_payment),
        manipulated=manipulated,
    )
    action, reasons = choose_action(rt, p, facts, detected_language, reply_problems, verdict)
    builders = {"close": _close, "reply": _reply, "handoff": _handoff}
    return PolicyResult(builders[action](rt, p, facts, reasons), reasons)


def posted_language(proposed: str, detected: str) -> Language | None:
    """The language we post: the detector's when it is confident, otherwise the proposer's."""
    if detected != "unknown":
        return detected
    return proposed if proposed != "other" else None


def posted_priority(p: Proposal, flags: list[RuleFlag], manipulated: bool, had_payment: bool) -> Priority:
    """A client rule fixes the priority. Otherwise the proposer's stands, unless it was manipulated."""
    if flags:
        priority = min(ALWAYS_HUMAN[flag].priority for flag in flags)  # "P1" sorts before "P2"
    elif manipulated:
        priority = "P3"
    else:
        priority = p.priority
    return min(priority, "P2") if had_payment else priority  # money is involved: at least P2


def choose_action(
    rt: RedactedTicket,
    p: Proposal,
    facts: Facts,
    detected_language: str,
    reply_problems: list[str],
    verdict: Verdict | None,
) -> tuple[Action, list[str]]:
    """Return the action and, for a handoff, the reasons. First match wins."""
    something_else_going_on = facts.flags or rt.had_payment or p.contains_instructions_to_ai
    if p.is_spam and p.proposed_action == "close" and not something_else_going_on:
        return "close", []
    if facts.language is None:
        return "handoff", ["unsupported_language"]
    if facts.flags:
        return "handoff", [f"rule:{flag}" for flag in facts.flags]
    if rt.had_payment:
        return "handoff", ["payment_details_in_ticket"]
    if facts.manipulated:
        return "handoff", ["possible_manipulation"]
    if facts.priority == "P1" or (p.category == "bug" and facts.priority == "P2"):
        return "handoff", ["urgent_or_blocking"]
    if detected_language not in ("unknown", p.language):
        return "handoff", ["language_disagreement"]
    if p.proposed_action != "reply":
        return "handoff", ["not_answered_by_help_centre"]
    if verdict is None:  # a reply is never sent unverified
        return "handoff", reply_problems or ["reply_not_verified"]
    problems = reply_problems + verdict.problems(facts.language)
    if problems:
        return "handoff", problems
    return "reply", []


# --- building the decision ------------------------------------------------------------

def _close(rt: RedactedTicket, p: Proposal, facts: Facts, reasons: list[str]) -> Decision:
    return Decision(
        action="close", language=facts.language or "nl", category="other", priority="P4",
        reason="Spam: unsolicited message with no support request.", confidence=CONFIDENCE_CLOSE,
    )


def _reply(rt: RedactedTicket, p: Proposal, facts: Facts, reasons: list[str]) -> Decision:
    faq_ids = ", ".join(sorted({request.faq_id for request in p.requests if request.faq_id}))
    return Decision(
        action="reply", language=facts.language, category=p.category, priority=facts.priority,
        reason=f"{_sentence(p.priority_reason)} Answered from help centre entry {faq_ids}.",
        draft_reply=p.draft_reply, confidence=CONFIDENCE_REPLY,
    )


def _handoff(rt: RedactedTicket, p: Proposal, facts: Facts, reasons: list[str]) -> Decision:
    decided_by_rule = bool(facts.flags) or rt.had_payment or reasons == ["urgent_or_blocking"]
    return Decision(
        action="handoff", language=facts.language or "en", category=p.category, priority=facts.priority,
        reason=_handoff_reason(p, facts, reasons), handoff_note=_handoff_note(rt, p, reasons),
        confidence=CONFIDENCE_RULE_HANDOFF if decided_by_rule else CONFIDENCE_UNSURE_HANDOFF,
    )


def _handoff_reason(p: Proposal, facts: Facts, reasons: list[str]) -> str:
    """Two sentences: why a human, and why this priority."""
    if facts.flags:
        labels = " and ".join(ALWAYS_HUMAN[flag].label for flag in facts.flags)
        why_human = f"Client rule: {labels} always goes to a human."
    else:
        why_human = EXPLANATIONS.get(reasons[0], f"The drafted reply failed a check ({reasons[0]}), so a human answers.")
    if facts.priority == p.priority:
        return f"{why_human} {_sentence(p.priority_reason)}"
    return f"{why_human} Priority set to {facts.priority} by policy."


def _handoff_note(rt: RedactedTicket, p: Proposal, reasons: list[str]) -> str:
    """The proposer's English note, plus the facts the policy knows."""
    note = p.handoff_note.strip()
    if not note or check_note(note):
        note = f"Summary: {p.summary_en.strip()}"
    lines = [note, f"Plan: {rt.ticket.customer_plan or 'unknown'}. Channel: {rt.ticket.channel or 'unknown'}."]
    if rt.had_payment:
        lines.append("The customer sent payment details. They were redacted; ask for a secure channel.")
    if p.contains_instructions_to_ai:
        lines.append("The ticket contains text addressed to the AI. It was ignored.")
    if "unsupported_language" in reasons:
        lines.append("The ticket is not clearly in Dutch, English or German; check the original text.")
    lines.append(f"Handed off because: {', '.join(reasons)}.")
    return "\n".join(lines)


def _sentence(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else f"{text}."
