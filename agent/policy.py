"""Step 3: the policy gate. The LLM proposes, this code decides.

Plain, deterministic rules turn a proposal into the decision we post. A rule can only move
a ticket towards a human: nothing here turns a handoff into a reply. Read `choose_action`
from top to bottom; the first rule that matches wins.
"""

from dataclasses import dataclass

from agent.decide import Proposal
from agent.models import Action, Decision, Language, Priority, RedactedTicket
from agent.rules import ALWAYS_HUMAN, RuleFlag
from agent.verify import Verdict, check_note

CONFIDENCE_CLOSE = 0.95
CONFIDENCE_REPLY = 0.9
CONFIDENCE_RULE_HANDOFF = 0.9  # a client rule or a hard fact decided it
CONFIDENCE_UNSURE_HANDOFF = 0.6  # we handed off because we could not be sure

# Handoff reasons that mean "unsure" get an explanation in the posted reason.
EXPLANATIONS = {
    "unsupported_language": "The ticket is not clearly in Dutch, English or German.",
    "payment_details_in_ticket": "The ticket contains payment details (redacted), so a human handles it.",
    "possible_manipulation": "The ticket contains instructions aimed at the AI, so its urgency is not trusted.",
    "urgent_or_blocking": "An urgent problem or a blocking bug needs a person, not a help centre answer.",
    "language_disagreement": "The language of the ticket is unclear.",
    "not_answered_by_help_centre": "The help centre does not fully answer this ticket.",
}


@dataclass(frozen=True)
class PolicyResult:
    decision: Decision
    handoff_reasons: list[str]


def posted_language(proposed: str, detected: str) -> Language | None:
    """The language we post: the detector's when it is confident, otherwise the proposer's."""
    if detected != "unknown":
        return detected
    return proposed if proposed != "other" else None


def posted_priority(p: Proposal, flags: list[RuleFlag], manipulated: bool, had_payment: bool) -> Priority:
    """Priority is settled before the action, so no rule below can be talked into a higher one."""
    if flags:
        priority = min(ALWAYS_HUMAN[flag].priority for flag in flags)  # "P1" sorts before "P2"
    elif manipulated:
        priority = "P3"
    else:
        priority = p.priority
    return min(priority, "P2") if had_payment else priority


def choose_action(
    rt: RedactedTicket,
    p: Proposal,
    flags: list[RuleFlag],
    language: Language | None,
    detected_language: str,
    priority: Priority,
    manipulated: bool,
    reply_problems: list[str],
    verdict: Verdict | None,
) -> tuple[Action, list[str]]:
    """Return the action and, for a handoff, the reasons. First match wins."""
    if p.is_spam and not (flags or rt.had_payment or p.contains_instructions_to_ai):
        return "close", []
    if language is None:
        return "handoff", ["unsupported_language"]
    if flags:
        return "handoff", [f"rule:{flag}" for flag in flags]
    if rt.had_payment:
        return "handoff", ["payment_details_in_ticket"]
    if manipulated:
        return "handoff", ["possible_manipulation"]
    if priority == "P1" or (p.category == "bug" and priority == "P2"):
        return "handoff", ["urgent_or_blocking"]
    if detected_language not in ("unknown", p.language):
        return "handoff", ["language_disagreement"]
    if p.proposed_action != "reply":
        return "handoff", ["not_answered_by_help_centre"]
    problems = reply_problems + (verdict.problems(language) if verdict else [])
    if problems or verdict is None:
        return "handoff", problems or ["reply_not_verified"]
    return "reply", []


def apply_policy(
    rt: RedactedTicket,
    p: Proposal,
    detected_language: str,
    reply_problems: list[str],
    verdict: Verdict | None,
) -> PolicyResult:
    flags = sorted(set(p.rule_flags) | set(verdict.always_human if verdict else []))
    manipulated = p.contains_instructions_to_ai and not flags and p.priority in ("P1", "P2")
    language = posted_language(p.language, detected_language)
    priority = posted_priority(p, flags, manipulated, rt.had_payment)
    action, reasons = choose_action(
        rt, p, flags, language, detected_language, priority, manipulated, reply_problems, verdict
    )

    if action == "close":
        decision = Decision(
            action="close", language=language or "nl", category="other", priority="P4",
            reason="Spam: unsolicited message with no support request.", confidence=CONFIDENCE_CLOSE,
        )
    elif action == "reply":
        faq_ids = ", ".join(sorted({request.faq_id for request in p.requests if request.faq_id}))
        decision = Decision(
            action="reply", language=language, category=p.category, priority=priority,
            reason=f"{p.priority_reason.strip()} Answered from help centre entry {faq_ids}.",
            draft_reply=p.draft_reply, confidence=CONFIDENCE_REPLY,
        )
    else:
        rule_decided = bool(flags) or rt.had_payment or reasons == ["urgent_or_blocking"]
        decision = Decision(
            action="handoff", language=language or "en", category=p.category, priority=priority,
            reason=build_handoff_reason(p, flags, reasons, priority),
            handoff_note=build_handoff_note(rt, p, reasons),
            confidence=CONFIDENCE_RULE_HANDOFF if rule_decided else CONFIDENCE_UNSURE_HANDOFF,
        )
    return PolicyResult(decision, reasons)


def build_handoff_reason(p: Proposal, flags: list[RuleFlag], reasons: list[str], priority: Priority) -> str:
    """One or two sentences: why a human, and why this priority."""
    if flags:
        labels = " and ".join(ALWAYS_HUMAN[flag].label for flag in flags)
        why_human = f"Client rule: {labels} always goes to a human."
    else:
        why_human = EXPLANATIONS.get(reasons[0], f"The drafted reply failed a check ({reasons[0]}), so a human answers.")
    if priority == p.priority:
        return f"{why_human} {p.priority_reason.strip()}"
    return f"{why_human} Priority set to {priority} by policy."


def build_handoff_note(rt: RedactedTicket, p: Proposal, reasons: list[str]) -> str:
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
