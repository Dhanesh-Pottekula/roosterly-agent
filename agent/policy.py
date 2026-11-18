"""Step 3: the policy gate. The LLM proposes, this code decides.

Plain, deterministic rules turn a proposal into the decision we post. A rule can only move
a ticket towards a human: nothing here turns a handoff into a reply. To follow a decision,
read `choose_action` from top to bottom; the first rule that matches wins.
"""

import re
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


# Text aimed at the reader's AI rather than at the support team: "ignore your instructions",
# "note for the AI assistant". The model is asked to flag this too; this pattern catches the
# common forms even when the model was talked out of flagging them, in Dutch, English and German.
# It matches text *addressed to* an AI, never a mere mention: "I asked an AI tool" keeps its priority.
_AI = r"(ai|ki|assistant|assistent|chatbot|bot|model|taalmodel|language model|sprachmodell)\b"
_ARTICLE = r"(?:(?:the|de|het|die|den|das|der)\s+)?"
_ADDRESSES_THE_AI = re.compile(
    # "ignore your instructions", "negeer alle eerdere regels", "vergiss deine Anweisungen"
    r"\b(negeer|ignore|ignoriere|disregard|vergeet|forget|vergiss)\b.{0,40}"
    r"\b(instructi\w*|regels|rules|richtlijnen|prompt\w*|anweisung\w*|regeln)\b"
    # "note to the assistant", "notitie voor de AI-assistent", "Hinweis für die KI"
    rf"|\b(note|message|instruction|opmerking|notitie|bericht|hinweis|nachricht|anweisung)\s+"
    rf"(to|for|voor|aan|für|an)\s+{_ARTICLE}{_AI}"
    # a line or sentence that opens by addressing the AI: "To the AI: ...", "Liebe KI, ..."
    rf"|(?:^|[.!?]\s+)(to|aan|dear|beste|liebe|lieber|hey|hi|hallo)\s+{_ARTICLE}{_AI}"
    rf"|(?:^|[.!?]\s+)an\s+(die|den|das|der)\s+{_AI}"  # German "An die KI"; English "An AI ..." is not"
    # a fake system header: "SYSTEM MESSAGE:", "SYSTEEMBERICHT:", "Systemnachricht:"
    r"|\b(system\s?message|system\s?prompt|systeembericht|systemnachricht)\s*:",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class Facts:
    """What we know about a ticket before choosing an action. Settled first, so no rule can change it."""

    flags: list[RuleFlag]  # always-human rules, from the proposer or the verifier
    language: Language | None  # None: not clearly Dutch, English or German
    priority: Priority
    instructed: bool  # the ticket contains text aimed at the AI
    manipulated: bool  # ... and asks for a high priority that no client rule explains


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
    instructed = p.contains_instructions_to_ai or bool(_ADDRESSES_THE_AI.search(f"{rt.subject}\n{rt.body}"))
    manipulated = instructed and not flags and p.priority in ("P1", "P2")
    facts = Facts(
        flags=flags,
        language=posted_language(p.language, detected_language),
        priority=posted_priority(p, flags, manipulated, rt.had_payment),
        instructed=instructed,
        manipulated=manipulated,
    )
    action, reasons = choose_action(rt, p, facts, detected_language, reply_problems, verdict)
    if action == "close":
        decision = _close(facts)
    elif action == "reply":
        decision = _reply(p, facts)
    else:
        decision = _handoff(rt, p, facts, reasons)
    return PolicyResult(decision, reasons)


def posted_language(proposed: str, detected: str) -> Language | None:
    """The language we post: the detector's when it is confident, otherwise the proposer's.

    The detector only knows Dutch, English and German, so it calls Spanish "English". The
    proposer saying "other" therefore wins: None means the language is not one we support.
    """
    if proposed == "other":
        return None
    return detected if detected != "unknown" else proposed


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
    something_else_going_on = facts.flags or rt.had_payment or facts.instructed
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


def _close(facts: Facts) -> Decision:
    return Decision(
        action="close",
        language=facts.language or "nl",
        category="other",
        priority="P4",
        reason="Spam: unsolicited message with no support request.",
        confidence=CONFIDENCE_CLOSE,
    )


def _reply(p: Proposal, facts: Facts) -> Decision:
    faq_ids = ", ".join(sorted({request.faq_id for request in p.requests if request.faq_id}))
    return Decision(
        action="reply",
        language=facts.language,
        category=p.category,
        priority=facts.priority,
        reason=f"{_sentence(p.priority_reason)} Answered from help centre entry {faq_ids}.",
        draft_reply=p.draft_reply,
        confidence=CONFIDENCE_REPLY,
    )


def _handoff(rt: RedactedTicket, p: Proposal, facts: Facts, reasons: list[str]) -> Decision:
    decided_by_rule = bool(facts.flags) or rt.had_payment or reasons == ["urgent_or_blocking"]
    return Decision(
        action="handoff",
        language=facts.language or "en",
        category=p.category,
        priority=facts.priority,
        reason=_handoff_reason(p, facts, reasons),
        handoff_note=_handoff_note(rt, p, facts, reasons),
        confidence=CONFIDENCE_RULE_HANDOFF if decided_by_rule else CONFIDENCE_UNSURE_HANDOFF,
    )


def _handoff_reason(p: Proposal, facts: Facts, reasons: list[str]) -> str:
    """Two sentences: why a human, and why this priority."""
    if facts.flags:
        labels = " and ".join(ALWAYS_HUMAN[flag].label for flag in facts.flags)
        why_human = f"Client rule: {labels} always goes to a human."
    else:
        why_human = EXPLANATIONS.get(
            reasons[0], f"The drafted reply failed a check ({reasons[0]}), so a human answers."
        )
    if facts.priority == p.priority:
        return f"{why_human} {_sentence(p.priority_reason)}"
    return f"{why_human} Priority set to {facts.priority} by policy."


def _handoff_note(rt: RedactedTicket, p: Proposal, facts: Facts, reasons: list[str]) -> str:
    """The proposer's English note, plus the facts the policy knows."""
    note = p.handoff_note.strip()
    if not note or check_note(note):
        note = f"Summary: {p.summary_en.strip()}"
    plan_line = f"Plan: {rt.ticket.customer_plan or 'unknown'}. Channel: {rt.ticket.channel or 'unknown'}."
    plan = re.escape(rt.ticket.customer_plan or "unknown")
    already_there = re.search(rf"\bplan\s*:\s*{plan}\b", note, re.IGNORECASE)  # the model often writes it itself
    lines = [note] if already_there else [note, plan_line]
    if rt.had_payment:
        lines.append("The customer sent payment details. They were redacted; ask for a secure channel.")
    if facts.instructed:
        lines.append("The ticket contains text addressed to the AI. It was ignored.")
    if "unsupported_language" in reasons:
        lines.append("The ticket is not clearly in Dutch, English or German; check the original text.")
    lines.append(f"Handed off because: {', '.join(reasons)}.")
    return "\n".join(lines)


def _sentence(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else f"{text}."
