"""Step 2a: deterministic checks on a draft reply. No model involved.

Each check answers one question that does not require reading Dutch: is the draft in the
right language, does it only use addresses and prices from the help centre, does it avoid
promises, and is every answer backed by a sentence quoted from the help centre?
"""

import re
from decimal import Decimal

from agent.decide import Request
from agent.faq import KnowledgeBase, find_euro_amounts, find_urls
from agent.language import detect_language
from agent.models import RedactedTicket
from agent.redact import contains_payment

MIN_QUOTE_LENGTH = 12  # long enough to be evidence, short enough for "We read every idea."

_PERCENT = re.compile(r"%|\b(procent|prozent|percent)\b", re.IGNORECASE)
_PROMISE = re.compile(
    r"korting|gratis|terugbetal|terugstort|vergoed|belov|garand|garant"
    r"|refund|discount|free of charge|guarantee|compensat"
    r"|rabatt|erstatt|kostenlos|entschädig",
    re.IGNORECASE,
)
# A response time is a promise. Minutes are left out: the help centre itself says "within 10 minutes".
_TIME_PROMISE = re.compile(
    r"\bbinnen (\d+|een|één|twee|drie|enkele) (uur|uren|dag|dagen|werkdag|werkdagen|week|weken)\b"
    r"|\bwithin (\d+|an?|one|two|three|a few) (hours?|days?|working days?|business days?|weeks?)\b"
    r"|\binnerhalb (von )?(\d+|einer|eines|einem|zwei|drei|weniger) "
    r"(stunden?|tag(e|en|es)?|werktag(e|en|es)?|wochen?)\b",
    re.IGNORECASE,
)


def check_reply(
    draft: str, language: str, requests: list[Request], rt: RedactedTicket, kb: KnowledgeBase
) -> list[str]:
    """Return the problems found in a draft reply. An empty list means it passed."""
    if not draft.strip():
        return ["empty_draft"]
    problems = set()
    if detect_language(draft) not in (language, "unknown"):
        problems.add("draft_in_wrong_language")
    if find_urls(draft) - kb.urls:
        problems.add("address_not_in_help_centre")
    if _PERCENT.search(draft):
        problems.add("percentage_in_draft")
    if find_euro_amounts(draft) - _allowed_amounts(rt, kb):
        problems.add("amount_not_in_help_centre")
    if _PROMISE.search(draft) or _TIME_PROMISE.search(draft):
        problems.add("promise_in_draft")
    if contains_payment(draft):
        problems.add("payment_details_in_draft")
    if not requests:
        problems.add("no_request_found")
    problems.update(problem for request in requests if (problem := _quote_problem(request, kb)))
    return sorted(problems)


def check_note(note: str) -> list[str]:
    """A handoff note must be in English and free of payment details."""
    problems = []
    if detect_language(note) not in ("en", "unknown"):
        problems.append("note_not_english")
    if contains_payment(note):
        problems.append("payment_details_in_note")
    return problems


def _allowed_amounts(rt: RedactedTicket, kb: KnowledgeBase) -> set[Decimal]:
    """Help centre prices, plus a price times a number the customer gave (25 employees x €6)."""
    counts = {Decimal(number) for number in re.findall(r"\b\d{1,4}\b", f"{rt.subject} {rt.body}")}
    products = {(count * price).normalize() for count in counts for price in kb.euro_amounts}
    return set(kb.euro_amounts) | products


def _quote_problem(request: Request, kb: KnowledgeBase) -> str | None:
    """Every request needs a sentence that really is in the help centre entry it names."""
    entry_id = re.search(r"F\d{2}", (request.faq_id or "").upper())  # "f10" and "F10 · Ideas" both mean F10
    entry = kb.entries.get(entry_id.group()) if entry_id else None
    if entry is None or not request.faq_quote:
        return "request_without_faq_answer"
    quote = _normalise(request.faq_quote)
    if len(quote) < MIN_QUOTE_LENGTH or quote not in _normalise(entry.text):
        return "faq_quote_not_in_help_centre"
    return None


def _normalise(text: str) -> str:
    """Make quote matching forgiving about typography, never about wording."""
    text = re.sub(r"\s*(->|=>|>|→)\s*", " → ", text)
    text = re.sub(r"[“”„]", '"', text)
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"€\s+", "€", text)
    return " ".join(text.lower().split()).rstrip(".,;:!?")
