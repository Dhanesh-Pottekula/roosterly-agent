"""Remove payment details from text before it reaches the LLM, a reply, a note or a log."""

import re

from agent.models import RedactedTicket, Ticket

_CARD = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}\b")
_CVC = re.compile(r"\b(CVC|CVV2?)\b\W{0,3}\d{3,4}\b", re.IGNORECASE)
_EXPIRY = r"\b(?:0[1-9]|1[0-2])\s?/\s?(?:\d{4}|\d{2})\b"
_EXPIRY_AFTER_LABEL = re.compile(rf"\b(geldig tot|exp\w*|valid (?:un)?til|gültig bis)\b(\W{{0,3}}){_EXPIRY}", re.IGNORECASE)
_EXPIRY_NEAR_CARD = re.compile(rf"(\[CARD\].{{0,30}}?){_EXPIRY}", re.DOTALL)


def _luhn_valid(digits: str) -> bool:
    total = 0
    for position, char in enumerate(reversed(digits)):
        digit = int(char)
        if position % 2 == 1:
            digit = digit * 2 - 9 if digit > 4 else digit * 2
        total += digit
    return total % 10 == 0


def _iban_valid(iban: str) -> bool:
    if not 15 <= len(iban) <= 34:
        return False
    rearranged = iban[4:] + iban[:4]
    return int("".join(str(int(char, 36)) for char in rearranged)) % 97 == 1


def _replace_card(match: re.Match) -> str:
    digits = re.sub(r"\D", "", match.group())
    return "[CARD]" if _luhn_valid(digits) else match.group()


def _replace_iban(match: re.Match) -> str:
    # The pattern can run on into the next word, so try shorter candidates until one is valid.
    text = match.group()
    for end in range(len(text), 14, -1):
        candidate = text[:end]
        if not candidate.endswith(" ") and _iban_valid(candidate.replace(" ", "")):
            return "[IBAN]" + text[end:]
    return text


def redact_payment(text: str) -> tuple[str, bool]:
    """Return the text with payment details replaced by placeholders, and whether any were found."""
    redacted = _CARD.sub(_replace_card, text)
    redacted = _IBAN.sub(_replace_iban, redacted)
    redacted = _CVC.sub(r"\1 [CVC]", redacted)
    redacted = _EXPIRY_AFTER_LABEL.sub(r"\1\2[EXP]", redacted)
    redacted = _EXPIRY_NEAR_CARD.sub(r"\1[EXP]", redacted)
    return redacted, redacted != text


def contains_payment(text: str) -> bool:
    return redact_payment(text)[1]


def redact_ticket(ticket: Ticket) -> RedactedTicket:
    subject, in_subject = redact_payment(ticket.subject)
    body, in_body = redact_payment(ticket.body)
    return RedactedTicket(ticket=ticket, subject=subject, body=body, had_payment=in_subject or in_body)
