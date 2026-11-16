"""Remove payment details from text before it reaches the LLM, a reply, a note or a log.

When in doubt, redact: a wrongly hidden number costs one handoff, a leaked card number
breaks a client rule.
"""

import re

from agent.models import RedactedTicket, Ticket

_SEPARATOR = r"[\s.\-]"  # people type "5555 5555", "5555-5555" and "5555.5555"
_CARD = re.compile(rf"\b\d(?:{_SEPARATOR}?\d){{12,18}}\b")  # 13 to 19 digits; kept only if the checksum fits
_CARD_IN_GROUPS = re.compile(rf"\b\d{{4}}(?:{_SEPARATOR}\d{{4}}){{3}}\b")  # 4-4-4-4 is a card, even with a typo
_IBAN = re.compile(rf"\b[A-Z]{{2}}\d{{2}}(?:{_SEPARATOR}?[A-Z0-9]){{11,30}}\b", re.IGNORECASE)
_CVC = re.compile(
    r"\b(CVC|CVV2?|security code|veiligheidscode|beveiligingscode|sicherheitscode)\b(\D{0,12}?)\d{3,4}\b",
    re.IGNORECASE,
)
_EXPIRY = r"\b(?:0[1-9]|1[0-2])\s?/\s?(?:\d{4}|\d{2})\b"
_EXPIRY_AFTER_LABEL = re.compile(
    r"\b(geldig tot|vervaldatum|verloopt(?: op)?|exp(?:ir\w*)?|valid (?:un)?til|valid thru|gültig bis|ablaufdatum)"
    rf"(\W{{0,3}}){_EXPIRY}",
    re.IGNORECASE,
)
_EXPIRY_NEAR_CARD = re.compile(rf"(\[CARD\].{{0,30}}?){_EXPIRY}", re.DOTALL)
_MIN_IBAN_LENGTH = 15


def _luhn_valid(digits: str) -> bool:
    total = 0
    for position, char in enumerate(reversed(digits)):
        digit = int(char)
        if position % 2 == 1:
            digit = digit * 2 - 9 if digit > 4 else digit * 2
        total += digit
    return total % 10 == 0


def _iban_valid(iban: str) -> bool:
    rearranged = iban[4:] + iban[:4]
    return int("".join(str(int(char, 36)) for char in rearranged)) % 97 == 1


def _replace_card(match: re.Match) -> str:
    return "[CARD]" if _luhn_valid(re.sub(r"\D", "", match.group())) else match.group()


def _replace_iban(match: re.Match) -> str:
    """The pattern can run on into the next words, so shorten the candidate until the checksum fits."""
    text = match.group()
    for end in range(len(text), _MIN_IBAN_LENGTH - 1, -1):
        candidate = re.sub(_SEPARATOR, "", text[:end]).upper()
        if text[end - 1].isalnum() and len(candidate) >= _MIN_IBAN_LENGTH and _iban_valid(candidate):
            return "[IBAN]" + text[end:]
    return text[:4] + _IBAN.sub(_replace_iban, text[4:])  # a false start; look further on


def redact_payment(text: str) -> tuple[str, bool]:
    """Return the text with payment details replaced by placeholders, and whether any were found."""
    redacted = _CARD_IN_GROUPS.sub("[CARD]", text)
    redacted = _CARD.sub(_replace_card, redacted)
    redacted = _IBAN.sub(_replace_iban, redacted)
    redacted = _CVC.sub(r"\1\2[CVC]", redacted)
    redacted = _EXPIRY_AFTER_LABEL.sub(r"\1\2[EXP]", redacted)
    redacted = _EXPIRY_NEAR_CARD.sub(r"\1[EXP]", redacted)
    return redacted, redacted != text


def contains_payment(text: str) -> bool:
    return redact_payment(text)[1]


def redact_ticket(ticket: Ticket) -> RedactedTicket:
    subject, in_subject = redact_payment(ticket.subject)
    body, in_body = redact_payment(ticket.body)
    return RedactedTicket(ticket=ticket, subject=subject, body=body, had_payment=in_subject or in_body)
