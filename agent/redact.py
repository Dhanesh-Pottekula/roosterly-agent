"""Remove payment details from text before it reaches the LLM, a reply, a note or a log.

A card number or IBAN is recognised by its checksum, so ordinary numbers are left alone.
Limit, on purpose: an IBAN with a typo fails its checksum and is not recognised.
"""

import re

from agent.models import RedactedTicket, Ticket

_SEPARATORS = r"[\s.\-/]{1,3}"  # people type "5555 5555", "5555-5555" and "5555.5555"
_DIGIT_GROUPS = re.compile(rf"\b\d+(?:{_SEPARATORS}\d+)*\b")
_CARD_LENGTHS = range(13, 20)
_CARD_WORD = re.compile(r"kaart|card|visa|master|amex|karte|pinpas", re.IGNORECASE)
_FOUR_BY_FOUR = re.compile(rf"\b\d{{4}}(?:{_SEPARATORS}\d{{4}}){{3}}\b")

_IBAN_START = re.compile(r"\b[A-Z]{2}\d{2}(?=[\s.\-]?[A-Z0-9])", re.IGNORECASE)
# IBAN length per country, from the SWIFT IBAN registry (every country that issues IBANs).
# The mod-97 checksum below is what proves an IBAN; the length only says where it ends.
IBAN_LENGTHS = {
    **dict.fromkeys(["NO"], 15),
    **dict.fromkeys(["BE"], 16),
    **dict.fromkeys(["DK", "FI", "FK", "FO", "GL", "NL", "SD"], 18),
    **dict.fromkeys(["MK", "SI"], 19),
    **dict.fromkeys(["AT", "BA", "EE", "KZ", "LT", "LU", "MN", "XK"], 20),
    **dict.fromkeys(["CH", "HR", "LI", "LV"], 21),
    **dict.fromkeys(["BG", "BH", "CR", "DE", "GB", "GE", "IE", "ME", "RS", "VA"], 22),
    **dict.fromkeys(["AE", "GI", "IL", "IQ", "OM", "SO", "TL"], 23),
    **dict.fromkeys(["CZ", "ES", "MD", "PK", "RO", "SA", "SE", "SK", "TN", "VG"], 24),
    **dict.fromkeys(["LY", "PT", "ST"], 25),
    **dict.fromkeys(["IS", "TR"], 26),
    **dict.fromkeys(["BI", "DJ", "FR", "GR", "IT", "MC", "MR", "SM"], 27),
    **dict.fromkeys(["AL", "AZ", "BY", "CY", "DO", "GT", "HN", "HU", "LB", "NI", "PL", "SV"], 28),
    **dict.fromkeys(["BR", "EG", "PS", "QA", "UA"], 29),
    **dict.fromkeys(["JO", "KW", "MU", "YE"], 30),
    **dict.fromkeys(["MT", "SC"], 31),
    **dict.fromkeys(["LC"], 32),
    **dict.fromkeys(["RU"], 33),
}

_CVC = re.compile(
    r"\b(CVC|CVV2?|security code|veiligheidscode|beveiligingscode|sicherheitscode)\b(\D{0,30}?)\d{3,4}\b",
    re.IGNORECASE,
)
_EXPIRY_NEAR_CARD = re.compile(r"(\[CARD\].{0,40}?)\b(?:0[1-9]|1[0-2])\s?[/\-]\s?(?:\d{4}|\d{2})\b", re.DOTALL)


def redact_payment(text: str) -> tuple[str, bool]:
    """Return the text with payment details replaced by placeholders, and whether any were found."""
    redacted = _DIGIT_GROUPS.sub(_replace_cards, text)
    if _CARD_WORD.search(text):  # next to the word "card", 4-4-4-4 is a card even if a typo breaks the checksum
        redacted = _FOUR_BY_FOUR.sub("[CARD]", redacted)
    redacted = _replace_ibans(redacted)
    redacted = _CVC.sub(r"\1\2[CVC]", redacted)
    redacted = _EXPIRY_NEAR_CARD.sub(r"\1[EXP]", redacted)  # a date only counts as an expiry next to a card
    return redacted, redacted != text


def contains_payment(text: str) -> bool:
    return redact_payment(text)[1]


def redact_ticket(ticket: Ticket) -> RedactedTicket:
    subject, in_subject = redact_payment(ticket.subject)
    body, in_body = redact_payment(ticket.body)
    return RedactedTicket(ticket=ticket, subject=subject, body=body, had_payment=in_subject or in_body)


def _replace_cards(match: re.Match) -> str:
    """Replace every card number in a run of digit groups such as "12 5555 5555 5555 4444 09".

    The card may sit next to other numbers, so from each group we look for the longest
    stretch of whole groups that forms a valid card number.
    """
    parts = re.split(rf"({_SEPARATORS})", match.group())  # digits, separator, digits, ...
    groups = parts[0::2]
    result, index = [], 0
    while index < len(groups):
        end = _card_end(groups, index)
        result.append("[CARD]" if end else groups[index])
        index = end or index + 1
        if index < len(groups):
            result.append(parts[2 * index - 1])  # the separator before the next group
    return "".join(result)


def _card_end(groups: list[str], start: int) -> int | None:
    """The index after the last group of a card number starting at `start`, or None.

    A card is one unbroken number, or blocks of 4 to 6 digits (4-4-4-4, or 4-6-5 for Amex).
    Without that rule, one list of small numbers in ten would pass the checksum by chance.
    """
    digits, best = "", None
    for end in range(start, len(groups)):
        digits += groups[end]
        in_blocks = end == start or all(4 <= len(group) <= 6 for group in groups[start : end + 1])
        if len(digits) > _CARD_LENGTHS[-1] or not in_blocks:
            break
        if len(digits) in _CARD_LENGTHS and _luhn_valid(digits):
            best = end + 1
    return best


def _luhn_valid(digits: str) -> bool:
    total = 0
    for position, char in enumerate(reversed(digits)):
        digit = int(char)
        if position % 2 == 1:
            digit = digit * 2 - 9 if digit > 4 else digit * 2
        total += digit
    return total % 10 == 0


def _replace_ibans(text: str) -> str:
    result, position = [], 0
    for start in _IBAN_START.finditer(text):
        end = _iban_end(text, start.start())
        if end and start.start() >= position:
            result.append(text[position : start.start()] + "[IBAN]")
            position = end
    return "".join(result) + text[position:]


def _iban_end(text: str, start: int) -> int | None:
    """Where the IBAN starting at `start` ends, or None if this is not a valid IBAN.

    An IBAN has a fixed length per country, so "NL..." is read for exactly 18 characters.
    That keeps a VAT number followed by ordinary words from being mistaken for one.
    """
    length = IBAN_LENGTHS.get(text[start : start + 2].upper())
    compact, end = "", start
    while length and end < len(text) and len(compact) < length:
        char = text[end]
        if char.isalnum() and char.isascii():
            compact += char.upper()
        elif char not in " .- ":
            return None
        end += 1
    complete = length and len(compact) == length and not text[end : end + 1].isalnum()
    return end if complete and _iban_valid(compact) else None


def _iban_valid(iban: str) -> bool:
    rearranged = iban[4:] + iban[:4]
    return int("".join(str(int(char, 36)) for char in rearranged)) % 97 == 1
