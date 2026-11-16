"""The help centre (faq.md) is the only knowledge the agent may answer from."""

import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

_ENTRY_HEADING = re.compile(r"^## (F\d+) · (.+)$", re.MULTILINE)
_RULES_HEADING = "## The client's rules for the support team"
# A domain, with an optional path. The lowercase ending keeps "word.Next sentence" from matching.
URL = re.compile(r"(?:https?://)?((?:[A-Za-z0-9-]+\.)+[a-z]{2,}(?:/[^\s)\"']*)?)")
EURO_AMOUNT = re.compile(r"(?:€|EUR)\s?(\d[\d.,]*)|(\d[\d.,]*)\s?(?:euro|EUR)\b", re.IGNORECASE)


@dataclass(frozen=True)
class FaqEntry:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class KnowledgeBase:
    entries: dict[str, FaqEntry]
    rules_text: str
    urls: frozenset[str]  # every address the help centre mentions; a reply may use no others
    euro_amounts: frozenset[Decimal]  # every price the help centre mentions

    def as_prompt_text(self) -> str:
        return "\n\n".join(f"{e.id} · {e.title}\n{e.text}" for e in self.entries.values())


def normalise_url(url: str) -> str:
    return re.sub(r"^https?://", "", url.strip()).rstrip("/.,;:!?").lower()


def find_urls(text: str) -> set[str]:
    return {normalise_url(match.group(1)) for match in URL.finditer(text)}


def parse_amount(number: str) -> Decimal | None:
    """Read a Dutch or English money number: '1.250,00', '150,00', '6' and '4.50' all work."""
    number = number.strip(".,")
    if re.fullmatch(r"\d{1,3}(\.\d{3})+(,\d+)?", number) or "," in number:
        number = number.replace(".", "").replace(",", ".")
    try:
        return Decimal(number)
    except ArithmeticError:
        return None


def find_euro_amounts(text: str) -> set[Decimal]:
    found = (parse_amount(a or b) for a, b in EURO_AMOUNT.findall(text))
    return {amount.normalize() for amount in found if amount is not None}


def load_kb(path: str | Path = "faq.md") -> KnowledgeBase:
    text = Path(path).read_text(encoding="utf-8")
    faq_part, _, rules_part = text.partition(_RULES_HEADING)
    headings = list(_ENTRY_HEADING.finditer(faq_part))
    entries = {}
    for heading, next_heading in zip(headings, headings[1:] + [None]):
        end = next_heading.start() if next_heading else len(faq_part)
        body = faq_part[heading.end() : end].strip().removesuffix("---").strip()
        entries[heading.group(1)] = FaqEntry(heading.group(1), heading.group(2).strip(), " ".join(body.split()))
    faq_text = " ".join(entry.text for entry in entries.values())
    return KnowledgeBase(
        entries=entries,
        rules_text=rules_part.strip(),
        urls=frozenset(find_urls(faq_text)),
        euro_amounts=frozenset(find_euro_amounts(faq_text)),
    )
