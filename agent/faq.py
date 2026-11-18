"""The help centre (faq.md) is the only knowledge the agent may answer from."""

import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

_ENTRY_HEADING = re.compile(r"^## (F\d+) · (.+)$", re.MULTILINE)
_RULES_HEADING = "## The client's rules for the support team"
FAQ_PATH = Path(__file__).parent.parent / "faq.md"

_URL = re.compile(r"(https?://|www\.)?((?:[A-Za-z0-9-]+\.)+([A-Za-z]{2,})(/[^\s()\[\]<>\"'“”‘’*]*)?)")
# Without "http", "www" or a path, a dotted word only counts as an address if it ends in a real
# top-level domain. The list is IANA's official one (agent/tlds.txt, refresh it from
# https://data.iana.org/TLD/tlds-alpha-by-domain.txt), so ".ru" or ".support" is caught while
# "excl.btw", "uren.csv" and a missing space after a full stop are not mistaken for links.
_TLD_FILE = Path(__file__).parent / "tlds.txt"
_ADDRESS_ENDINGS = frozenset(
    line.strip().lower() for line in _TLD_FILE.read_text(encoding="utf-8").splitlines() if line and line[0] != "#"
)
_EURO_AMOUNT = re.compile(  # "€6", "EUR 6", "euro 6" and "6 €", "6,- euro", "6 euros"
    r"(?:€|\b(?:EUR|euro))\s{0,2}(\d[\d.,]*)|(\d[\d.,]*)-?\s?(?:€|(?:euros?|EUR)\b)", re.IGNORECASE
)


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


def find_urls(text: str) -> set[str]:
    """Web addresses in the text, without scheme or "www", lowercased: "app.roosterly.nl/login"."""
    return {
        address.rstrip("/.,;:!?").lower()
        for prefix, address, ending, path in _URL.findall(text)
        if prefix or path or ending.lower() in _ADDRESS_ENDINGS
    }


def parse_amount(number: str) -> Decimal | None:
    """Read a Dutch or English money number: '1.250,00', '1,250.00', '150,00', '6' and '4.50' all work."""
    number = number.strip(".,")
    if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", number):  # English thousands
        number = number.replace(",", "")
    elif re.fullmatch(r"\d{1,3}(\.\d{3})+(,\d+)?", number) or "," in number:  # Dutch thousands or decimals
        number = number.replace(".", "").replace(",", ".")
    try:
        return Decimal(number)
    except ArithmeticError:
        return None


def find_euro_amounts(text: str) -> set[Decimal]:
    found = (parse_amount(a or b) for a, b in _EURO_AMOUNT.findall(text))
    return {amount.normalize() for amount in found if amount is not None}


def load_kb(path: str | Path = FAQ_PATH) -> KnowledgeBase:
    text = Path(path).read_text(encoding="utf-8")
    faq_part, found_rules, rules_part = text.partition(_RULES_HEADING)
    if not found_rules or not rules_part.strip():
        raise ValueError(f"{path}: no client rules found under '{_RULES_HEADING}'")
    headings = list(_ENTRY_HEADING.finditer(faq_part))
    if not headings:
        raise ValueError(f"{path}: no help centre entries found (expected headings like '## F01 · Title')")
    entries = {}
    for heading, next_heading in zip(headings, headings[1:] + [None], strict=True):
        end = next_heading.start() if next_heading else len(faq_part)
        body = faq_part[heading.end() : end].strip().removesuffix("---").strip()
        entries[heading.group(1)] = FaqEntry(heading.group(1), heading.group(2).strip(), " ".join(body.split()))
    faq_text = " ".join(entry.text for entry in entries.values())
    urls = find_urls(faq_text)
    return KnowledgeBase(
        entries=entries,
        rules_text=rules_part.strip(),
        urls=frozenset(
            urls | {url.split("/")[0] for url in urls}
        ),  # "app.roosterly.nl/login" also allows the bare site
        euro_amounts=frozenset(find_euro_amounts(faq_text)),
    )
