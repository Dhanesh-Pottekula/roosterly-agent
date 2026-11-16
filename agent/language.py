"""Which language is this text in? Used for tickets, draft replies and handoff notes."""

import re
from functools import cache
from typing import Literal

from lingua import Language, LanguageDetectorBuilder

DetectedLanguage = Literal["nl", "en", "de", "unknown"]
MIN_CONFIDENCE = 0.6

_CODES = {Language.DUTCH: "nl", Language.ENGLISH: "en", Language.GERMAN: "de"}
# Menu names, quotes and addresses stay Dutch in every reply, so they are removed before detecting.
_NOT_PROSE = re.compile(
    r"\([^)]*\)"  # (English meaning)
    r"|\"[^\"\n]*\"|[“„][^”“\n]*[”“]"  # "quoted text"
    r"|(?<!\w)['‘][^'’\n]*['’](?!\w)"  # 'quoted text', but not the apostrophe in zo'n or can't
    r"|(?:\S+\s+)?\S+(?:\s*(?:→|->)\s*\S+(?:\s+\S+)?)+"  # Menu → Sub menu
    r"|(?:https?://)?(?:[A-Za-z0-9-]+\.)+[a-z]{2,}\S*"  # addresses
)


@cache
def _detector():
    return LanguageDetectorBuilder.from_languages(*_CODES).build()


def detect_language(text: str) -> DetectedLanguage:
    """The language of the prose in `text`, or "unknown" when the detector is not confident."""
    prose = _NOT_PROSE.sub(" ", text).strip()
    if not prose:
        return "unknown"
    best = _detector().compute_language_confidence_values(prose)[0]
    return _CODES[best.language] if best.value >= MIN_CONFIDENCE else "unknown"
