from decimal import Decimal

from agent.faq import find_euro_amounts, find_urls, load_kb
from agent.rules import ALWAYS_HUMAN, RuleFlag, definitions_text
from typing import get_args

kb = load_kb()


def test_all_twelve_entries_are_parsed():
    assert list(kb.entries) == [f"F{n:02d}" for n in range(1, 13)]
    assert kb.entries["F01"].title == "Forgotten password"
    assert "---" not in kb.entries["F12"].text


def test_allow_lists_come_from_the_help_centre():
    assert {"roadmap.roosterly.nl", "status.roosterly.nl", "app.roosterly.nl/login", "loket.nl"} <= kb.urls
    assert kb.euro_amounts == {Decimal(4), Decimal(6)}
    assert "always go to a human" in kb.rules_text


def test_urls_and_amounts_are_normalised():
    assert find_urls("Kijk op https://Status.roosterly.nl/. Klaar.Volgende zin") == {"status.roosterly.nl"}
    assert find_urls("www.roosterly.nl en hulp@voorbeeld.com") == {"roosterly.nl", "voorbeeld.com"}
    assert find_euro_amounts("€150, € 1.250,00, 72 euro, 90,- euro en EUR 4.50") == {Decimal(150), Decimal(1250), Decimal(72), Decimal(90), Decimal("4.5")}


def test_ordinary_words_are_not_mistaken_for_addresses_or_prices():
    assert find_urls("Prijzen zijn excl.btw. Open het bestand uren.csv. Het is geïnstalleerd.daarna werkt het.") == set()
    assert find_euro_amounts("De kleur 5 is anders en de directeur 2 keer gebeld. iOS 16 en hoger.") == set()


def test_every_rule_flag_has_a_rule():
    assert set(get_args(RuleFlag)) == set(ALWAYS_HUMAN)
    assert all(flag in definitions_text() for flag in ALWAYS_HUMAN)
