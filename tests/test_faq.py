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
    assert kb.urls == {
        "roadmap.roosterly.nl",
        "status.roosterly.nl",
        "app.roosterly.nl/login",
        "app.roosterly.nl",
        "loket.nl",
    }
    assert kb.euro_amounts == {Decimal(4), Decimal(6)}
    assert "always go to a human" in kb.rules_text


def test_urls_and_amounts_are_normalised():
    assert find_urls("Kijk op https://Status.roosterly.nl/. Klaar.Volgende zin") == {"status.roosterly.nl"}
    assert find_urls("www.roosterly.nl en hulp@voorbeeld.com") == {"roosterly.nl", "voorbeeld.com"}
    assert find_urls("“app.roosterly.nl/login” of **STATUS.ROOSTERLY.NL** of [x](roadmap.roosterly.nl)") == {
        "app.roosterly.nl/login",
        "status.roosterly.nl",
        "roadmap.roosterly.nl",
    }
    assert find_urls("ga naar roosterly.help/reset of bit.ly/x") == {"roosterly.help/reset", "bit.ly/x"}
    assert find_euro_amounts("€150, € 1.250,00, 72 euro, 90,- euro en EUR 4.50") == {
        Decimal(150),
        Decimal(1250),
        Decimal(72),
        Decimal(90),
        Decimal("4.5"),
    }
    assert find_euro_amounts("999 €, 998€, 997 euros, euro 996, €  995, €1,200 and €6,000.50") == {
        Decimal(n) for n in ("999", "998", "997", "996", "995", "1200", "6000.5")
    }


def test_ordinary_words_are_not_mistaken_for_addresses_or_prices():
    assert (
        find_urls("Prijzen zijn excl.btw. Open het bestand uren.csv. Het is geïnstalleerd.daarna werkt het.") == set()
    )
    assert find_euro_amounts("De kleur 5 is anders en de directeur 2 keer gebeld. iOS 16 en hoger.") == set()


def test_every_rule_flag_has_a_rule():
    assert set(get_args(RuleFlag)) == set(ALWAYS_HUMAN)
    assert all(flag in definitions_text() for flag in ALWAYS_HUMAN)


# --- review fixes: no hand-picked lists where an official one exists ----------------------


def test_any_official_domain_ending_counts_as_an_address():
    assert find_urls("Ga naar roosterly-hulp.ru of login.roosterly.support") == {
        "roosterly-hulp.ru",
        "login.roosterly.support",
    }


def test_the_clients_rules_are_parsed_from_the_help_centre():
    assert "Never repeat payment details" in kb.rules_text
    assert "always go to a human" in kb.rules_text


def test_a_help_centre_without_entries_or_rules_is_refused(tmp_path):
    import pytest

    no_rules = tmp_path / "faq.md"
    no_rules.write_text("## F01 · Title\nSome text.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rules"):
        load_kb(no_rules)
    no_entries = tmp_path / "empty.md"
    no_entries.write_text("## The client's rules for the support team\n- Be nice.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="entries"):
        load_kb(no_entries)
