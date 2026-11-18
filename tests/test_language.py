"""Language detection: these tests gate the choice of detector."""

import pytest

from agent.language import detect_language


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "Gehen Sie als Manager zu Team → Medewerker toevoegen (Add employee) "
            "und klicken Sie auf Uitnodigen (Invite).",
            "de",
        ),
        ("Thanks, that worked.", "en"),
        ("Op welke dagen is jullie klantenservice bereikbaar?", "nl"),
        ("Wo finde ich die Übersicht meiner Stunden?", "de"),
        ("The label on the export screen reads 'Urenoverzicht downloaden' twice.", "en"),
        (
            "Go to Instellingen → Abonnement → Facturen (Settings → Subscription → Invoices) "
            "to download your invoices.",
            "en",
        ),
        ("Ik heb zo'n 20 medewerkers en de foto's van 's ochtends laden niet meer.", "nl"),
        ("I can't log in and my staff don't see next week's rota.", "en"),
        ("", "unknown"),
        ("→ (ok) 'x'", "unknown"),
    ],
)
def test_detect_language(text, expected):
    assert detect_language(text) == expected


def test_very_long_unbroken_text_does_not_stall_language_detection():
    import time

    started = time.monotonic()
    detect_language("x" * 40_000 + " → menu")
    assert time.monotonic() - started < 2
