import pytest

from agent.models import Ticket
from agent.redact import contains_payment, redact_payment, redact_ticket
from tests.fakes import CARD, IBAN


@pytest.mark.parametrize("text, placeholder", [
    (f"mijn kaart is {CARD}.", "[CARD]"),
    ("kaart 5555-5555-5555-4444", "[CARD]"),
    (f"rekening {IBAN} graag", "[IBAN]"),
    ("rekening NL91ABNA0417164300", "[IBAN]"),
    ("cvv: 987", "[CVC]"),
    ("CVC 4821", "[CVC]"),
    (f"{CARD}, exp 11/29", "[EXP]"),
    ("kaart 5555.5555.5555.4444", "[CARD]"),
    ("kaart 5555\u00a05555\u00a05555\u00a04444", "[CARD]"),
    ("kaart 4000 1234 5678 9010", "[CARD]"),  # a typo breaks the checksum, but 4-4-4-4 is still a card
    ("iban nl91abna0417164300 graag", "[IBAN]"),
    ("NL91-ABNA-0417-1643-00", "[IBAN]"),
    ("NL91.ABNA.0417.1643.00.", "[IBAN]"),
    ("ref AB12 NL91ABNA0417164300", "[IBAN]"),  # a false start right before the real one
    ("CVC code: 4821", "[CVC]"),
    ("veiligheidscode 482", "[CVC]"),
    ("security code is 482", "[CVC]"),
    (f"kaart {CARD} geldig tot 03/31", "[EXP]"),
    (f"{CARD} exp 03-31", "[EXP]"),
    ("cvv2 staat achterop de kaart: 482", "[CVC]"),
    ("5555  5555  5555  4444", "[CARD]"),
    ("5555/5555/5555/4444", "[CARD]"),
    ("Amex 3782 822463 10005", "[CARD]"),
    ("twee rekeningen: NL91ABNA0417164300 en BE68539007547034", "[IBAN] en [IBAN]"),
])
def test_payment_details_are_replaced(text, placeholder):
    redacted, found = redact_payment(text)
    assert found and placeholder in redacted


def test_card_number_itself_is_gone():
    redacted, _ = redact_payment(f"Card {CARD}, exp 11/29, cvv 987")
    assert "5555" not in redacted and "987" not in redacted and "11/29" not in redacted


@pytest.mark.parametrize("text", [
    "kaart 5555555555554444 09/28",  # followed by an expiry date
    "kaart 5555555555554444 482",  # followed by the security code
    "klant 12 5555555555554444",  # preceded by another number
    "5555555555554444\n12 medewerkers",
    "Amex 3782 822463 10005 09/28",
])
def test_card_next_to_other_numbers_is_still_redacted(text):
    redacted, found = redact_payment(text)
    assert found and "[CARD]" in redacted and "5555" not in redacted and "3782" not in redacted


def test_iban_followed_by_a_word_keeps_the_word():
    redacted, _ = redact_payment(f"van {IBAN} afschrijven")
    assert redacted == "van [IBAN] afschrijven"


@pytest.mark.parametrize("text", [
    "De link is binnen 60 minuten geldig.",
    "Het kost € 1.250,00 per jaar.",
    "Klantnummer 1234567890123456",  # 16 digits in one run, and the card checksum does not fit
    "De export 10/25 klopt niet.",  # a pay period, not an expiry date
    "De offerte is geldig tot 12/26. Hoe upgrade ik?",  # a date without a card is not an expiry
    "Ons nieuwe btw-nummer is NL869073662B02 is het nummer per 1 januari 2027.",  # a VAT number is not an IBAN
    "Onze jaren 2021 2022 2023 2024 en 2025 staan erin.",
    "Uren per week: 38 40 36 40 32 40 38 40.",
    "Bel 06 12345678 of 020 1234567.",
    "Wij hebben 12 medewerkers op 3 locaties sinds 2019 en 1.500 uur per maand.",
    "Open van 09:30 tot 17:45 op 11/12",
    "We hebben 25 medewerkers op iOS 16.",
])
def test_ordinary_numbers_are_left_alone(text):
    assert redact_payment(text) == (text, False)
    assert not contains_payment(text)


def test_ticket_subject_and_body_are_both_redacted():
    rt = redact_ticket(Ticket(id="X1", subject=f"Kaart {CARD}", body="Zie onderwerp."))
    assert rt.had_payment and "[CARD]" in rt.subject and rt.body == "Zie onderwerp."
