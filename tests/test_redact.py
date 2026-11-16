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
    ("geldig tot 03/31", "[EXP]"),
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
    ("vervaldatum 03/31", "[EXP]"),
    ("expires 03/2031", "[EXP]"),
])
def test_payment_details_are_replaced(text, placeholder):
    redacted, found = redact_payment(text)
    assert found and placeholder in redacted


def test_card_number_itself_is_gone():
    redacted, _ = redact_payment(f"Card {CARD}, exp 11/29, cvv 987")
    assert "5555" not in redacted and "987" not in redacted and "11/29" not in redacted


def test_iban_followed_by_a_word_keeps_the_word():
    redacted, _ = redact_payment(f"van {IBAN} afschrijven")
    assert redacted == "van [IBAN] afschrijven"


@pytest.mark.parametrize("text", [
    "De link is binnen 60 minuten geldig.",
    "Het kost € 1.250,00 per jaar.",
    "Klantnummer 1234567890123456",  # 16 digits in one run, and the card checksum does not fit
    "De export 10/25 klopt niet.",  # a pay period, not an expiry date
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
