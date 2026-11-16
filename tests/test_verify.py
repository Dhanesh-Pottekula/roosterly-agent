import pytest

from agent.decide import Request
from agent.faq import load_kb
from agent.models import Ticket
from agent.redact import redact_ticket
from agent.rules import ALWAYS_HUMAN
from agent.checks import check_note, check_reply
from agent.language import detect_language
from agent.verify import build_verifier_prompt
from tests.fakes import CARD, DUTCH_DRAFT, F12_QUOTE, make_verdict

kb = load_kb()
F12 = [Request(description_en="Which Android version is needed", faq_id="F12", faq_quote=F12_QUOTE)]
F05 = [
    Request(
        description_en="Price of Plus",
        faq_id="F05",
        faq_quote="Plus: €6 per active employee per month, unlimited locations, API access and the payroll integrations.",
    )
]


def ticket(body="Op welke versie van Android draait jullie app?", subject="Vraag"):
    return redact_ticket(Ticket(id="X1", subject=subject, body=body, customer_plan="Basis"))


def problems(draft, language="nl", requests=F12, rt=None):
    return check_reply(draft, language, requests, rt or ticket(), kb)


# --- language detection: these gate the choice of detector ---------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "Gehen Sie als Manager zu Team → Medewerker toevoegen (Add employee) und klicken Sie auf Uitnodigen (Invite).",
            "de",
        ),
        ("Thanks, that worked.", "en"),
        ("Op welke dagen is jullie klantenservice bereikbaar?", "nl"),
        ("Wo finde ich die Übersicht meiner Stunden?", "de"),
        ("The label on the export screen reads 'Urenoverzicht downloaden' twice.", "en"),
        (
            "Go to Instellingen → Abonnement → Facturen (Settings → Subscription → Invoices) to download your invoices.",
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


# --- deterministic reply checks -------------------------------------------------------


def test_a_sound_dutch_draft_passes():
    assert problems(DUTCH_DRAFT) == []


def test_draft_in_the_wrong_language_fails():
    assert "draft_in_wrong_language" in problems(DUTCH_DRAFT, language="en")


@pytest.mark.parametrize(
    "extra, problem",
    [
        ("Kijk ook op www.example.com.", "address_not_in_help_centre"),
        ("Mail naar hulp@roosterly.nl.", "address_not_in_help_centre"),
        ("Log in op roosterly.nl.", "address_not_in_help_centre"),
        ("Dat kost 5,- euro per maand.", "amount_not_in_help_centre"),
        ("Je krijgt 20% minder.", "percentage_in_draft"),
        ("Je krijgt 50 procent minder.", "percentage_in_draft"),
        ("Dat kost €5 per maand.", "amount_not_in_help_centre"),
        ("Je krijgt korting.", "promise_in_draft"),
        ("De eerste maand is gratis.", "promise_in_draft"),
        ("We reageren binnen 24 uur.", "promise_in_draft"),
        ("We reageren binnen een week.", "promise_in_draft"),
        ("Dat kost 999 € per maand.", "amount_not_in_help_centre"),
        ("Reset het via roosterly.help/reset.", "address_not_in_help_centre"),
        (f"Je kaart {CARD} is aangepast.", "payment_details_in_draft"),
    ],
)
def test_unsafe_content_is_caught(extra, problem):
    assert problem in problems(f"{DUTCH_DRAFT} {extra}")


@pytest.mark.parametrize(
    "extra",
    [
        "Kijk op status.roosterly.nl voor storingen.",
        "Ga naar app.roosterly.nl/login.",
        "De koppeling met Loket.nl zit in Plus.",
        "Prijzen zijn excl.btw en je kunt de uren als uren.csv downloaden.",
        "Kies een andere kleur 5 keer.",
        "De link is 60 minuten geldig. Geen mail binnen 10 minuten? Kijk in je spam.",
    ],
)
def test_help_centre_facts_are_allowed(extra):
    assert problems(f"{DUTCH_DRAFT} {extra}") == []


def test_german_reply_may_mention_the_help_centre_time_limits():
    draft = "Guten Tag, der Link ist 60 Minuten gültig. Falls innerhalb von 10 Minuten keine E-Mail ankommt, prüfen Sie bitte Ihren Spam-Ordner."
    quote = [
        Request(description_en="Reset password", faq_id="F01", faq_quote="The reset link is valid for 60 minutes.")
    ]
    assert problems(draft, language="de", requests=quote) == []


def test_price_times_the_customers_own_number_is_allowed():
    rt = ticket("Wij groeien naar 15 mensen in dienst. Wat betalen we dan per maand met Plus?")
    assert (
        problems(
            "Hoi! Plus kost €6 per actieve medewerker per maand, dus 15 × €6 = €90 per maand, exclusief btw.",
            requests=F05,
            rt=rt,
        )
        == []
    )
    assert "amount_not_in_help_centre" in problems(
        "Hoi! Dat wordt dan €91 per maand, exclusief btw.", requests=F05, rt=rt
    )
    assert problems("Hoi! Dat wordt dan 90 euro per maand, exclusief btw.", requests=F05, rt=rt) == []


# --- every answer must quote the help centre ------------------------------------------


@pytest.mark.parametrize(
    "request_, problem",
    [
        (Request(description_en="x", faq_id=None, faq_quote=None), "request_without_faq_answer"),
        (Request(description_en="x", faq_id="F99", faq_quote=F12_QUOTE), "request_without_faq_answer"),
        (
            Request(description_en="x", faq_id="F12", faq_quote="Roosterly also integrates with every payroll system."),
            "faq_quote_not_in_help_centre",
        ),
        (Request(description_en="x", faq_id="F05", faq_quote="Plus"), "faq_quote_not_in_help_centre"),
        (Request(description_en="x", faq_id="F05", faq_quote="Plus plan."), "faq_quote_not_in_help_centre"),
        (Request(description_en="x", faq_id="F05", faq_quote=F12_QUOTE), "faq_quote_not_in_help_centre"),
    ],
)
def test_missing_or_invented_quotes_fail(request_, problem):
    assert problem in problems(DUTCH_DRAFT, requests=[request_])


def test_quote_matching_forgives_typography_only():
    typed = "add one under instellingen -> vestigingen ->  nieuwe vestiging (settings -> locations -> new location)"
    assert problems(DUTCH_DRAFT, requests=[Request(description_en="x", faq_id="F07", faq_quote=typed)]) == []


@pytest.mark.parametrize(
    "faq_id, quote",
    [
        ("F10", "We read every idea."),  # the shortest sentences in the help centre are still evidence
        ("F05", "Prices exclude VAT."),
        ("f10", "We read every idea."),
        ("F10 · Ideas and feature requests", "We read every idea."),
        ("F07", "Add one under Instellingen > Vestigingen > Nieuwe vestiging (Settings > Locations > New location)."),
    ],
)
def test_real_help_centre_sentences_pass_however_the_model_writes_them(faq_id, quote):
    assert problems(DUTCH_DRAFT, requests=[Request(description_en="x", faq_id=faq_id, faq_quote=quote)]) == []


def test_promise_patterns_in_other_languages():
    quote = [Request(description_en="x", faq_id="F12", faq_quote=F12_QUOTE)]
    assert "promise_in_draft" in problems(
        "Guten Tag, wir antworten innerhalb 24 Stunden auf Ihre Frage zur App.", language="de", requests=quote
    )
    assert "promise_in_draft" in problems(
        "Hello, we will get back to you within one business day about the app.", language="en", requests=quote
    )


def test_very_long_unbroken_text_does_not_stall_language_detection():
    import time

    started = time.monotonic()
    detect_language("x" * 40_000 + " → menu")
    assert time.monotonic() - started < 2


def test_two_requests_with_one_unanswered_fail():
    unanswered = Request(description_en="Does it work with Exact?", faq_id=None, faq_quote=None)
    assert "request_without_faq_answer" in problems(DUTCH_DRAFT, requests=F12 + [unanswered])


def test_no_requests_or_no_draft_fail():
    assert problems(DUTCH_DRAFT, requests=[]) == ["no_request_found"]
    assert problems("  ") == ["empty_draft"]


# --- handoff notes and the verifier ---------------------------------------------------


def test_note_must_be_english_and_free_of_payment_details():
    assert check_note("Customer asks which Android version the app needs. Covered by F12.") == []
    assert check_note("De klant vraagt welke versie van Android nodig is voor de app.") == ["note_not_english"]
    assert "payment_details_in_note" in check_note(f"Customer sent card {CARD}.")


def test_verifier_prompt_uses_the_same_rule_definitions():
    prompt = build_verifier_prompt(kb)
    assert all(rule.definition in prompt for rule in ALWAYS_HUMAN.values())
    assert all(entry_id in prompt for entry_id in kb.entries)


@pytest.mark.parametrize(
    "overrides, problem",
    [
        ({"draft_language": "en"}, "verifier_wrong_language"),
        ({"unaddressed_requests": ["Does it work with Exact?"]}, "verifier_unaddressed_request"),
        ({"unsupported_claims": ["Says the app is free"]}, "verifier_unsupported_claim"),
        ({"promises_or_discounts": True}, "verifier_promise"),
    ],
)
def test_each_verdict_finding_is_reported_separately(overrides, problem):
    assert make_verdict().problems("nl") == []
    assert make_verdict(**overrides).problems("nl") == [problem]
