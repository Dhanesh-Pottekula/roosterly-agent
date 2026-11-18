import pytest

from agent.decide import Proposal
from agent.faq import load_kb
from agent.llm import LLMDeterministicError, LLMTransientError
from agent.models import Ticket
from agent.pipeline import decide_ticket, is_empty
from agent.verify import Verdict
from tests.fakes import CARD, FakeLLM, make_proposal

kb = load_kb()
MODEL = "gpt-6-luna"


def ticket(body="Hallo, op welke versie van Android draait jullie app? Wij willen tablets kopen.", subject="Tablet"):
    return Ticket(id="X1", subject=subject, body=body, customer_plan="Basis", channel="email")


def test_good_reply_goes_through_proposer_checks_and_verifier():
    llm = FakeLLM()
    outcome = decide_ticket(ticket(), kb, llm, MODEL)
    assert outcome.decision.action == "reply" and outcome.verdict is not None
    assert (llm.count(Proposal), llm.count(Verdict)) == (1, 1)
    assert outcome.usage.input_tokens == 2000  # both calls are counted


@pytest.mark.parametrize(
    "subject, body, empty",
    [
        ("Re:", "", True),
        ("", "  \n", True),
        ("FW: re: ", "", True),
        ("Antw:", "", True),
        ("Re: factuur", "", False),
        ("Re:", "Al nieuws?", False),
    ],
)
def test_is_empty(subject, body, empty):
    assert is_empty(ticket(body=body, subject=subject)) is empty


def test_empty_ticket_is_closed_without_calling_the_llm():
    llm = FakeLLM()
    outcome = decide_ticket(ticket(body="", subject="Re:"), kb, llm, MODEL)
    assert (outcome.decision.action, outcome.decision.category, outcome.decision.priority) == ("close", "other", "P4")
    assert llm.calls == []


def test_transient_llm_error_leaves_the_ticket_undecided():
    outcome = decide_ticket(ticket(), kb, FakeLLM(proposal=LLMTransientError("529")), MODEL)
    assert outcome.decision is None and outcome.transient_error == "529"


def test_refusal_becomes_a_handoff_not_a_crash():
    outcome = decide_ticket(ticket(), kb, FakeLLM(proposal=LLMDeterministicError("refusal")), MODEL)
    assert outcome.decision.action == "handoff" and outcome.handoff_reasons == ["triage_failed:refusal"]
    assert "refusal" in outcome.decision.handoff_note


def test_verifier_is_skipped_when_the_deterministic_checks_already_failed():
    llm = FakeLLM(proposal=make_proposal(draft_reply="Hoi! Je krijgt 25% korting als je een jaar vooruit betaalt."))
    outcome = decide_ticket(ticket(), kb, llm, MODEL)
    assert outcome.decision.action == "handoff" and llm.count(Verdict) == 0
    assert {"percentage_in_draft", "promise_in_draft"} <= set(outcome.handoff_reasons)


def test_verifier_failure_keeps_the_proposal_and_hands_off():
    # A refusal by the verifier must not throw away the proposer's category, priority and note.
    llm = FakeLLM(verdict=LLMDeterministicError("refusal"))
    outcome = decide_ticket(ticket(), kb, llm, MODEL)
    d = outcome.decision
    assert (d.action, d.category, d.priority) == ("handoff", "question", "P4")
    assert outcome.handoff_reasons == ["verifier_failed:refusal"] and "F12" in d.handoff_note


def test_verifier_outage_leaves_the_ticket_open():
    outcome = decide_ticket(ticket(), kb, FakeLLM(verdict=LLMTransientError("529")), MODEL)
    assert outcome.decision is None and outcome.transient_error == "529"


def test_nothing_we_post_repeats_a_card_number():
    # Worst case: the model copies the card number into every free-text field.
    leaky = make_proposal(
        proposed_action="handoff",
        draft_reply=None,
        priority_reason=f"Card {CARD} was sent.",
        summary_en=f"Wants to pay with {CARD}.",
        handoff_note=f"Customer card: {CARD}, cvv 987.",
    )
    outcome = decide_ticket(ticket(body=f"Mijn kaart is {CARD}, cvv 987"), kb, FakeLLM(proposal=leaky), MODEL)
    posted = outcome.decision.model_dump_json()
    assert outcome.decision.action == "handoff" and "5555" not in posted and "987" not in posted


def test_card_followed_by_its_expiry_never_reaches_the_llm_or_the_post():
    llm = FakeLLM(proposal=make_proposal(proposed_action="handoff", draft_reply=None))
    outcome = decide_ticket(ticket(body="Graag deze kaart gebruiken: 5555555555554444 03/31 482"), kb, llm, MODEL)
    sent = " ".join(user for _, _, user in llm.calls)
    assert "5555" not in sent and "03/31" not in sent and "[CARD]" in sent
    assert outcome.handoff_reasons == ["payment_details_in_ticket"] and outcome.decision.priority == "P2"
