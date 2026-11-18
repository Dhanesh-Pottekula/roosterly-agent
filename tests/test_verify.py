"""The independent verifier: its prompt and how its verdict becomes problems."""

import pytest

from agent.faq import load_kb
from agent.rules import ALWAYS_HUMAN
from agent.verify import build_verifier_prompt
from tests.fakes import make_verdict

kb = load_kb()


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


def test_verifier_prompt_says_restating_a_help_centre_fact_is_not_a_promise():
    prompt = build_verifier_prompt(kb)
    assert "Restating a fact from the help centre" in prompt


def test_verifier_prompt_checks_every_always_human_rule():
    prompt = build_verifier_prompt(kb)
    assert all(rule.definition in prompt for rule in ALWAYS_HUMAN.values())


def test_verifier_prompt_lets_a_help_centre_statement_apply_to_the_customers_own_case():
    assert "applies a help centre statement to the customer's own case" in build_verifier_prompt(kb)
