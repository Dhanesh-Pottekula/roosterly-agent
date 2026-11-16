import pytest
from pydantic import ValidationError

from agent.config import ConfigError, load_settings
from agent.models import Decision, Ticket, Usage


def _decision(**overrides):
    base = dict(
        action="reply",
        language="nl",
        category="question",
        priority="P4",
        reason="How-to question answered by the help centre.",
        draft_reply="Hoi, ga naar Team.",
        handoff_note=None,
        confidence=0.9,
    )
    base.update(overrides)
    return Decision(**base)


def test_reply_requires_draft():
    with pytest.raises(ValidationError):
        _decision(draft_reply=None)


def test_handoff_requires_note():
    with pytest.raises(ValidationError):
        _decision(action="handoff", draft_reply=None, handoff_note="")


def test_reason_required():
    with pytest.raises(ValidationError):
        _decision(reason="  ")


def test_to_api_body_excludes_action_and_nones():
    body = _decision().to_api_body()
    assert "action" not in body
    assert "handoff_note" not in body
    assert body["draft_reply"] == "Hoi, ga naar Team."
    assert body["confidence"] == 0.9


def test_load_settings_reports_all_missing(monkeypatch):
    for var in ("HELPDESK_URL", "HELPDESK_API_KEY", "CANDIDATE_EMAIL", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ConfigError) as err:
        load_settings(dotenv=False)
    assert "HELPDESK_API_KEY" in str(err.value)
    assert "CANDIDATE_EMAIL" in str(err.value)


def test_load_settings_llm_key_optional_for_offline(monkeypatch):
    monkeypatch.setenv("HELPDESK_URL", "https://example.test/helpdesk")
    monkeypatch.setenv("HELPDESK_API_KEY", "k")
    monkeypatch.setenv("CANDIDATE_EMAIL", "me@example.test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s = load_settings(dotenv=False, require_llm=False)
    assert s.llm_model == "claude-haiku-5-5"
    assert s.rps == 4.0


def test_usage_cost():
    u = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert u.cost_usd("claude-haiku-5-5") == pytest.approx(0.60)
    assert (u + Usage(input_tokens=1)).input_tokens == 1_000_001


def test_model_without_a_price_is_rejected_at_startup(monkeypatch):
    for var, value in (("HELPDESK_URL", "https://x.test"), ("HELPDESK_API_KEY", "k"), ("CANDIDATE_EMAIL", "a@b.test"), ("LLM_MODEL", "some-new-model")):
        monkeypatch.setenv(var, value)
    with pytest.raises(ConfigError, match="some-new-model"):
        load_settings(dotenv=False, require_llm=False)


def test_ticket_with_null_fields_is_still_a_ticket():
    ticket = Ticket.model_validate({"id": "X1", "subject": None, "body": None, "customer_plan": None, "handled": None})
    assert (ticket.subject, ticket.body, ticket.customer_plan) == ("", "", "")
    assert Ticket.model_validate({"id": "X1", "handled": True}).handled is not None  # any shape counts as handled


def test_cached_input_is_cheaper():
    assert Usage(cache_read_tokens=1_000_000).cost_usd("claude-haiku-5-5") == pytest.approx(0.01)
