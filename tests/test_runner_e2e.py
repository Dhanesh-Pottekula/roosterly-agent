"""End to end against the fake helpdesk: every ticket handled exactly once, whatever goes wrong."""

import json

import pytest

from agent.decide import Proposal
from agent.faq import load_kb
from agent.helpdesk import HelpdeskClient
from agent.llm import LLMFatalError, LLMTransientError
from agent.runner import Runner
from tests.fakes import CARD, SETTINGS, FakeClock, FakeHelpdesk, FakeLLM, make_proposal, make_tickets

kb = load_kb()


def setup(count=30, **fake_options):
    clock = FakeClock()
    fake = FakeHelpdesk(make_tickets(count), clock=clock, **fake_options)
    return HelpdeskClient(SETTINGS, transport=fake.transport, sleep=clock.sleep, clock=clock), fake


def log_lines(tmp_path, summary):
    return [json.loads(line) for line in (tmp_path / f"{summary.run_id}.jsonl").read_text().splitlines()]


def test_every_ticket_is_handled_exactly_once_despite_503s(tmp_path):
    client, fake = setup(fail_rate=0.2)
    summary = Runner(SETTINGS, client, FakeLLM(), kb, log_dir=tmp_path).run()
    assert len(fake.handled) == 30 and set(fake.post_count.values()) == {1}
    assert fake.statuses[503] > 0 and fake.statuses[409] == 0 and fake.statuses[429] == 0
    assert summary.left_open == 0 and summary.outcomes == {"posted": 30}
    assert summary.status == {"mode": "practice", "open_tickets": 0, "handled_tickets": 30}


def test_crash_then_rerun_gives_no_customer_two_answers(tmp_path):
    client, fake = setup(fail_rate=0.05)
    fake.crash_after_posts = 7
    with pytest.raises(KeyboardInterrupt):
        Runner(SETTINGS, client, FakeLLM(), kb, log_dir=tmp_path).run()
    assert len(fake.handled) == 7

    summary = Runner(SETTINGS, client, FakeLLM(), kb, log_dir=tmp_path).run()
    assert len(fake.handled) == 30 and max(fake.post_count.values()) == 1 and fake.statuses[409] == 0
    assert summary.outcomes == {"posted": 23}


def test_ticket_answered_by_someone_else_meanwhile_is_skipped(tmp_path):
    client, fake = setup(count=3)
    original_list = client.list_open_ids

    def list_then_someone_answers():
        ids = original_list()
        fake.handled["X002"] = {"action": "handoff"}
        return ids

    client.list_open_ids = list_then_someone_answers
    summary = Runner(SETTINGS, client, FakeLLM(), kb, log_dir=tmp_path).run()
    assert summary.outcomes == {"posted": 2, "skipped_handled": 1} and fake.post_count["X002"] == 0


def test_llm_outage_leaves_tickets_open_and_reports_it(tmp_path):
    client, fake = setup(count=4)
    summary = Runner(SETTINGS, client, FakeLLM(proposal=LLMTransientError("overloaded")), kb, log_dir=tmp_path).run()
    assert fake.handled == {} and summary.left_open == 4
    assert [line["outcome"] for line in log_lines(tmp_path, summary)] == ["left_open_llm"] * 4


def test_ticket_that_fails_once_is_retried_at_the_end_of_the_run(tmp_path):
    client, fake = setup(count=3)
    failures = iter([LLMTransientError("overloaded")])
    llm = FakeLLM(proposal=lambda user: next(failures, None) or make_proposal())
    summary = Runner(SETTINGS, client, llm, kb, log_dir=tmp_path).run()
    assert len(fake.handled) == 3 and summary.left_open == 0 and llm.count(Proposal) == 4
    assert len(log_lines(tmp_path, summary)) == 3  # still one line per ticket


def test_dry_run_posts_nothing(tmp_path, capsys):
    client, fake = setup(count=3)
    summary = Runner(SETTINGS, client, FakeLLM(), kb, dry_run=True, log_dir=tmp_path).run(limit=2)
    assert fake.handled == {} and sum(fake.post_count.values()) == 0
    assert summary.run_id.startswith("dry-") and summary.outcomes == {"dry_run": 2}
    assert "draft in English" in capsys.readouterr().out


def test_one_log_line_per_ticket_without_customer_text_or_payment_details(tmp_path):
    client, fake = setup(count=5)
    for ticket in fake.tickets.values():
        ticket["body"] = f"Welke Android-versie is nodig? Mijn kaart is {CARD}, cvv 987."
    leaky = make_proposal(priority_reason=f"Card {CARD}.", handoff_note=f"Card {CARD}.")
    summary = Runner(SETTINGS, client, FakeLLM(proposal=leaky), kb, log_dir=tmp_path).run()

    lines = log_lines(tmp_path, summary)
    assert [line["ticket_id"] for line in lines] == [f"X{i:03d}" for i in range(1, 6)]
    assert all(line["action"] == "handoff" and line["handoff_reasons"] == ["payment_details_in_ticket"] for line in lines)
    everything_written = "".join(path.read_text() for path in tmp_path.iterdir())
    assert "5555" not in everything_written and "Android" not in everything_written and "test-key" not in everything_written
    posted = json.dumps(fake.handled)
    assert "5555" not in posted and "987" not in posted
    assert (tmp_path / "spend.jsonl").exists() and (tmp_path / f"{summary.run_id}.summary.json").exists()


def test_rejected_decision_falls_back_to_a_handoff(tmp_path):
    client, fake = setup(count=1)
    original = fake._post
    rejected_once = []

    def reject_first(request, ticket_id, action):
        if not rejected_once:
            rejected_once.append(True)
            return type(original(request, "missing", action))(422, json={"error": "validation_failed", "problems": ["draft_reply"]})
        return original(request, ticket_id, action)

    fake._post = reject_first
    summary = Runner(SETTINGS, client, FakeLLM(), kb, log_dir=tmp_path).run()
    assert fake.handled["X001"]["action"] == "handoff"
    line = log_lines(tmp_path, summary)[0]
    assert line["outcome"] == "posted" and line["action"] == "handoff" and line["validation_problems"] == ["draft_reply"]


def test_one_broken_ticket_does_not_block_the_queue(tmp_path):
    client, fake = setup(count=3)
    fake.tickets["X002"]["body"] = "kapot"

    def proposal_or_bug(user):
        if "kapot" in user:
            raise RuntimeError("unexpected bug")
        return make_proposal()

    summary = Runner(SETTINGS, client, FakeLLM(proposal=proposal_or_bug), kb, log_dir=tmp_path).run()
    assert set(fake.handled) == {"X001", "X003"} and summary.left_open == 1
    broken = [line for line in log_lines(tmp_path, summary) if line["ticket_id"] == "X002"]
    assert broken == [broken[0]] and (broken[0]["outcome"], broken[0]["error"]) == ("failed_error", "RuntimeError")


def test_wrong_llm_key_stops_the_run_before_anything_is_posted(tmp_path):
    client, fake = setup(count=3)
    with pytest.raises(LLMFatalError):
        Runner(SETTINGS, client, FakeLLM(proposal=LLMFatalError("401")), kb, log_dir=tmp_path).run()
    assert fake.handled == {}


def test_command_line_reports_setup_problems_without_a_traceback(monkeypatch, capsys):
    from agent import __main__ as cli

    for var in ("HELPDESK_URL", "HELPDESK_API_KEY", "CANDIDATE_EMAIL", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(cli, "load_settings", lambda **_: (_ for _ in ()).throw(cli.ConfigError("Missing environment variables: HELPDESK_API_KEY")))
    monkeypatch.setattr("sys.argv", ["agent", "status"])
    assert cli.main() == 2
    assert "HELPDESK_API_KEY" in capsys.readouterr().err


def test_log_writer_redacts_whatever_it_is_given(tmp_path):
    from agent.runlog import RunLogger

    log = RunLogger("unit", tmp_path)
    log.write({"ticket_id": "X1", "note": f"card {CARD} and iban NL91ABNA0417164300"})
    written = log.path.read_text()
    assert "5555" not in written and "NL91" not in written and json.loads(written)["ticket_id"] == "X1"
