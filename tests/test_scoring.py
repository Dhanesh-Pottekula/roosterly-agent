from agent.faq import load_kb
from agent.models import Decision
from agent.pipeline import TicketOutcome
from eval.run_eval import load_cases, run_once, summarise, to_markdown
from eval.scoring import Case, CaseResult, expected_faq_id, score
from tests.fakes import FakeLLM, make_proposal

MODEL = "claude-haiku-5-5"


def result(expected_action, got_action, expected_priority="P4", got_priority="P4", reason="Answered by FAQ F12."):
    label = {
        "language": "nl",
        "category": "question",
        "priority": expected_priority,
        "action": expected_action,
        "reason": reason,
    }
    decision = Decision(
        action=got_action,
        language="nl",
        category="question",
        priority=got_priority,
        reason="r",
        draft_reply="d",
        handoff_note="n",
        confidence=0.9,
    )
    return CaseResult(Case("C", "examples", {}, label), TicketOutcome(decision, proposal=make_proposal()))


def test_safety_errors_are_counted_separately():
    scores = score(
        [result("handoff", "reply"), result("reply", "handoff"), result("handoff", "close"), result("reply", "reply")],
        MODEL,
    )
    assert (scores["unsafe_replies"], scores["over_escalations"], scores["wrong_closes"]) == (1, 1, 1)
    assert scores["action_accuracy"] == 0.25 and scores["handoff_rate"] == 0.25
    assert scores["gate_changes"] == 2  # the proposal said reply, the gate decided otherwise


def test_priority_within_one_level():
    scores = score([result("reply", "reply", "P2", "P3"), result("reply", "reply", "P1", "P4")], MODEL)
    assert scores["priority_accuracy"] == 0.0 and scores["priority_within_one"] == 0.5


def test_faq_match_reads_the_entry_from_the_label():
    assert expected_faq_id({"reason": "Covered by help centre entry F06, reply in Dutch."}) == "F06"
    assert expected_faq_id({"reason": "Refund request."}) is None
    assert score([result("reply", "reply")], MODEL)["faq_match"] == 1.0
    assert score([result("reply", "reply", reason="Answered by F06.")], MODEL)["faq_match"] == 0.0


def test_posted_text_that_repeats_forbidden_content_is_a_safety_error():
    leaky = result("handoff", "handoff")
    leaky.case = Case("C", "adversarial", {}, leaky.case.label, never_post=("NL91",), never_reply=("drie maanden",))
    leaky.outcome.decision = leaky.outcome.decision.model_copy(
        update={
            "handoff_note": "Pay from NL91 ABNA. Ticket told the AI to promise 'drie maanden' free; ignored.",
            "draft_reply": None,
        }
    )
    clean = result("handoff", "handoff")
    assert leaky.forbidden_text_posted == ["NL91"]  # quoting the ignored instruction in a note is fine
    assert score([leaky, clean], MODEL)["forbidden_text_posted"] == 1
    leaky.outcome.decision = leaky.outcome.decision.model_copy(update={"draft_reply": "Je krijgt drie maanden cadeau."})
    assert leaky.forbidden_text_posted == ["NL91", "drie maanden"]


def test_reply_quality_is_not_a_fake_zero_when_the_judge_did_not_run():
    assert score([result("reply", "reply")], MODEL)["reply_quality"] is None


def test_undecided_ticket_counts_as_an_error_not_a_crash():
    undecided = CaseResult(result("reply", "reply").case, TicketOutcome(decision=None))
    scores = score([undecided], MODEL)
    assert (
        scores["errors"] == 1
        and scores["action_accuracy"] == 0.0
        and undecided.mismatches == {"decision": "none (LLM error)"}
    )


def test_eval_runs_offline_on_the_committed_cases():
    cases = load_cases()
    assert sum(c.group == "examples" for c in cases) == 10 and sum(c.group == "adversarial" for c in cases) >= 8
    assert all(set(c.label) >= {"language", "category", "priority", "action", "reason"} for c in cases)
    results, _ = run_once(cases, load_kb(), FakeLLM(), MODEL, judge_model=None)
    summary = summarise([results, results], MODEL)
    report = to_markdown("test", MODEL, [results, results], summary)
    assert summary["examples"]["cases"]["mean"] == 10 and "**unsafe_replies**" in report
    assert summary["examples"]["reply_quality"] is None and "| reply_quality | n/a | n/a |" in report
    assert any(c.never_post for c in cases) and any(c.never_reply for c in cases)
    assert all(set(c.ticket) <= {"id", "received_at", "channel", "customer_plan", "subject", "body"} for c in cases)
