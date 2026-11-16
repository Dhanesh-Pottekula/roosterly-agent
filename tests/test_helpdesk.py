import pytest

from agent.helpdesk import FatalAPIError, HelpdeskClient, PostOutcome, TransientExhausted, ValidationRejected
from agent.models import Decision
from tests.fakes import SETTINGS, FakeClock, FakeHelpdesk, make_tickets

HANDOFF = Decision(action="handoff", language="nl", category="question", priority="P4",
                   reason="Not in the help centre.", handoff_note="Customer asks about X.", confidence=0.6)


def make_client(count=3, **fake_options):
    clock = FakeClock()
    fake = FakeHelpdesk(make_tickets(count), clock=clock, **fake_options)
    return HelpdeskClient(SETTINGS, transport=fake.transport, sleep=clock.sleep, clock=clock), fake, clock


def test_503_is_retried_until_it_succeeds():
    client, fake, _ = make_client()
    fake.scripted = [503, 503]
    assert client.post_decision("X001", HANDOFF) is PostOutcome.POSTED
    assert fake.post_count["X001"] == 1  # the 503s never reached the ticket: nothing was saved


def test_429_waits_for_retry_after():
    client, fake, clock = make_client()
    fake.scripted = [429]
    client.status()
    assert 2.0 in clock.sleeps


def test_409_is_never_retried():
    client, fake, _ = make_client()
    client.post_decision("X001", HANDOFF)
    assert client.post_decision("X001", HANDOFF) is PostOutcome.ALREADY_HANDLED
    assert fake.post_count["X001"] == 2  # the second call got one 409 and stopped


@pytest.mark.parametrize("failure", ["timeout", 504])
def test_lost_response_after_a_saved_post_is_not_posted_again(failure):
    client, fake, _ = make_client()
    fake.commit_then_fail = failure
    assert client.post_decision("X001", HANDOFF) is PostOutcome.POSTED
    assert fake.post_count["X001"] == 1 and fake.statuses[409] == 0


def test_timeout_before_the_post_was_saved_is_posted_again():
    client, fake, _ = make_client()
    fake.scripted = ["timeout"]
    assert client.post_decision("X001", HANDOFF) is PostOutcome.POSTED
    assert fake.post_count["X001"] == 1 and "X001" in fake.handled


def test_pagination_collects_every_open_ticket():
    client, _, _ = make_client(count=25)
    assert client.list_open_ids() == [f"X{i:03d}" for i in range(1, 26)]


def test_unknown_ticket():
    client, _, _ = make_client()
    assert client.get_ticket("NOPE") is None
    assert client.post_decision("NOPE", HANDOFF) is PostOutcome.NOT_FOUND


def test_wrong_key_is_fatal_and_not_retried():
    clock = FakeClock()
    fake = FakeHelpdesk(make_tickets(1))
    client = HelpdeskClient(SETTINGS.__class__(**{**SETTINGS.__dict__, "api_key": "wrong"}), transport=fake.transport, sleep=clock.sleep, clock=clock)
    with pytest.raises(FatalAPIError):
        client.status()
    assert len(fake.requests) == 1


def test_422_reports_the_problems():
    client, _, _ = make_client()
    bad = HANDOFF.model_copy(update={"language": "fr"})
    with pytest.raises(ValidationRejected) as rejected:
        client.post_decision("X001", bad)
    assert "language" in rejected.value.problems


def test_400_on_a_post_is_a_rejected_decision_not_a_dead_run():
    client, fake, _ = make_client()
    fake.scripted = [400]
    with pytest.raises(ValidationRejected):
        client.post_decision("X001", HANDOFF)
    assert "X001" not in fake.handled


def test_reset_is_retried_through_any_failure():
    client, fake, _ = make_client()
    fake.scripted = ["timeout", 500, 503]
    client.reset_practice()
    assert len(fake.requests) == 4


def test_retry_after_is_capped():
    client, fake, clock = make_client()
    fake.scripted = [429]
    original = fake._respond

    def with_huge_wait(request):
        response = original(request)
        if response.status_code == 429:
            response.headers["Retry-After"] = "86400"
        return response

    fake._respond = with_huge_wait
    client.status()
    assert max(clock.sleeps) <= 30


def test_gives_up_when_the_helpdesk_stays_down():
    client, fake, _ = make_client()
    fake.scripted = [503] * 20
    with pytest.raises(TransientExhausted):
        client.list_open_ids()


def test_throttle_stays_under_the_rate_limit():
    client, fake, _ = make_client()
    for _ in range(50):
        client.status()
    assert fake.statuses[429] == 0


def test_name_header_is_sent_only_when_asked():
    client, fake, _ = make_client()
    client.status(send_name=True)
    client.status()
    assert [r.headers.get("X-Candidate-Name") for r in fake.requests] == ["Test Candidate", None]
    assert all(r.headers["X-Candidate-Email"] == "me@example.test" for r in fake.requests)
