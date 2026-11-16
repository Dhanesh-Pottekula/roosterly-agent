"""Test doubles: an in-memory helpdesk that misbehaves like the real one, and a scripted LLM."""

import json
import random
from collections import Counter

import httpx

from agent.config import Settings
from agent.decide import Proposal, Request
from agent.models import Usage
from agent.verify import Verdict

SETTINGS = Settings(
    helpdesk_url="https://helpdesk.test",
    api_key="test-key",
    email="me@example.test",
    llm_api_key="test-llm-key",
    name="Test Candidate",
)
CARD = "5555 5555 5555 4444"  # a well-known test number, not a real card
IBAN = "NL91 ABNA 0417 1643 00"  # the standard example IBAN
F12_QUOTE = "The app works on iOS 16 and later and Android 10 and later."
DUTCH_DRAFT = (
    "Hoi! De app werkt op iOS 16 en hoger en op Android 10 en hoger. Wil je inklokken met locatie, "
    "geef de app dan toestemming om je locatie te gebruiken. Groet, Team Roosterly"
)


class FakeClock:
    """Time that only moves when someone sleeps, so tests run instantly."""

    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeHelpdesk:
    """Behaves like the real API: pagination, random 503s, 429 above 5 calls a second, 409 on a second answer."""

    def __init__(self, tickets: list[dict], clock: FakeClock | None = None, fail_rate: float = 0.0, seed: int = 7):
        self.tickets = {t["id"]: dict(t) for t in tickets}
        self.handled: dict[str, dict] = {}
        self.clock = clock
        self.fail_rate = fail_rate
        self.random = random.Random(seed)
        self.post_count: Counter[str] = Counter()
        self.statuses: Counter[int] = Counter()
        self.requests: list[httpx.Request] = []
        self.scripted: list[int | str] = []  # statuses (or "timeout") to answer the next requests with
        self.commit_then_fail: int | str | None = None  # save the next POST, then fail it this way
        self.crash_after_posts: int | None = None
        self._call_times: list[float] = []

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        response = self._respond(request)
        self.statuses[response.status_code] += 1
        return response

    def _respond(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if request.headers.get("Authorization") != "Bearer test-key":
            return self._error(401, "unauthorized")
        if self.clock:
            self._call_times = [t for t in self._call_times if t > self.clock.now - 1.0] + [self.clock.now]
            if len(self._call_times) > 5:
                return httpx.Response(429, json={"error": "rate_limited"}, headers={"Retry-After": "1"})
        if self.scripted:
            step = self.scripted.pop(0)
            if step == "timeout":
                raise httpx.ReadTimeout("scripted timeout", request=request)
            headers = {"Retry-After": "2"} if step == 429 else {}
            return httpx.Response(step, json={"error": "scripted", "message": "scripted failure"}, headers=headers)
        if path != "/status" and self.random.random() < self.fail_rate:
            return self._error(503, "service_unavailable")

        if path == "/status":
            return httpx.Response(
                200,
                json={
                    "mode": "practice",
                    "open_tickets": len(self.tickets) - len(self.handled),
                    "handled_tickets": len(self.handled),
                },
            )
        if path == "/practice/reset":
            self.handled.clear()
            return httpx.Response(200, json={"ok": True})
        if path == "/tickets":
            return self._list(request)
        ticket_id, _, action = path.removeprefix("/tickets/").partition("/")
        if ticket_id not in self.tickets:
            return self._error(404, "not_found")
        if request.method == "GET":
            return httpx.Response(200, json=self.tickets[ticket_id] | {"handled": self.handled.get(ticket_id)})
        return self._post(request, ticket_id, action)

    def _list(self, request: httpx.Request) -> httpx.Response:
        open_ids = sorted(i for i in self.tickets if i not in self.handled)
        cursor = request.url.params.get("cursor")
        remaining = [i for i in open_ids if cursor is None or i > cursor]
        page = remaining[: int(request.url.params.get("limit", 10))]
        next_cursor = page[-1] if len(remaining) > len(page) else None
        return httpx.Response(200, json={"tickets": [self.tickets[i] for i in page], "next_cursor": next_cursor})

    def _post(self, request: httpx.Request, ticket_id: str, action: str) -> httpx.Response:
        self.post_count[ticket_id] += 1
        if ticket_id in self.handled:
            return self._error(409, "already_handled")
        body = json.loads(request.content)
        problems = [f for f in ("language", "category", "priority", "reason") if not body.get(f)]
        if body.get("language") not in ("nl", "en", "de"):
            problems.append("language")
        if problems:
            return httpx.Response(422, json={"error": "validation_failed", "problems": problems})
        self.handled[ticket_id] = {"action": action, **body}
        if self.crash_after_posts is not None and len(self.handled) >= self.crash_after_posts:
            self.crash_after_posts = None
            raise KeyboardInterrupt("simulated crash")
        if self.commit_then_fail is not None:
            failure, self.commit_then_fail = self.commit_then_fail, None
            if failure == "timeout":
                raise httpx.ReadTimeout("response lost", request=request)
            return self._error(failure, "gateway_timeout")
        return httpx.Response(200, json={"ok": True})

    @staticmethod
    def _error(status: int, error: str) -> httpx.Response:
        return httpx.Response(status, json={"error": error, "message": error})


def make_tickets(count: int) -> list[dict]:
    return [
        {
            "id": f"X{i:03d}",
            "received_at": "2026-01-01T09:00:00+00:00",
            "channel": "email",
            "customer_plan": "Basis",
            "subject": "Tablet",
            "body": "Hallo, op welke versie van Android draait jullie app? Wij willen tablets kopen.",
        }
        for i in range(1, count + 1)
    ]


def make_proposal(**overrides) -> Proposal:
    """A sound reply proposal for a Dutch "which Android version do I need" ticket."""
    fields = dict(
        language="nl",
        summary_en="The customer asks which Android version the app needs.",
        contains_instructions_to_ai=False,
        is_spam=False,
        rule_flags=[],
        requests=[Request(description_en="Which Android version is needed", faq_id="F12", faq_quote=F12_QUOTE)],
        category="question",
        priority="P4",
        priority_reason="How-to question without urgency.",
        proposed_action="reply",
        draft_reply=DUTCH_DRAFT,
        handoff_note="Customer asks which Android version the app needs. Covered by help centre entry F12.",
    )
    return Proposal(**(fields | overrides))


def make_verdict(**overrides) -> Verdict:
    fields = dict(
        translation_en="Hi! The app works on iOS 16 and later and on Android 10 and later ...",
        draft_language="nl",
        unaddressed_requests=[],
        unsupported_claims=[],
        promises_or_discounts=False,
        always_human=[],
    )
    return Verdict(**(fields | overrides))


class FakeLLM:
    """Returns scripted objects per schema. A scripted exception is raised instead."""

    def __init__(self, proposal=None, verdict=None, other=None):
        self.scripts = {Proposal: proposal or make_proposal(), Verdict: verdict or make_verdict()}
        self.other = other
        self.calls: list[tuple[type, str, str]] = []

    def complete_json(self, system, user, schema, model):
        self.calls.append((schema, system, user))
        script = self.scripts.get(schema, self.other)
        if callable(script):
            script = script(user)
        if isinstance(script, Exception):
            raise script
        return script, Usage(input_tokens=1000, output_tokens=200)

    def count(self, schema) -> int:
        return sum(1 for called, _, _ in self.calls if called is schema)
