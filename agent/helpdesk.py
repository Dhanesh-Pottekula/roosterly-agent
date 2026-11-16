"""HTTP client for the helpdesk: throttled, retried, and safe against double answers.

The three failure kinds are handled differently, on purpose:
- 503 means "nothing was saved" and 429 means "too fast": both are safe to retry.
- 409 means the ticket already has an answer: never retry.
- A timeout or another 5xx on a POST is ambiguous, because the write may have succeeded.
  We then read the ticket back, and post again only if it is still open.
"""

import random
import time
from enum import Enum

import httpx

from agent.config import Settings
from agent.models import Decision, Ticket

MAX_TRIES = 8  # per request, for 429/503 and failed reads
MAX_POST_ROUNDS = 3  # per decision, for ambiguous writes
PAGE_SIZE = 10


class PostOutcome(Enum):
    POSTED = "posted"
    ALREADY_HANDLED = "already_handled"
    NOT_FOUND = "not_found"


class FatalAPIError(Exception):
    """The request itself is wrong (400, 401, ...). Retrying will not help."""


class ValidationRejected(Exception):
    """422: the helpdesk refused the decision body."""

    def __init__(self, problems: list):
        super().__init__(f"validation failed: {problems}")
        self.problems = problems


class TransientExhausted(Exception):
    """The helpdesk kept failing after every retry. The ticket is left for the next run."""


class _AmbiguousWrite(Exception):
    """A POST failed in a way that does not tell us whether it was saved."""


class HelpdeskClient:
    def __init__(self, settings: Settings, transport=None, sleep=time.sleep, clock=time.monotonic):
        self._http = httpx.Client(
            base_url=settings.helpdesk_url,
            headers={"Authorization": f"Bearer {settings.api_key}", "X-Candidate-Email": settings.email},
            timeout=20.0,
            transport=transport,
        )
        self._name = settings.name
        self._min_interval = 1.0 / settings.rps
        self._sleep = sleep
        self._clock = clock
        self._last_request_at: float | None = None
        self.attempts = 0  # total HTTP requests sent, for the log

    # --- endpoints -------------------------------------------------------------------

    def status(self, send_name: bool = False) -> dict:
        headers = {"X-Candidate-Name": self._name} if send_name and self._name else None
        return self._read("/status", headers=headers).json()

    def examples(self) -> list[dict]:
        return self._read("/examples").json()["examples"]

    def list_open_ids(self) -> list[str]:
        """Every open ticket id. All pages are collected before any ticket is handled."""
        ids: list[str] = []
        cursor = None
        while True:
            params = {"limit": PAGE_SIZE, "status": "open"} | ({"cursor": cursor} if cursor else {})
            page = self._read("/tickets", params=params).json()
            ids += [ticket["id"] for ticket in page["tickets"]]
            cursor = page.get("next_cursor")
            if not cursor:
                return ids

    def get_ticket(self, ticket_id: str) -> Ticket | None:
        """The ticket with its current `handled` state, or None if it is not in our set."""
        response = self._request("GET", f"/tickets/{ticket_id}")
        if response.status_code == 404:
            return None
        return Ticket.model_validate(self._checked(response).json())

    def post_decision(self, ticket_id: str, decision: Decision) -> PostOutcome:
        path = f"/tickets/{ticket_id}/{decision.action}"
        for _ in range(MAX_POST_ROUNDS):
            try:
                response = self._request("POST", path, json=decision.to_api_body())
            except _AmbiguousWrite:
                ticket = self.get_ticket(ticket_id)
                if ticket is not None and ticket.handled is not None:
                    return PostOutcome.POSTED  # the write did land; do not answer twice
                continue
            if response.status_code == 409:
                return PostOutcome.ALREADY_HANDLED
            if response.status_code == 404:
                return PostOutcome.NOT_FOUND
            if response.status_code in (400, 422):  # the body is wrong; the same body will fail again
                error = response.json()
                raise ValidationRejected(error.get("problems") or [error.get("message", "bad request")])
            self._checked(response)
            return PostOutcome.POSTED
        raise TransientExhausted(f"POST {path}: still ambiguous after {MAX_POST_ROUNDS} rounds")

    def reset_practice(self) -> dict:
        """Re-open all practice tickets. Resetting twice changes nothing, so every failure is retried."""
        return self._checked(self._request("POST", "/practice/reset", repeatable=True)).json()

    # --- transport -------------------------------------------------------------------

    def _read(self, path: str, **kwargs) -> httpx.Response:
        return self._checked(self._request("GET", path, **kwargs))

    @staticmethod
    def _checked(response: httpx.Response) -> httpx.Response:
        if not response.is_success:
            raise FatalAPIError(f"{response.request.method} {response.request.url.path}: {response.status_code} {response.text[:200]}")
        return response

    def _request(self, method: str, path: str, repeatable: bool | None = None, **kwargs) -> httpx.Response:
        """Send one request, retrying what is safe to retry. Returns any non-retryable response.

        A request is repeatable when sending it twice does no harm. Reads are; posting an answer is not.
        """
        repeatable = method == "GET" if repeatable is None else repeatable
        for attempt in range(MAX_TRIES):
            self._throttle()
            self.attempts += 1
            try:
                response = self._http.request(method, path, **kwargs)
            except httpx.TransportError as error:
                if not repeatable:
                    raise _AmbiguousWrite(str(error)) from error
                self._backoff(attempt)
                continue
            if response.status_code == 429:
                self._sleep(_retry_after_seconds(response))
            elif response.status_code == 503:
                self._backoff(attempt)
            elif response.status_code >= 500 and not repeatable:
                raise _AmbiguousWrite(f"{response.status_code}")
            elif response.status_code >= 500:
                self._backoff(attempt)
            else:
                return response
        raise TransientExhausted(f"{method} {path}: gave up after {MAX_TRIES} tries")

    def _throttle(self) -> None:
        """Space requests at least 1/rps apart, so we stay under the rate limit."""
        if self._last_request_at is not None:
            wait = self._last_request_at + self._min_interval - self._clock()
            if wait > 0:
                self._sleep(wait)
        self._last_request_at = self._clock()

    def _backoff(self, attempt: int) -> None:
        self._sleep(min(0.5 * 2**attempt, 8.0) + random.uniform(0, 0.25))


def _retry_after_seconds(response: httpx.Response) -> float:
    try:
        return max(float(response.headers.get("Retry-After", "")), 0.0)
    except ValueError:
        return 1.0
