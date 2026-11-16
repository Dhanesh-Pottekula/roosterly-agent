"""The data that moves through the pipeline: tickets in, one decision out."""

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, field_validator, model_validator

from agent.config import PRICES

Language = Literal["nl", "en", "de"]
Category = Literal["bug", "billing", "feature_request", "question", "other"]
Priority = Literal["P1", "P2", "P3", "P4"]
Action = Literal["reply", "handoff", "close"]

# Cache reads and writes are billed as a fraction of the normal input price.
CACHE_READ_FACTOR = 0.1
CACHE_WRITE_FACTOR = 1.25


class Ticket(BaseModel):
    id: str
    received_at: str = ""
    channel: str = ""
    customer_plan: str = ""
    subject: str = ""
    body: str = ""
    handled: Any = None  # how the ticket was handled; None while it is still open

    @field_validator("received_at", "channel", "customer_plan", "subject", "body", mode="before")
    @classmethod
    def _null_means_empty(cls, value):
        return "" if value is None else value


@dataclass(frozen=True)
class RedactedTicket:
    """A ticket with payment details removed. The only form the LLM and the logs ever see."""

    ticket: Ticket
    subject: str
    body: str
    had_payment: bool


class Decision(BaseModel):
    """Exactly what we post to the helpdesk for one ticket."""

    action: Action
    language: Language
    category: Category
    priority: Priority
    reason: str
    draft_reply: str | None = None
    handoff_note: str | None = None
    confidence: float

    @model_validator(mode="after")
    def _required_texts(self) -> "Decision":
        if not self.reason.strip():
            raise ValueError("reason is required")
        if self.action == "reply" and not (self.draft_reply or "").strip():
            raise ValueError("a reply needs a draft_reply")
        if self.action == "handoff" and not (self.handoff_note or "").strip():
            raise ValueError("a handoff needs a handoff_note")
        return self

    def to_api_body(self) -> dict:
        """The request body: the action is in the URL, and empty fields are left out."""
        return self.model_dump(exclude={"action"}, exclude_none=True)


@dataclass(frozen=True)
class Usage:
    """Tokens billed for LLM calls. Usages add up, so a ticket's total is a simple sum."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )

    def cost_usd(self, model: str) -> float:
        price_in, price_out = PRICES[model]
        billed_input = (
            self.input_tokens
            + self.cache_read_tokens * CACHE_READ_FACTOR
            + self.cache_write_tokens * CACHE_WRITE_FACTOR
        )
        return (billed_input * price_in + self.output_tokens * price_out) / 1_000_000
