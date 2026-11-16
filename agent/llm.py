"""The one place that talks to the Anthropic SDK: a structured call that returns a pydantic object."""

from typing import Protocol, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from agent.config import Settings
from agent.models import Usage

T = TypeVar("T", bound=BaseModel)


class LLMTransientError(Exception):
    """Overload, rate limit or timeout that outlived the SDK's own retries. Try the ticket again later."""


class LLMDeterministicError(Exception):
    """The model refused this ticket or returned an invalid object. Retrying will not help."""


class LLMFatalError(Exception):
    """Wrong key, unknown model or a rejected request. Every ticket would fail, so the run must stop."""


class LLM(Protocol):
    def complete_json(self, system: str, user: str, schema: type[T], model: str) -> tuple[T, Usage]: ...


class LLMClient:
    def __init__(self, settings: Settings):
        self._client = anthropic.Anthropic(api_key=settings.llm_api_key, max_retries=3, timeout=60.0)

    def complete_json(
        self, system: str, user: str, schema: type[T], model: str, max_tokens: int = 6000
    ) -> tuple[T, Usage]:
        """Ask for an object of type `schema`. The system prompt is the stable part, so it is cached."""
        try:
            response = self._client.messages.parse(
                model=model,
                max_tokens=max_tokens,  # thinking is billed as output, so leave room for it
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                output_format=schema,
                output_config={"effort": "low"},
            )
        except anthropic.APIConnectionError as error:  # includes timeouts
            raise LLMTransientError(str(error)) from error
        except anthropic.APIStatusError as error:
            if error.status_code in (408, 409, 429) or error.status_code >= 500:
                raise LLMTransientError(f"{error.status_code}") from error
            # Wrong key, wrong model or a request the API rejects: the next ticket would fail too.
            raise LLMFatalError(f"{error.status_code}: {error.message}") from error
        except ValidationError as error:
            raise LLMDeterministicError("invalid_output") from error

        if response.stop_reason in ("refusal", "max_tokens"):
            raise LLMDeterministicError(response.stop_reason)
        if response.parsed_output is None:
            raise LLMDeterministicError("invalid_output")
        usage = response.usage
        return response.parsed_output, Usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_input_tokens or 0,
            cache_write_tokens=usage.cache_creation_input_tokens or 0,
        )
