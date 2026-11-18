"""The one place that talks to the OpenAI SDK: a structured call that returns a pydantic object."""

import hashlib
from typing import Protocol, TypeVar

import openai
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
        self._client = openai.OpenAI(api_key=settings.llm_api_key, max_retries=3, timeout=60.0)

    def complete_json(
        self, system: str, user: str, schema: type[T], model: str, max_tokens: int = 6000
    ) -> tuple[T, Usage]:
        """Ask for an object of type `schema` (strict structured output).

        The system prompt comes first and never changes within a run, so OpenAI caches it;
        the cache key keeps calls with the same prompt on the same cache.
        """
        try:
            response = self._client.responses.parse(
                model=model,
                instructions=system,
                input=[{"role": "user", "content": user}],
                text_format=schema,
                reasoning={"effort": "low"},
                max_output_tokens=max_tokens,  # reasoning is billed as output, so leave room for it
                prompt_cache_key=hashlib.sha256(system.encode()).hexdigest()[:32],
                store=False,  # customer text is not kept on OpenAI's side
            )
        except openai.APIConnectionError as error:  # includes timeouts
            raise LLMTransientError(str(error)) from error
        except openai.APIStatusError as error:
            if error.status_code in (408, 409, 429) or error.status_code >= 500:
                raise LLMTransientError(f"{error.status_code}") from error
            # Wrong key, wrong model or a request the API rejects: the next ticket would fail too.
            raise LLMFatalError(f"{error.status_code}: {error.message}") from error
        except (ValidationError, ValueError) as error:  # truncated or malformed JSON
            raise LLMDeterministicError("invalid_output") from error

        if response.status == "incomplete":
            reason = getattr(response.incomplete_details, "reason", None) or "incomplete"
            raise LLMDeterministicError(reason)
        if _refused(response):
            raise LLMDeterministicError("refusal")
        if response.output_parsed is None:
            raise LLMDeterministicError("invalid_output")
        usage = response.usage
        cached = getattr(usage.input_tokens_details, "cached_tokens", 0) or 0
        return response.output_parsed, Usage(
            input_tokens=usage.input_tokens - cached,  # OpenAI counts cached tokens inside input_tokens
            output_tokens=usage.output_tokens,
            cache_read_tokens=cached,
        )


def _refused(response) -> bool:
    for item in response.output or []:
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", None) == "refusal":
                return True
    return False
