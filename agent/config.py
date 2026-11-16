"""Settings come from environment variables only (a local .env is loaded for convenience)."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# USD per million tokens (input, output). Source: Anthropic price page, checked 2026-11.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
}


class ConfigError(Exception):
    """A required environment variable is missing."""


@dataclass(frozen=True)
class Settings:
    helpdesk_url: str
    api_key: str
    email: str
    llm_api_key: str
    name: str | None = None
    llm_model: str = "claude-haiku-5-5"
    eval_judge_model: str = "claude-sonnet-5-5"
    rps: float = 4.0  # the helpdesk allows 5 requests per second; stay below it


def load_settings(dotenv: bool = True, require_llm: bool = True) -> Settings:
    """Read settings from the environment and report every missing variable at once."""
    if dotenv:
        load_dotenv()
    required = ["HELPDESK_URL", "HELPDESK_API_KEY", "CANDIDATE_EMAIL"]
    if require_llm:
        required.append("ANTHROPIC_API_KEY")
    missing = [var for var in required if not os.environ.get(var)]
    if missing:
        raise ConfigError(f"Missing environment variables: {', '.join(missing)}. See .env.example.")
    env = os.environ
    return Settings(
        helpdesk_url=env["HELPDESK_URL"].rstrip("/"),
        api_key=env["HELPDESK_API_KEY"],
        email=env["CANDIDATE_EMAIL"],
        llm_api_key=env.get("ANTHROPIC_API_KEY", ""),
        name=env.get("CANDIDATE_NAME") or None,
        llm_model=env.get("LLM_MODEL") or Settings.llm_model,
        eval_judge_model=env.get("EVAL_JUDGE_MODEL") or Settings.eval_judge_model,
    )
