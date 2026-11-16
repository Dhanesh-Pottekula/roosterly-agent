"""The client's "always a human" rules, in one place.

Each rule has a definition (shown to the proposer and to the verifier), the priority a
matching ticket gets, and the wording used in the posted reason. Adding a rule means adding
its name to RuleFlag and one entry to ALWAYS_HUMAN.
"""

from dataclasses import dataclass
from typing import Literal

from agent.models import Priority

RuleFlag = Literal[
    "refund_or_dispute",
    "cancellation_or_contract_change",
    "personal_data_request",
    "security_incident",
    "legal_threat",
    "threat_to_leave",
]


@dataclass(frozen=True)
class Rule:
    definition: str
    priority: Priority  # P1 for a security or legal risk, P2 when money or the contract is involved
    label: str


ALWAYS_HUMAN: dict[RuleFlag, Rule] = {
    "refund_or_dispute": Rule(
        "The customer asks for money back, or says a charge or invoice amount is wrong. "
        "NOT a neutral question about prices or where to find invoices.",
        "P2",
        "refund or disputed charge",
    ),
    "cancellation_or_contract_change": Rule(
        "The customer wants to cancel, downgrade, pause or renegotiate the subscription or contract. "
        "NOT a question about how to upgrade or what a plan costs.",
        "P2",
        "cancellation or contract change",
    ),
    "personal_data_request": Rule(
        "A person asks to see, export, correct or delete personal data (GDPR, 'AVG' in Dutch). "
        "NOT a general question about where or how data is stored.",
        "P2",
        "personal data (GDPR/AVG) request",
    ),
    "security_incident": Rule(
        "Possible unauthorised access, a suspicious login, a leak, phishing or a hacked account. "
        "NOT an ordinary forgotten password.",
        "P1",
        "possible security incident",
    ),
    "legal_threat": Rule(
        "The customer mentions a lawyer, legal action, a claim for damages or a regulator.",
        "P1",
        "legal threat",
    ),
    "threat_to_leave": Rule(
        "The customer threatens to leave, cancel or switch to a competitor unless something happens.",
        "P2",
        "customer threatening to leave",
    ),
}


def definitions_text() -> str:
    return "\n".join(f"- {flag}: {rule.definition}" for flag, rule in ALWAYS_HUMAN.items())
