"""An independent judge for reply quality, used only by the eval.

It is a different model from the agent, with its own prompt, so the agent never grades
itself. The judge scores drafts; it never writes or changes a label.
"""

from pydantic import BaseModel

from agent.llm import LLM, LLMDeterministicError, LLMTransientError
from agent.models import Usage

JUDGE_PROMPT = """\
You grade a support reply for a shift-planning app. You get the customer's question, the help \
centre entry that contains the correct answer, and the reply that was drafted. The reply may be \
in Dutch, German or English; grade its content, not its language.

correctness (1-5): 5 means every statement agrees with the help centre entry and nothing is \
invented. 1 means the reply is wrong or misleading.
completeness (1-5): 5 means the customer can act on the reply without asking again. 1 means the \
question is left unanswered.
explanation: one sentence in English.
"""


class Grade(BaseModel):
    correctness: int
    completeness: int
    explanation: str


def judge_reply(question: str, faq_entry: str, draft: str, llm: LLM, model: str) -> tuple[float | None, Usage]:
    """Mean of correctness and completeness, or None if the judge could not grade."""
    user = f"<question>\n{question}\n</question>\n\n<help_centre_entry>\n{faq_entry}\n</help_centre_entry>\n\n<reply>\n{draft}\n</reply>"
    try:
        grade, usage = llm.complete_json(JUDGE_PROMPT, user, Grade, model)
    except (LLMTransientError, LLMDeterministicError):
        return None, Usage()
    return (grade.correctness + grade.completeness) / 2, usage
