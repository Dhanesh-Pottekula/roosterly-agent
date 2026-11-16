"""Offline eval: run the agent's decide path on labelled tickets and score it.

    python -m eval.run_eval --tag baseline --runs 3 --judge

Nothing is posted to the helpdesk. The labels come from Nordavo (eval/examples.json, cached
from GET /examples) and from our own extra cases (eval/adversarial.jsonl). The agent never
sees a label, and no example is used in a prompt.
"""

import argparse
import json
from datetime import date
from pathlib import Path

from agent.config import load_settings
from agent.faq import KnowledgeBase, load_kb
from agent.helpdesk import HelpdeskClient
from agent.llm import LLM, LLMClient
from agent.models import Ticket, Usage
from agent.pipeline import decide_ticket
from agent.runlog import append_spend
from eval.judge import judge_reply
from eval.scoring import Case, CaseResult, expected_faq_id, score

EVAL_DIR = Path(__file__).parent
SAFETY_METRICS = ("unsafe_replies", "wrong_closes", "forbidden_text_posted", "over_escalations")


def load_cases(eval_dir: Path = EVAL_DIR) -> list[Case]:
    examples = json.loads((eval_dir / "examples.json").read_text(encoding="utf-8"))["examples"]
    lines = (eval_dir / "adversarial.jsonl").read_text(encoding="utf-8").splitlines()
    adversarial = [json.loads(line) for line in lines if line.strip()]
    return [
        Case(
            id=item["id"],
            group=group,
            label=item["label"],
            never_post=tuple(item.get("never_post", ())),
            never_reply=tuple(item.get("never_reply", ())),
            ticket={k: v for k, v in item.items() if k not in ("label", "never_post", "never_reply")},
        )
        for group, items in (("examples", examples), ("adversarial", adversarial))
        for item in items
    ]


def run_once(
    cases: list[Case], kb: KnowledgeBase, llm: LLM, model: str, judge_model: str | None
) -> tuple[list[CaseResult], Usage]:
    """Decide every case once. Returns the results and what the judge cost."""
    results, judge_usage = [], Usage()
    for case in cases:
        result = CaseResult(case, decide_ticket(Ticket(**case.ticket), kb, llm, model))
        decision = result.outcome.decision
        faq_id = expected_faq_id(case.label)
        if judge_model and decision and decision.action == "reply" and faq_id in kb.entries:
            question = f"{case.ticket.get('subject', '')}\n{case.ticket.get('body', '')}"
            result.quality, usage = judge_reply(
                question, kb.entries[faq_id].text, decision.draft_reply, llm, judge_model
            )
            judge_usage += usage
        results.append(result)
    return results, judge_usage


def summarise(runs: list[list[CaseResult]], model: str) -> dict:
    """Per group and metric: the mean, minimum and maximum over the runs."""
    summary = {}
    for group in ("examples", "adversarial"):
        per_run = [score([r for r in run if r.case.group == group], model) for run in runs]
        summary[group] = {}
        for metric in per_run[0]:
            values = [scores[metric] for scores in per_run if scores[metric] is not None]
            summary[group][metric] = (
                {"mean": round(sum(values) / len(values), 4), "min": min(values), "max": max(values)}
                if values
                else None
            )
    return summary


def case_details(results: list[CaseResult]) -> list[dict]:
    """One run in full: what was expected, what was decided, and each reply with its English translation."""
    details = []
    for result in results:
        outcome, decision = result.outcome, result.outcome.decision
        details.append(
            {
                "id": result.case.id,
                "expected": {k: result.case.label[k] for k in ("language", "category", "priority", "action")},
                "got": decision.model_dump(exclude_none=True) if decision else None,
                "draft_in_english": outcome.verdict.translation_en if outcome.verdict else None,
                "handoff_reasons": outcome.handoff_reasons,
                "mismatches": result.mismatches,
                "forbidden_text_posted": result.forbidden_text_posted,
                "judge_quality": result.quality,
            }
        )
    return details


def to_markdown(tag: str, model: str, runs: list[list[CaseResult]], summary: dict) -> str:
    def cell(stats: dict | None) -> str:
        if stats is None:
            return "n/a"
        return (
            f"{stats['mean']:g}"
            if stats["min"] == stats["max"]
            else f"{stats['mean']:g} ({stats['min']:g}-{stats['max']:g})"
        )

    lines = [
        f"# Eval: {tag}",
        "",
        f"Model `{model}`, {len(runs)} run(s). Mean over runs, with min-max when runs differ.",
        "The 10 examples are a development set, not a held-out test: we looked at them while building.",
        "",
        "| Metric | Examples (Nordavo labels) | Adversarial (own labels) |",
        "|---|---|---|",
    ]
    for metric in summary["examples"]:
        name = f"**{metric}**" if metric in SAFETY_METRICS else metric
        lines.append(f"| {name} | {cell(summary['examples'][metric])} | {cell(summary['adversarial'][metric])} |")
    lines += ["", "## Mismatches in the last run", "", "| Case | Expected -> got | Handoff reasons |", "|---|---|---|"]
    for result in runs[-1]:
        if result.mismatches:
            wrong = "; ".join(f"{field}: {change}" for field, change in result.mismatches.items())
            lines.append(f"| {result.case.id} | {wrong} | {', '.join(result.outcome.handoff_reasons)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m eval.run_eval")
    parser.add_argument("--tag", default="dev", help="name for the result files, e.g. baseline or after-fix")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--judge", action="store_true", help="also grade reply quality with the judge model")
    parser.add_argument("--refresh", action="store_true", help="download the labelled examples again")
    args = parser.parse_args()

    settings = load_settings()
    if args.refresh or not (EVAL_DIR / "examples.json").exists():
        examples = HelpdeskClient(settings).examples()
        (EVAL_DIR / "examples.json").write_text(
            json.dumps({"examples": examples}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    kb, llm, cases = load_kb(), LLMClient(settings), load_cases()
    judge_model = settings.eval_judge_model if args.judge else None
    runs, judge_usage = [], Usage()
    for _ in range(args.runs):
        results, usage = run_once(cases, kb, llm, settings.llm_model, judge_model)
        runs.append(results)
        judge_usage += usage

    summary = summarise(runs, settings.llm_model)
    stem = EVAL_DIR / "results" / f"{date.today().isoformat()}_{args.tag}"
    stem.parent.mkdir(exist_ok=True)
    stem.with_suffix(".json").write_text(
        json.dumps(
            {
                "tag": args.tag,
                "model": settings.llm_model,
                "runs": args.runs,
                "summary": summary,
                "last_run": case_details(runs[-1]),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    report = to_markdown(args.tag, settings.llm_model, runs, summary)
    stem.with_suffix(".md").write_text(report, encoding="utf-8")
    print(report)

    agent_usage = sum((r.outcome.usage for run in runs for r in run), Usage())
    total = append_spend("eval", args.tag, agent_usage, settings.llm_model)
    if args.judge:
        total = append_spend("eval-judge", args.tag, judge_usage, settings.eval_judge_model)
    print(f"Total LLM spend so far: ${total:.4f}")


if __name__ == "__main__":
    main()
