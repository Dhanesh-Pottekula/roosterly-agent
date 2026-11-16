# Roosterly support ticket agent

Reads every open helpdesk ticket and posts one decision per ticket: reply from the help centre, hand off to a human, or close.

## How to run

```bash
python3 -m venv .venv && source .venv/bin/activate    # Python 3.12+, tested on 3.14
pip install -r requirements.txt
cp .env.example .env                                  # fill in the keys and your email
python -m agent run                                   # handle every open ticket (add --dry-run to post nothing)
python -m agent status                                # open / handled counts
python -m eval.run_eval --tag baseline --runs 3 --judge
pytest                                                # 146 tests, no network
```

## Decisions

1. **The LLM proposes, code decides.** One structured call proposes a decision ([decide.py](agent/decide.py)). A deterministic gate ([policy.py](agent/policy.py)) posts it, and can only move a ticket towards a human. The client's "always a human" rules live in one table ([rules.py](agent/rules.py)) that feeds both prompts and the gate.
2. **Dutch: work in the customer's language, verify in English.** The model reads and writes Dutch directly (translating first loses menu names and tone). I cannot read Dutch, so a reply is sent only if it passes ([verify.py](agent/verify.py)): a language detector, an allow-list of addresses and prices from `faq.md`, a ban on promises and percentages, a help-centre sentence quoted word for word for every request, and a second model call in a fresh context that translates the draft back to English and re-checks it. Cost: one extra call per reply. `--dry-run` prints that English translation.
3. **The helpdesk is the only memory.** No local state file: the agent lists open tickets and re-reads each one before deciding, so a re-run cannot answer twice.

**Unsure** means any of: a request without a quoted help-centre sentence, a failed check, a verifier objection, detector and model disagreeing on the language, or a model failure. Each one is a handoff, and the log names it.

**Assumptions** (the brief left these open): outages and blocking bugs (P1, or bug at P2) always go to a human; a ticket with text aimed at the AI still gets its help-centre answer if it is a low-priority how-to, otherwise it is handed off at P3 because its urgency cannot be trusted; "does Roosterly support X?" is a handoff when the help centre is silent; a half-answerable ticket is a handoff; an unsupported language is a handoff posted as `en`; payment details are redacted before the model sees them and force a handoff; Dutch replies use "je", German "Sie".

**AI tools:** TODO (one or two sentences in your own words).

## Reliability

- **503**: nothing was saved, so retry with exponential backoff and jitter (8 tries).
- **429**: requests are spaced at 4 per second, so it should not happen; if it does, wait `Retry-After`.
- **409**: never retried. It is logged as `already_handled`.
- **Timeout or other 5xx on a POST**: the write may have landed. The agent reads the ticket back and posts again only if it is still open.
- **Crash**: run the same command again. Tests inject a crash after 7 posts and check that no ticket gets two answers.
- **LLM outage**: the ticket stays open, is retried once, and the run exits non-zero. A refusal becomes a handoff.
- **Security**: keys only from the environment; ticket text is wrapped as data and cannot raise its own priority; the model has no tools; payment details never reach the model, a reply, a note or a log; logs hold codes, not customer text.

## Eval and fix

TODO after the first real run: before/after table from `eval/results/`, the fix and its commit, and the measured cost.

The eval runs the real decide path offline on Nordavo's 10 labelled examples plus 10 extra adversarial cases in [adversarial.jsonl](eval/adversarial.jsonl). Labels are never written by the tested model, no example appears in a prompt, and unsafe replies are counted separately from accuracy. The 10 examples are a development set, not a held-out test.

Cost per 100 tickets = 100 x (input tokens x $0.10 + output tokens x $0.50) / 1,000,000 per ticket, with cached input at a tenth of the price. TODO: measured figure.

## What I would improve

- **Next step:** TODO.
- **One change to the rules:** TODO.
