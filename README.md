# Roosterly support ticket agent

Reads every open helpdesk ticket and posts one decision per ticket: reply from the help centre, hand off to a human, or close.

## How to run

```bash
python3 -m venv .venv && source .venv/bin/activate    # Python 3.12+, tested on 3.14
pip install -r requirements.txt
cp .env.example .env                                  # helpdesk key, your email, OPENAI_API_KEY
python -m agent run                                   # handle every open ticket (add --dry-run to post nothing)
python -m agent status                                # open / handled counts
python -m eval.run_eval --tag baseline --runs 3 --judge
pytest                                                # 247 tests, no network
```

## Decisions

1. **The LLM proposes, code decides.** One structured call to `gpt-6-luna` (OpenAI's small model, $0.10/$0.50 per million tokens) proposes a decision ([decide.py](agent/decide.py)). A deterministic gate ([policy.py](agent/policy.py)) posts it, and can only move a ticket towards a human. The client's "always a human" rules live in one table ([rules.py](agent/rules.py)) that feeds both prompts and the gate.
2. **Dutch: work in the customer's language, verify in English.** The model reads and writes Dutch directly (translating first loses menu names and tone). I cannot read Dutch, so a reply is sent only if it passes deterministic checks ([checks.py](agent/checks.py): language detector, allow-list of addresses and prices from `faq.md`, no promises or percentages, a help-centre sentence quoted word for word for every request) and an independent verifier ([verify.py](agent/verify.py)): a second model call in a fresh context that translates the draft back to English and re-checks it. Cost: one extra call per reply. `--dry-run` prints that English translation.
3. **The helpdesk is the only memory.** No local state file: the agent lists open tickets and re-reads each one before deciding, so a re-run cannot answer twice.

**Unsure** means any of: a request without a quoted help-centre sentence, a failed check, a verifier objection, detector and model disagreeing on the language, or a model failure. Each one is a handoff, and the log names it.

**Assumptions** (the brief left these open): outages and blocking bugs (P1, or bug at P2) always go to a human; a ticket with text aimed at the AI still gets its help-centre answer if it is a low-priority how-to, otherwise it is handed off at P3 because its urgency cannot be trusted; "does Roosterly support X?" is a handoff when the help centre is silent; a half-answerable ticket is a handoff; an unsupported language is a handoff posted as `en`; payment details are redacted before the model sees them and force a handoff; Dutch replies use "je", German "Sie".

**AI tools:** I built this with Claude Code: it turned my reading of the brief into a plan, wrote the code test-first, and ran review passes that I checked and acted on. Every Dutch reply I judged through the verifier's English translation, never by reading the Dutch.

## Reliability

- **503**: nothing was saved, so retry with exponential backoff and jitter (8 tries).
- **429**: requests are spaced at 4 per second, so it should not happen; if it does, wait `Retry-After`.
- **409**: never retried. It is logged as `already_handled`. A randomised failure test (503s, lost responses, crashes; 25 seeds) never produces one.
- **Timeout or other 5xx on a POST**: the write may have landed. The agent reads the ticket back and posts again only if it is still open.
- **Crash**: run the same command again. Tests inject crashes and check that no ticket gets two answers. A kill in the instant between a post and its log write loses that log line, never the answer.
- **LLM outage or a bug on one ticket**: the ticket stays open, is retried once, and the run exits non-zero; the rest of the queue continues. A refusal becomes a handoff.
- **Security**: keys only from the environment; LLM calls use `store=False`, so the provider keeps no customer text; ticket text is wrapped as data; text aimed at the AI is flagged by the model and by a pattern for common override phrases, and a flagged ticket cannot raise its priority (a paraphrase that fools both still ends in a handoff, never a reply); the model has no tools; card numbers and IBANs are redacted by checksum before the model sees the ticket and again on everything posted or logged (an IBAN with a typo is not recognised); logs hold codes, not customer text.

## Eval and fix

The eval runs the real decide path offline on Nordavo's 10 labelled examples plus 10 extra adversarial cases in [adversarial.jsonl](eval/adversarial.jsonl). Labels are never written by the tested model, no example appears in a prompt, and unsafe replies are counted separately from accuracy. The 10 examples are a development set, not a held-out test.

Nordavo's 10 examples, mean of 3 runs ([results](eval/results/)):

| Metric | Baseline | Fix 1 | Fix 2 |
|---|---|---|---|
| Unsafe replies / forbidden text / wrong closes | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Action accuracy | 0.87 | 0.83 | **0.93** |
| Priority accuracy | 0.90 | **1.00** | **1.00** |
| Over-escalations (answerable, sent to a human) | 1.33 | 1.67 | **0.67** |
| Correct help-centre entry cited | 0.89 | 0.94 | **1.00** |
| Reply quality, judged by `gpt-6.1-terra` (1-5) | 4.93 | 5.00 | 4.91 |

The adversarial set stayed at 1.00 action accuracy and 0 unsafe replies throughout.

**The fix** (`231c719`): the model rated a bug the help centre fully explains (a device requirement not met) as P2, and the gate sends every P2 bug to a human. The brief's P3 is "wrong, but there is a workaround"; the prompt now says a help-centre explanation with a way out is that workaround. Priority reached 1.00, but the same tickets were then stopped by a second cause, so **fix 2** (`265dcdb`): the proposer and the verifier treated "here is where you do it yourself" as unanswered; both prompts now count it as the answer. One case is 0.1, so read these as directions, not decimals.

**Cost per 100 tickets: $0.023.** Measured over 60 eval tickets, per ticket: (125 uncached input x $0.10 + 2,803 cached input x $0.01 + 377 output x $0.50) / 1,000,000 = $0.00023, proposer and verifier included. Without the prompt cache: $0.048. The whole assignment, judge included, cost under $0.20 ([spend log](logs/spend.jsonl)).

## What I would improve

- **Next step:** a held-out test set. The 10 examples shaped both fixes, so they no longer measure fairly: I would have the support team label 50 real tickets, and a Dutch speaker spot-check a weekly sample of sent replies.
- **One change to the rules:** `close` for an empty ticket leaves a real customer with no answer. I would send a fixed, pre-approved reply asking what they need, and keep `close` for spam only.
