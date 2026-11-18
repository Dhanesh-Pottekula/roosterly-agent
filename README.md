# Roosterly support ticket agent

Reads every open helpdesk ticket and posts one decision each: reply from the help centre, hand off to a human, or close.

## How to run

```bash
python3 -m venv .venv && source .venv/bin/activate   # Python 3.12+, tested on 3.14
pip install -r requirements.txt
cp .env.example .env              # helpdesk key, your email, OPENAI_API_KEY
python -m agent run               # --dry-run posts nothing
python -m eval.run_eval --tag baseline --runs 3 --judge
pytest                            # 264 tests, no network
```

## Decisions

1. **The LLM proposes, code decides.** One structured call to `gpt-6-luna` proposes a decision ([decide.py](agent/decide.py)). A deterministic gate ([policy.py](agent/policy.py)) can only move a ticket towards a human. The always-human rules are one table ([rules.py](agent/rules.py)) that feeds the prompts and the gate.
2. **Dutch: written in the customer's language, verified in English.** Translating first loses menu names and tone. I can't read Dutch, so a reply is only sent if it passes deterministic checks (language; links and prices from `faq.md`; no promises; a help-centre sentence quoted word for word for each request) and a second, independent call that translates it to English and re-checks it. That costs one extra call per reply.
3. **The helpdesk is the only memory.** There is no local state. Tickets are re-read before posting, so a re-run can't answer twice.

**Unsure** (a failed check, a verifier objection or a language doubt) means handoff. **Assumptions:** outages, blocking bugs and "does it support X?" questions the help centre is silent on go to a human; an unsupported language is posted as `en` ([docs/design.md](docs/design.md)). **AI tools:** I built this with Claude Code. It turned my reading of the brief into a plan, wrote the code test-first and ran reviews that I checked. I judged Dutch replies through the verifier's English translation.

## Reliability

- **503:** retry with backoff and jitter. **429:** calls are spaced at 4/s, and a 429 waits `Retry-After`.
- **409:** never retried. A POST that times out is read back, and the agent posts only if the ticket is still open.
- **Crash:** run the same command again. Tests inject crashes and lost responses; no ticket gets two answers.
- **Security:** keys come from the environment; ticket text is escaped as data, and instructions in it can't raise priority. Card numbers and IBANs are redacted before the model and in everything posted or logged. Logs hold codes, never customer text.

## Eval and fix

The eval covers Nordavo's 10 labelled examples, plus 10 adversarial cases I wrote ([adversarial.jsonl](eval/adversarial.jsonl)). Mean of 3 runs on the examples ([results](eval/results/)):

| Metric | Baseline | After fixes |
|---|---|---|
| Unsafe replies / wrong closes | 0 / 0 | 0 / 0 |
| Action accuracy | 0.87 | **0.97** |
| Priority accuracy | 0.90 | **1.00** |
| Answerable tickets sent to a human | 1.33 | **0.33** |

**The fix** (`231c719`, `265dcdb`): a bug the help centre fully explains (an unmet device requirement) was rated P2, so it went to a human. The prompt now treats that explanation as the brief's P3 "workaround", and "here is where you do it yourself" counts as an answer.

**A second fix** (`1222d60`) came from the practice run: 3 answerable tickets were handed off because the checks tripped on wording that varies between runs, such as "we can't promise a date". The final run went from 9 to 11 replies, with every always-human ticket still handed off. With 10 cases, one case is 0.1, so read these numbers as directions.

**Cost per 100 tickets: $0.023.** Per ticket, measured: (125 uncached input × $0.10 + 2,803 cached input × $0.01 + 377 output × $0.50) / 1M = $0.00023, verifier included. Total spend was under $0.20 ([spend.jsonl](logs/spend.jsonl)).

## What I would improve

- **Next step:** a held-out test set. The 10 examples shaped the fixes, so I'd have the support team label 50 real tickets and a Dutch speaker spot-check sent replies weekly.
- **One change to the rules:** `close` on an empty ticket leaves a real customer unanswered. I'd send a pre-approved "what do you need?" reply instead, and keep `close` for spam.
