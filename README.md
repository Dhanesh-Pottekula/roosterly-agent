# Roosterly support ticket agent

Reads every open helpdesk ticket and posts one decision per ticket: reply from the help centre, hand off to a human, or close.

## How to run

```bash
python3 -m venv .venv && source .venv/bin/activate   # Python 3.12+, tested on 3.14
pip install -r requirements.txt
cp .env.example .env                                 # helpdesk key, your email, OPENAI_API_KEY
python -m agent run                                  # handle every open ticket (--dry-run posts nothing)
python -m agent status                               # open / handled counts
python -m eval.run_eval --tag baseline --runs 3 --judge
pytest                                               # 250 tests, no network
```

## Decisions

1. **The LLM proposes, code decides.** One structured call to `gpt-6-luna` proposes a decision ([decide.py](agent/decide.py)); a deterministic gate ([policy.py](agent/policy.py)) posts it and can only move a ticket towards a human. The client's always-human rules live in one table ([rules.py](agent/rules.py)) that feeds the prompts and the gate.
2. **Dutch: work in the customer's language, verify in English.** The model reads and writes Dutch directly (translating first loses menu names and tone). Since I cannot read Dutch, a reply is sent only if it passes deterministic checks (language, addresses and prices from `faq.md`, no promises, a help-centre sentence quoted word for word per request) and an independent verifier: a second call that translates the draft to English and re-checks it. Cost: one extra call per reply.
3. **The helpdesk is the only memory.** No local state: the agent lists open tickets and re-reads each one before posting, so a re-run cannot answer twice.

**Unsure** (any failed check, verifier objection, missing quote or language doubt) means handoff. **Assumptions** where the brief was open: outages and blocking bugs go to a human; a "does it support X?" the help centre is silent on is a handoff; an unsupported language is posted as `en`. Full list in [docs/design.md](docs/design.md).

**AI tools:** I built this with Claude Code: it turned my reading of the brief into a plan, wrote the code test-first, and ran review passes that I checked and acted on. I judged every Dutch reply through the verifier's English translation, never by reading the Dutch.

## Reliability

- **503**: nothing was saved; retry with exponential backoff and jitter. **429**: calls are spaced at 4/s; if one comes anyway, wait `Retry-After`.
- **409**: never retried, logged as `already_handled`. After a timeout on a POST the agent reads the ticket back and posts only if it is still open.
- **Crash**: run the same command again; tests inject crashes and lost responses and check that no ticket gets two answers.
- **Security**: keys from the environment; `store=False` on LLM calls; ticket text wrapped and escaped as data; text aimed at the AI is flagged and cannot raise priority; card numbers and IBANs are redacted before the model and again on everything posted or logged; logs hold codes, not customer text. More in [docs/design.md](docs/design.md).

## Eval and fix

Nordavo's 10 labelled examples plus 10 own adversarial cases ([adversarial.jsonl](eval/adversarial.jsonl)), run through the real decide path. Labels are never written by the tested model. Mean of 3 runs on the 10 examples ([results](eval/results/)):

| Metric | Baseline | Fix 1 | Fix 2 |
|---|---|---|---|
| Unsafe replies / forbidden text / wrong closes | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Action accuracy | 0.87 | 0.83 | **0.93** |
| Priority accuracy | 0.90 | **1.00** | **1.00** |
| Answerable tickets sent to a human | 1.33 | 1.67 | **0.67** |
| Correct help-centre entry cited | 0.89 | 0.94 | **1.00** |

**The fix** (`231c719`): a bug the help centre fully explains (a device requirement not met) was rated P2, and the gate sends P2 bugs to a human. The brief's P3 is "wrong, but there is a workaround"; the prompt now says such an explanation is that workaround. Priority reached 1.00, but those tickets then failed on a second cause, so **fix 2** (`265dcdb`): "here is where you do it yourself" now counts as an answer for both proposer and verifier. A later fix (`07335c5`) re-asks the proposer when it and the language detector disagree: German help-centre questions went from 0/2 to 2/2 replies in a dry run on the practice tickets. With 10 cases, one case is 0.1: read these as directions.

**Cost per 100 tickets: $0.023.** Per ticket (measured over 60): (125 uncached input x $0.10 + 2,803 cached input x $0.01 + 377 output x $0.50) / 1,000,000 = $0.00023, verifier included. Without the prompt cache: $0.048. Everything, judge included, cost under $0.20 ([spend](logs/spend.jsonl)).

## What I would improve

- **Next step:** a held-out test set. The 10 examples shaped the fixes, so they no longer measure fairly: I would have the support team label 50 real tickets, and a Dutch speaker spot-check a weekly sample of sent replies.
- **One change to the rules:** `close` for an empty ticket leaves a real customer with no answer. I would send a fixed, pre-approved reply asking what they need, and keep `close` for spam.
