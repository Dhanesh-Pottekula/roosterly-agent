# Design notes

The detail behind the one-page [README](../README.md).

## The pipeline, per ticket

1. **Redact** card numbers and IBANs (checksum-validated) from subject and body ([redact.py](../agent/redact.py)). The model, the logs and every post only ever see the redacted text.
2. **Propose**: one structured call returns language, category, priority, rule flags, the customer's requests with a quoted help-centre sentence for each, and a draft ([decide.py](../agent/decide.py)). If the language detector is confident and disagrees with the proposer, the proposer is asked once more with the detector's finding.
3. **Check** the draft deterministically ([checks.py](../agent/checks.py)): language detector, allow-list of web addresses and prices from `faq.md`, no promises or percentages, every quote copied word for word from the help centre.
4. **Verify** a draft that passed: a second call in a fresh context translates it to English and lists unanswered requests, unsupported claims, promises and always-human flags ([verify.py](../agent/verify.py)).
5. **Gate** ([policy.py](../agent/policy.py)): settles the facts (flags, language, priority), then picks the action. The first matching rule wins, and a rule can only move a ticket towards a human.

## Assumptions (the brief left these open)

- Outages and blocking bugs (P1, or a bug at P2) always go to a human, even when the status page could be quoted.
- A ticket with text aimed at the AI still gets its help-centre answer if it is a low-priority how-to; otherwise it is handed off at P3, because its stated urgency cannot be trusted.
- "Does Roosterly support X?" is a handoff when the help centre is silent. A half-answerable ticket is a handoff.
- Telling a customer where they can do or look something up themselves counts as answering, when that is what the help centre says.
- An unsupported language is a handoff, posted as `en`, with the real language named in the note.
- Payment details in a ticket force a handoff at P2 or higher.
- Dutch replies use "je", German replies "Sie".

## When the agent counts itself unsure

Any of: a request without a quoted help-centre sentence, a failed deterministic check, a verifier objection, detector and model still disagreeing on the language, or a model failure. Each one is a handoff, and the log names the reason.

## Reliability in detail

- **503**: nothing was saved, so retry with exponential backoff and jitter (8 tries).
- **429**: requests are spaced at 4 per second, below the limit of 5; if one still comes, wait `Retry-After`.
- **409**: never retried, logged as `already_handled`. A randomised failure test (503s, lost responses, crashes; 25 seeds) never produces one.
- **Timeout or another 5xx on a POST**: the write may have landed, so the agent reads the ticket back and posts only if it is still open.
- **Crash**: run the same command again. The helpdesk is the only state: the agent lists open tickets and re-reads each one before posting. A kill between a post and its log write loses that log line, never the answer.
- **LLM outage or a bug on one ticket**: the ticket stays open and is retried once at the end of the run; if it is still open, the run exits non-zero. A wrong key or unknown model stops the run at once instead of escalating every ticket. A refusal becomes a handoff.

## Security in detail

- Keys come only from the environment (`.env` is git-ignored).
- LLM calls use `store=False`, so the provider keeps no customer text.
- Ticket text is wrapped in `<ticket>` tags and escaped so it cannot close its own tag; the prompts say it is data.
- Text aimed at the AI is flagged by the model and by a pattern for common override phrases. A flagged ticket cannot raise its own priority, and a paraphrase that fools both still needs a quoted help-centre sentence and a clean verification before anything is sent.
- The model has no tools: it can only return the structured object.
- Redaction runs before the model and again on everything posted or logged. An IBAN with a typo fails the checksum and is not recognised.
- Logs hold codes and labels, never customer text.

## Cost formula

Per ticket: (uncached input x input price + cached input x cached price + output x output price) / 1,000,000, summed over the proposer and, for replies, the verifier. Prices are in [config.py](../agent/config.py); every run appends its measured spend to [logs/spend.jsonl](../logs/spend.jsonl).
