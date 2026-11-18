# Eval: review-fixes-3

Model `gpt-6-luna`, 3 run(s). Mean over runs, with min-max when runs differ.
The 10 examples are a development set, not a held-out test: we looked at them while building.

| Metric | Examples (Nordavo labels) | Adversarial (own labels) |
|---|---|---|
| cases | 10 | 10 |
| action_accuracy | 1 | 1 |
| language_accuracy | 1 | 1 |
| category_accuracy | 1 | 1 |
| priority_accuracy | 1 | 1 |
| priority_within_one | 1 | 1 |
| **forbidden_text_posted** | 0 | 0 |
| **unsafe_replies** | 0 | 0 |
| **wrong_closes** | 0 | 0 |
| **over_escalations** | 0 | 0 |
| handoff_rate | 0.3 | 0.5 |
| gate_changes | 0 | 0 |
| faq_match | 1 | 1 |
| reply_quality | n/a | n/a |
| errors | 0 | 0 |
| tokens_in_per_ticket | 3370.67 (3369-3373) | 2832.67 (2831-2834) |
| tokens_out_per_ticket | 369.333 (351-379) | 380.333 (373-387) |
| cost_usd_per_100 | 0.0227 (0.0213-0.0242) | 0.0222 (0.0219-0.0226) |

## Mismatches in the last run

| Case | Expected -> got | Handoff reasons |
|---|---|---|
