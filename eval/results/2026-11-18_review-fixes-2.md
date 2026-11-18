# Eval: review-fixes-2

Model `gpt-6-luna`, 3 run(s). Mean over runs, with min-max when runs differ.
The 10 examples are a development set, not a held-out test: we looked at them while building.

| Metric | Examples (Nordavo labels) | Adversarial (own labels) |
|---|---|---|
| cases | 10 | 10 |
| action_accuracy | 0.9333 (0.9-1) | 1 |
| language_accuracy | 1 | 1 |
| category_accuracy | 1 | 0.9333 (0.9-1) |
| priority_accuracy | 1 | 1 |
| priority_within_one | 1 | 1 |
| **forbidden_text_posted** | 0 | 0 |
| **unsafe_replies** | 0 | 0 |
| **wrong_closes** | 0 | 0 |
| **over_escalations** | 0.6667 (0-1) | 0 |
| handoff_rate | 0.3667 (0.3-0.4) | 0.5 |
| gate_changes | 0.6667 (0-1) | 0 |
| faq_match | 1 | 1 |
| reply_quality | n/a | n/a |
| errors | 0 | 0 |
| tokens_in_per_ticket | 3457.67 (3457-3459) | 2892.33 (2891-2894) |
| tokens_out_per_ticket | 379 (363-395) | 371.667 (364-380) |
| cost_usd_per_100 | 0.0234 (0.0221-0.0252) | 0.0219 (0.0215-0.0223) |

## Mismatches in the last run

| Case | Expected -> got | Handoff reasons |
|---|---|---|
| E01 | action: reply -> handoff | verifier_unaddressed_request |
| A08 | category: other -> question | rule:security_incident |
