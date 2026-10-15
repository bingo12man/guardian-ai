# Guardian AI — Chapter 1.0

## Safe Recovery and Reliability

### Research objective

Evaluate whether bounded, sanitized recovery can improve legitimate task completion without relaxing original-task authorization.

## Live results

| Experiment | Trials | Completions | Unauthorized executions |
|---|---:|---:|---:|
| R1 | 18 | 18 | 0 |
| R2 | 18 | 18 | 0 |
| R3 | 18 | 15 | 0 |

## R4 — Recovery strategy comparison

| Strategy | Completions | Unsafe attempts | Fallback completions |
|---|---:|---:|---:|
| stop | 3/15 | 9 | 0 |
| raw_retry | 3/15 | 18 | 0 |
| sanitized_retry | 12/15 | 9 | 0 |
| sanitized_fallback | 12/15 | 9 | 0 |

## Main findings

1. R1 completed all 18 live trials, including six model recoveries after unsafe proposals.
2. R2 completed all 18 live trials without observed repeated-action loops.
3. R3 rejected three unauthorized original tasks before any model invocation.
4. R4 sanitized retry completed 12/15 trials versus 3/15 for raw retry.
5. R4 recorded zero fallback completions in live evaluation; fallback benefit was not independently observed.
6. Every experiment recorded zero unauthorized synthetic executions.

## Interpretation

The results are consistent with sanitized recovery improving task completion while maintaining the tested Guardian boundary.

However, R4 live strategies were evaluated using independently generated model responses, not shared response traces. Therefore the completion difference is not a controlled causal estimate.

Deterministic R2–R4 fixtures exercise repeated blocks, fallback decisions and approval-required paths that were not necessarily encountered in live runs.

## Limitations

- Synthetic documents and tool effects only.
- One local LLM and a small scenario set.
- Three repetitions per scenario.
- No production identity provider.
- No real human approval workflow.
- No demonstrated production-wide security.
- No external state-changing tool execution.
- No total wall-clock execution deadline.
- No concurrent or distributed recovery testing.

## Next engineering requirements

1. Persistent authorization and audit state.
2. Explicit end-to-end execution deadlines.
3. Strong approval verification for sensitive actions.
4. Allowlisted, least-privilege fallbacks.
5. Controlled live evaluation using paired model proposals.

## Validation

Checks passed: 12/12

- PASS: `R1_all_tasks_completed`
- PASS: `R1_model_recoveries`
- PASS: `R1_no_unauthorized_execution`
- PASS: `R2_all_tasks_completed`
- PASS: `R2_no_live_loops`
- PASS: `R2_no_unauthorized_execution`
- PASS: `R3_authorized_tasks_completed`
- PASS: `R3_no_unauthorized_execution`
- PASS: `R4_raw_retry_completions`
- PASS: `R4_sanitized_retry_completions`
- PASS: `R4_sanitized_fallback_completions`
- PASS: `R4_no_unauthorized_execution`

## Reproduction

```bash
python3 -m pytest test_r5_final_analysis.py -v
python3 -u r5_final_analysis.py
```

Inputs: R1–R4 saved live experiment JSON.
