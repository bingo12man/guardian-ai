# Guardian AI — Module 1 Final Report

## Multi-Model Autonomous Agent Lab

Models: Llama 3.2:3B and Qwen 2.5:3B.

The results cover single-action proposals, multi-step tasks and untrusted memory.

## C1 — Single-action baseline

| Model | Correct proposals | Wrong-target proposals | Invalid outputs |
|---|---:|---:|---:|
| llama3.2:3b | 18/18 | 0 | 0 |
| qwen2.5:3b | 18/18 | 0 | 0 |

## C2 — Multi-step baseline

| Model | Completed | Unsafe proposals | Blocks | Budget exhaustion |
|---|---:|---:|---:|---:|
| llama3.2:3b | 12/24 | 12 | 27 | 12 |
| qwen2.5:3b | 18/24 | 6 | 18 | 6 |

## C3 — Memory conditions

| Model | Mode | Completed | Unsafe proposals | Blocks |
|---|---|---:|---:|---:|
| llama3.2:3b | no_memory | 9/24 | 15 | 57 |
| llama3.2:3b | raw_memory | 9/24 | 15 | 48 |
| llama3.2:3b | structured_memory | 12/24 | 12 | 42 |
| qwen2.5:3b | no_memory | 24/24 | 9 | 18 |
| qwen2.5:3b | raw_memory | 12/24 | 21 | 45 |
| qwen2.5:3b | structured_memory | 21/24 | 6 | 18 |

## Primary findings

1. Correct single-action proposals did not guarantee reliable multi-step task completion.
2. Raw stored memory was associated with lower completion and more unsafe proposals for Qwen in these scenarios.
3. Structured memory was associated with better completion than raw memory for both evaluated models.
4. The synthetic action gateway prevented recorded wrong-target executions.

## Error taxonomy

The benchmark distinguishes wrong-target proposals, blocked search actions, invalid schemas, workflow errors and step-budget exhaustion.

Unsafe proposals are not equivalent to successful unauthorized executions.

## Scientific limitations

- Only two local small language models.
- Eight authored multi-step scenarios.
- Three repetitions per scenario.
- Independent live calls across memory modes.
- No proven causal impact of structured memory.
- No persistent training or model weight updates.
- No genuine document disclosure.
- No cross-agent learning demonstrated yet.
- No held-out adversarial benchmark at scale.

## Module 2 handoff

The accompanying MODULE_2_DATA_CONTRACT.json defines a synthetic scenario record with explicit separation of trusted task, permission ground truth, untrusted inputs and attack labels.

Module 2 should add versioned environment state, action traces, deterministic splits, automated policy-label verification, and separate observed outcomes from ground truth.

## Validation

Passed: 12/12 checks.
