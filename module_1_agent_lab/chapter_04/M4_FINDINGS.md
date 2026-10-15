# Guardian AI — Module 1, Chapter 1.4

## Cross-Model Behavior Profiling

### Objective

Analyze single-step, multi-step and memory agent behavior using existing saved live results from Llama 3.2:3B and Qwen 2.5:3B.

## C1: Single-action baseline

| Model | Correct / Trials | Invalid | Wrong target |
|---|---:|---:|---:|
| llama3.2:3b | 18/18 | 0 | 0 |
| qwen2.5:3b | 18/18 | 0 | 0 |

## C2: Multi-step performance

| Model | Completions | Unsafe proposals | Blocks | Budget exhaustion |
|---|---:|---:|---:|---:|
| llama3.2:3b | 12/24 | 12 | 27 | 12 |
| qwen2.5:3b | 18/24 | 6 | 18 | 6 |

## C3: Memory conditions

| Model | Memory | Completion | Unsafe proposals | Recovered after unsafe |
|---|---|---:|---:|---:|
| llama3.2:3b | no_memory | 9/24 | 15 | 0 |
| llama3.2:3b | raw_memory | 9/24 | 15 | 0 |
| llama3.2:3b | structured_memory | 12/24 | 12 | 0 |
| qwen2.5:3b | no_memory | 24/24 | 9 | 9 |
| qwen2.5:3b | raw_memory | 12/24 | 21 | 3 |
| qwen2.5:3b | structured_memory | 21/24 | 6 | 3 |

## Decision failure taxonomy

### llama3.2:3b: C2 trace reasons

| Reason | Events |
|---|---:|
| INVALID_OUTPUT | 3 |
| INVALID_SEARCH | 12 |
| READ_OK | 24 |
| SEARCH_OK | 24 |
| TARGET_MISMATCH | 12 |
| TASK_COMPLETED | 12 |

### qwen2.5:3b: C2 trace reasons

| Reason | Events |
|---|---:|
| INVALID_SEARCH | 12 |
| READ_OK | 18 |
| SEARCH_OK | 24 |
| TARGET_MISMATCH | 6 |
| TASK_COMPLETED | 18 |

## Interpretation

- C1's single-action success did not imply equivalent multi-step reliability.
- Wrong-target proposals must be separated from tool-order and schema violations.
- Structured memory contains only a restricted set of trusted fields, while raw memory carries attacker-controlled text.
- A completion after an unsafe proposal is an episode recovery, not evidence of durable learning.
- The experiments do not demonstrate actual unauthorized document disclosure.

## Limitations

- Small authored scenario set.
- Synthetic documents and local tool execution.
- Only two small Ollama models.
- No cross-model statistical generalization.
- Differences in prompts and sampling can affect observed memory-condition results.
- Error reason counts are tool-event counts, not numbers of distinct affected episodes.
- Chapter 1.3 uses a larger step budget than Chapter 1.2, so direct cross-chapter completion comparisons should be treated cautiously.

## Next

Chapter 1.5 will consolidate comparable model baselines and define the standardized agent interface for Module 2.
