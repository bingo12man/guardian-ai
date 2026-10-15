# Guardian AI — Module 6 Final Findings

## Research question

Does adaptive synthetic safety practice
improve agent behavior and support
evidence-based trust recovery?

## Observed live results

| Model | Initial | Practice | Final | Review state |
|---|---:|---:|---:|---|
| llama3.2:3b | 4/4 | 5/8 | 4/4 | PRACTICE_REQUIRED |
| qwen2.5:3b | 0/3 | 3/8 | 3/3 | PRACTICE_REQUIRED |

## Scientific interpretation

Qwen improved from zero to three correct
on the same repeated assessment items.
Llama stayed at four out of four.

The results show a within-session change
but do not establish a causal effect
of adaptive curriculum practice.

## Critical limitations

- One live session per model
- Only four Llama and three Qwen assessment items
- The initial and final items were identical
- Final assessment included practice context
- No no-practice or fixed-practice control
- Practice and assessment reused rule templates
- Scores are recorded by an in-process scorer
- Saved audit flags do not authenticate all
  underlying raw model responses
- Model weights were not updated
- No durable learning was demonstrated

## Authorization outcome

Both agents remain PRACTICE_REQUIRED.
No Guardian permissions or privileges changed.

## Module 7 handoff

Investigate verified, deduplicated collective
observations with isolated-agent controls.
Keep shared safety knowledge separate
from authorization and tool execution.
