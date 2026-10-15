# Guardian AI — Module 6, Chapter 6.1

## Agent risk profiles and curriculum data

Profiles are derived from observed training-split denial frequencies.

| Model | Denials | Practice items | Assessment items | Main weakness |
|---|---:|---:|---:|---|
| llama3.2:3b | 113 | 8 | 4 | workflow_scope_error |
| qwen2.5:3b | 52 | 6 | 3 | workflow_scope_error |

## Curriculum design

- Exercises use synthetic public documents.
- Lessons are grounded in training failures.
- Scoring is deterministic and does not execute tools.
- Assessment uses a different synthetic target but shares exercise templates.

## Scientific limitations

- Denial counts depend on model call volume.
- The profile priority is a within-model frequency, not an estimated risk probability.
- Repeated failures may come from the same scenario and are not independent.
- Generated practice and assessment items share rules and exercise templates.
- No model has practiced these exercises yet.
- No trust recovery has been demonstrated.

## Module 6.2 handoff

Choose exercises adaptively using trusted performance evidence and bounded practice budgets, without changing permissions.
