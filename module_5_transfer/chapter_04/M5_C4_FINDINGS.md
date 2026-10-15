# Guardian AI — Module 5, Chapter 5.4

## Negative Transfer and Safety Regression

Source: completed live Chapter 5.3 experiment.

| Recipient | Condition | Completion gains | Completion losses | Net delta | Exact p-value |
|---|---|---:|---:|---:|---:|
| llama3.2:3b | same_model | 2 | 2 | +0 | 1.0000 |
| llama3.2:3b | cross_model | 0 | 0 | +0 | 1.0000 |
| qwen2.5:3b | same_model | 0 | 1 | -1 | 1.0000 |
| qwen2.5:3b | cross_model | 0 | 1 | -1 | 1.0000 |

## Safety audit

- Episodes validated: 180
- Unauthorized synthetic executions: 0
- Invalid saved audit flags: 0

## Interpretation

Pair-level regressions are descriptive associations. More rejected proposals do not necessarily imply fewer completed tasks, but do indicate additional friction.

Lesson hit rates are affected by donor training evidence coverage. They cannot be interpreted as lesson effectiveness.

## Important limitations

- Separate model calls per condition
- One repetition per scenario
- Shared attack templates across splits
- Controller-authored correction text
- Potential donor coverage confounding
- In-process Guardian prototype
- Gateway audit verification represented by saved producer flags

## Chapter 5.5 handoff

Consolidate model-specific performance, donor availability, negative transfer cases and statistical uncertainty before making cross-agent learning claims.
