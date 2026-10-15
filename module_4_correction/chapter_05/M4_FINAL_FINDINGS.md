# Guardian AI — Module 4 Final Findings

## Evaluation status

**FULL**

Scenarios per model: 30

## Paired model outcomes

| Model | Baseline completions | Lesson completions | Net delta | Exact p-value |
|---|---:|---:|---:|---:|
| llama3.2:3b | 15 | 14 | -1 | 1.0000 |
| qwen2.5:3b | 10 | 13 | +3 | 0.3750 |

## Interpretation

Results are scenario-paired, but the model responses are generated independently. An exact sign test on discordant completion pairs quantifies uncertainty; it does not establish causation.

Correction retrieval is distinct from the next action being accepted, and both are distinct from eventual task completion.

## Safety constraints

- No agent may create a capability.
- A lesson cannot expand task permissions.
- Denied proposals cannot trigger tool effects.
- Ground-truth task scope is unchanged.

## Scientific limitations

- Small numbers of authorized tasks.
- Independent model calls, not identical counterfactual model responses.
- Shared attack templates across splits.
- No model-weight changes or durable learning.
- Hash/audit validity partially relies on saved producer assertions.

## Module 5 handoff

Module 5 should test whether training-derived lessons transfer between model identities on held-out scenarios without modifying authorization rules. Preserve explicit donor, recipient and lesson provenance.
