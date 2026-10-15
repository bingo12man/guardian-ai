# Guardian AI — Module 4, Chapter 4.2

## Lesson quality and correction validation

Evidence observations: 165
Candidate lessons validated: 5
Template corrections generated: 5

| Category | Stage | Observations | Quality heuristic |
|---|---|---:|---:|
| workflow_scope_error | need_search | 68 | 100/100 |
| target_mismatch | need_search | 3 | 63/100 |
| workflow_scope_error | need_search | 53 | 90/100 |
| target_mismatch | need_search | 23 | 100/100 |
| schema_error | need_search | 18 | 90/100 |

## Validation properties

- Corrections are chosen from controller-owned allowlisted templates.
- Every correction is linked to validated training-only evidence.
- Source fingerprints and observations are rechecked against Module 3 live traces.
- Correction text never grants new permissions.

## Limitations

- Quality scores are deterministic heuristics, not calibrated or human-assessed quality.
- No behavioral improvement is demonstrated.
- Common attack templates appear in multiple dataset splits.
- A lesson with more supporting observations is not necessarily more effective.
- Authorization remains an independent Guardian responsibility.

## Chapter 4.3 handoff

Create a retriever that uses trusted stage, failure category and approved lesson IDs. Do not index arbitrary raw attack instructions as trusted lessons.
