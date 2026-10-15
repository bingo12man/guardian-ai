# Guardian AI — Module 9.4 Findings

| Dataset | Gains | Losses | Delta pp | Exact p |
|---|---:|---:|---:|---:|
| original | 0 | 3 | -25.00 | 0.2500 |
| development | 13 | 0 | +54.17 | 0.0002 |
| fresh | 6 | 0 | +50.00 | 0.0312 |

## Release recommendation

HOLD_FOR_REVIEW

Candidate v2 regressed on original fixtures.
Improved synthetic scores do not substitute
for a large independent evaluation or live
agent-safety evidence.

## Limitations

- Small hand-authored synthetic sample
- Data-dependent heuristic development
- Conditional paired bootstrap intervals
- No multiple-testing claims
- No real-world performance inference
- No independent reviewer authentication

