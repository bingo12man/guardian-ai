# Guardian AI — Module 9 Research Findings

## Research question

Can Guardian preserve research evidence and
evaluate candidate improvements without
silently overwriting the baseline or lessons?

## Experiment registry

- Frozen baseline experiments: 2
- Candidate detector experiments: 3
- Candidate state: CANDIDATE_PENDING
- Original baseline replaced: No

## Lesson lifecycle

- Original candidates pending review: 8
- Original candidates activated: 0
- Synthetic rollback fixture: Verified

## Detector comparisons

| Dataset | v1 | v2 | Change (pp) |
|---|---:|---:|---:|
| original | 1.0000 | 0.7500 | -25.00 |
| development | 0.2083 | 0.7500 | +54.17 |
| fresh | 0.4167 | 0.9167 | +50.00 |

## Release decision

**HOLD_FOR_REVIEW**

The candidate improved on two synthetic
datasets but regressed on the original suite.
Six of nine release requirements remain unmet.

## Limitations

- No real-world generalization established
- Development examples were previously inspected
- Additional assessment is small and project-authored
- Local audit chains are not tamper-proof
- No independently authenticated reviewer
- No live agent improvement demonstrated

## Module 10 handoff

Integrate validated research summaries through
a read-only interface. Never allow dashboard
controls or model output to activate a lesson,
promote a detector or change Guardian policy.

