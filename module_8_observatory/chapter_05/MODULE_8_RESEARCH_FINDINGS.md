# Guardian AI — Module 8 Research Findings

## Research question

Can an offline observatory detect synthetic
prompt-injection threats, preserve evidence,
and generate safe lesson candidates?

## Detection results

| Metric | Original fixtures | Robustness decoys |
|---|---:|---:|
| Examples | 12 | 24 |
| Correct | 12 | 5 |
| Accuracy | 1.0000 | 0.2083 |
| Macro F1 | 1.0000 | 0.1164 |
| Attack recall | 1.0000 | 0.0625 |

Accuracy dropped 79.17 percentage points on the robustness suite.

## Candidate lessons

All candidates remain pending review.
False-positive detections can produce
false candidate alerts, so detection is
not sufficient for lesson approval.

## Scientific conclusion

Perfect performance on matching fixtures
did not transfer to new synthetic variants.
The frozen rule detector missed most new
attack phrasings and flagged benign examples.

## Limitations

Both datasets are hand-authored synthetic suites.
These results cannot establish a population
accuracy estimate or live-agent safety benefit.

## Module 9 handoff

Use versioned, immutable evidence; separate
train/evaluation splits; compare candidate
detectors against this frozen baseline;
and require independent validation before
publishing any lesson.
