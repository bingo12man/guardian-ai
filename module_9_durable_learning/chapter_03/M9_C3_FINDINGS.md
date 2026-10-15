# Guardian AI — Module 9.3 Findings

## Detector comparison

| Dataset | v1 accuracy | v2 accuracy | Delta (pp) |
|---|---:|---:|---:|
| original | 1.0000 | 0.7500 | -25.00 |
| development | 0.2083 | 0.7500 | +54.17 |
| fresh | 0.4167 | 0.9167 | +50.00 |

## Interpretation

The previously inspected 24-case robustness
dataset is development evidence, not a blind test.
The separate 12-case synthetic assessment
provides a preliminary additional comparison.

## Limitations

- Heuristic detector, not learned model
- All examples are synthetic
- The new suite is still hand-authored
- No real-world prevalence estimate
- No live agent behavioral evaluation
- Future detector revisions need a new
  independently prepared test suite
- Candidate remains unpromoted

