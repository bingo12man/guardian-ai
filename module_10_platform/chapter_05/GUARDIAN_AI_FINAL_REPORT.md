# Guardian AI — Final Research Report

## Executive summary

Guardian AI is a local research prototype that
combines independent agent-action enforcement,
synthetic safety experiments, threat detection,
durable experiment evidence, and a read-only
research dashboard.

The system does not establish autonomous
self-improvement or real-world safety guarantees.

## Platform evidence

- Frozen baseline experiments: 2
- Candidate detector experiments: 3
- Lessons pending review: 8
- Lessons approved or active: 0
- Verified local audit chains: 4
- Paired detector examples: 48
- API write attempts rejected: 24
- Protected source files changed: 0

## Detector findings

| Dataset | v1 accuracy | v2 accuracy | Delta |
|---|---:|---:|---:|
| original | 1.0000 | 0.7500 | -25.00 pp |
| development | 0.2083 | 0.7500 | +54.17 pp |
| fresh | 0.4167 | 0.9167 | +50.00 pp |

## Release decision

**HOLD_FOR_REVIEW**

The candidate passed 3 of 9 gates.
It regressed on original known-good cases.
Independent external validation and live-agent
safety evaluation are still missing.

## Integrity and safety

The research registry and lesson journals
support deterministic verification and
conflict detection. The API does not expose
operations to alter Guardian permissions.

## Scientific limitations

- Small hand-authored synthetic evaluation suites
- Additional assessment not independently blinded
- No representative real-world generalization estimate
- No production authentication or external audit anchor
- Single-writer local research registries
- No automated browser test in Chapter 10.4
- No production security certification
- No demonstrated durable model-weight learning

## Future research

Prepare independently collected evaluation data,
evaluate false-positive regressions, run properly
controlled live-agent safety experiments,
authenticate reviewers, and add external
audit anchoring before production deployment.

## Status

Research platform complete for the specified
local prototype scope. Production readiness
and causal safety improvement are unproven.
