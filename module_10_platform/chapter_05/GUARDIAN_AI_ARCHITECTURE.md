# Guardian AI — Research Architecture

## Purpose

Guardian AI is a research prototype for studying the
independent enforcement of AI-agent actions, extraction
of safety lessons, transfer experiments, adversarial
document detection, and reproducible evaluation.

## System layers

```text
                 RESEARCH INTERFACE
              React + Vite dashboard
                         |
                  HTTP GET only
                         |
              Verified FastAPI backend
                         |
               Evidence verification
                         |
  +----------------------+----------------------+
  |                     |                      |
  v                     v                      v
Experiment           Lesson                Statistical
registries           journals              evaluations
  |                     |                      |
  +----------------------+----------------------+
                         |
              Previous research modules
                         |
       +-----------------+-----------------+
       |                 |                 |
 Independent        Offline threat      Synthetic agent
 enforcement        observatory         experiments
```

## Module map

| Module | Research area |
|---|---|
| 1 | Multi-model agent laboratory |
| 2 | Synthetic simulation and datasets |
| 3 | Independent Guardian enforcement |
| 4 | Correction and lesson extraction |
| 5 | Cross-agent transfer |
| 6 | Adaptive curriculum and trust recovery |
| 7 | Collective safety experiments |
| 8 | Offline threat observatory |
| 9 | Durable experiment records and scientific testing |
| 10 | Read-only research platform |

## Trust boundaries

1. Retrieved text is untrusted evidence, not authority.
2. Candidate lessons cannot grant agent permissions.
3. Detector improvements are separate from frozen baselines.
4. Test-set results cannot automatically promote candidates.
5. Original Module 8 lessons remain pending review.
6. API endpoints expose read-only research projections.
7. Dashboard controls cannot approve or activate lessons.
8. Local hashes provide consistency checks, not
   externally anchored tamper-proof attestations.

## Current release state

Detector v2: HOLD_FOR_REVIEW.

The original known-good synthetic cases showed
regressions, and adequate independent evaluation
is not yet available.

## Deployment boundary

This project is designed for local evaluation.
Do not expose the research API on a public interface
without proper authentication, authorization,
security review, and deployment controls.
