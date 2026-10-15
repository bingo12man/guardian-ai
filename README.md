
# Guardian AI

**A research prototype for independent AI-agent safety enforcement, collective safety experiments, prompt-injection detection, and reproducible evaluation.**

Guardian AI explores whether AI agents can operate under an independent policy enforcement layer, learn from controller-approved corrections, and share safety evidence without automatically expanding their own permissions.

The project includes a read-only FastAPI backend and React dashboard for exploring verified research results.

## System Architecture

```text
Agent Proposals
      |
      v
Independent Guardian Enforcement
      |
      +---- Authorized actions
      |
      +---- Denied actions
                |
                v
       Correction / Lesson Extraction
                |
                v
       Transfer and Curriculum Experiments
                |
                v
       Collective Safety Research

Offline Threat Observatory
      |
      v
Threat Detection / Pending Lesson Candidates
      |
      v
Versioned Experiment Registry
      |
      v
Statistical Regression Gates
      |
      v
Verified FastAPI Research API
      |
      v
React Research Dashboard
```

## Research Modules

| Module | Topic |
|---|---|
| 1 | Multi-model agent laboratory |
| 2 | Synthetic simulation and attack datasets |
| 3 | Independent Guardian enforcement |
| 4 | Correction and lesson extraction |
| 5 | Cross-agent lesson transfer |
| 6 | Adaptive safety curriculum |
| 7 | Collective learning experiments |
| 8 | Offline threat observatory |
| 9 | Durable evidence and detector benchmarking |
| 10 | Unified research platform |

## Key Results

### Frozen Detector vs Candidate Detector

| Dataset | v1 Accuracy | v2 Accuracy |
|---|---:|---:|
| Original fixtures (12) | 100% | 75% |
| Development decoys (24) | 20.83% | 75% |
| Additional synthetic assessment (12) | 41.67% | 91.67% |

Candidate v2 improved substantially on the later synthetic datasets but regressed on original examples.

**Release decision: HOLD_FOR_REVIEW.**

This project does not claim that these results demonstrate real-world generalization or a causal improvement in live-agent safety.

### Research Integrity

- 2 frozen baseline experiments
- 3 candidate detector experiments
- 8 pending lesson candidates
- 4 locally verified audit chains
- 48 paired evaluation examples
- 24 rejected integration-test write requests
- 0 automatic detector promotions
- 0 live lesson activations
- 0 Guardian permission changes

## Tech Stack

**Backend:** Python, FastAPI, pytest

**Frontend:** React, Vite, JavaScript, CSS

**Research:** Local Ollama experiments, synthetic datasets, rule-based detection, paired statistical evaluation, versioned evidence records

**Integrity:** SHA-256 fingerprints, local hash-linked audit records, idempotency and fail-closed evidence validation

## Run Locally

### Backend

```bash
source venv/bin/activate
cd module_10_platform/chapter_02

python3 -m uvicorn guardian_research_detail_api:app \
  --host 127.0.0.1 --port 8000
```

### Frontend

```bash
cd module_10_platform/chapter_03

npm install
npm run dev
```

Open http://127.0.0.1:5173

## Documentation

- [Final Research Report](module_10_platform/chapter_05/GUARDIAN_AI_FINAL_REPORT.md)
- [Architecture](module_10_platform/chapter_05/GUARDIAN_AI_ARCHITECTURE.md)
- [Setup Guide](module_10_platform/chapter_05/GUARDIAN_AI_SETUP.md)
- [Module 9 Research Findings](module_9_durable_learning/chapter_05/MODULE_9_RESEARCH_FINDINGS.md)

## Limitations

The evaluation suites are primarily small, hand-authored synthetic datasets. The detector has not been validated on representative real-world attacks.

The local experiment registry is not an externally anchored, tamper-proof service. The platform does not provide production authentication, independently authenticated review, or demonstrated durable model-weight learning.

Guardian AI is a local research prototype, not a production-certified agent security system.

## Current Status

**Research platform implementation complete.**

Candidate detector v2 remains pending review. All original safety lessons remain pending, and no automated permission changes are permitted.
