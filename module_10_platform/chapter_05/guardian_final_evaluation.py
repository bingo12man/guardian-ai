
"""
Guardian AI — Module 10, Chapter 10.5

Final research platform evaluation and documentation.

Independently rechecks:
- Module 9 final scientific evidence
- Module 10 read-only research API
- Module 10.4 integration artifacts
- React production-build fingerprint
- Research evidence and lifecycle invariants

Produces deterministic reports and documentation.

No Ollama calls, external network requests,
candidate promotion, lesson activation,
or permission changes.
"""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
M10 = ROOT / "module_10_platform"
RESULTS = HERE / "results"

sys.path.insert(0, str(M10 / "chapter_04"))
sys.path.insert(0, str(M10 / "chapter_02"))

import guardian_integration_checks as integration
import guardian_research_detail_api as api

C4_RESULTS = M10 / "chapter_04/results"
C4_SUMMARY = C4_RESULTS / "m10_4_summary.json"
C4_MANIFEST = C4_RESULTS / "manifest.json"

OUTPUT_NAMES = (
    "guardian_final_report.json",
    "manifest.json",
)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def digest(value):
    return hashlib.sha256(
        canonical(value).encode("utf-8")
    ).hexdigest()


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def sha256_file(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value, indent=2, ensure_ascii=False
        ) + "\n",
        encoding="utf-8",
    )


def validate_chapter_10_4(
    summary_path=C4_SUMMARY,
    manifest_path=C4_MANIFEST,
):
    """
    Reject altered Chapter 10.4 summaries before
    trusting their recorded benchmark claims.
    """
    require(
        Path(summary_path).is_file()
        and Path(manifest_path).is_file(),
        "Missing Chapter 10.4 evidence",
    )

    summary = read_json(summary_path)
    manifest = read_json(manifest_path)

    require(
        manifest["summary_sha256"] == digest(summary),
        "Chapter 10.4 summary fingerprint mismatch",
    )

    require(
        summary["module"] == 10
        and summary["chapter"] == "10.4"
        and summary["frontend_build_verified"] is True
        and summary["experiments"] == 5
        and summary["pending_lessons"] == 8
        and summary["verified_audit_components"] == 4
        and summary["release_gates"] == 9
        and summary["detail_lookups_verified"] == 17
        and summary["paired_examples"] == 48
        and summary["write_requests_rejected"] == 24
        and summary["source_files_checked"] >= 14
        and summary["source_files_unchanged"] is True
        and summary["release_decision"]
        == "HOLD_FOR_REVIEW"
        and summary["candidate_promoted"] is False
        and summary["live_lessons_activated"] == 0
        and summary["permission_changes"] == 0
        and summary["browser_automation_completed"] is False
        and summary["production_security_certified"] is False,
        "Chapter 10.4 research invariants violated",
    )

    frontend = integration.frontend_preflight()
    require(
        summary["frontend_build_sha256"]
        == frontend["build_sha256"]
        == manifest["frontend_build_sha256"],
        "Frontend build changed after Chapter 10.4",
    )

    return {
        "summary": summary,
        "summary_sha256": sha256_file(summary_path),
        "manifest_sha256": sha256_file(manifest_path),
        "frontend": frontend,
    }


def validate_platform(
    summary_path=C4_SUMMARY,
    manifest_path=C4_MANIFEST,
):
    """
    Recompute evidence from the earlier modules,
    then execute Chapter 10.4 cross-layer checks.

    Important: these checks attest local consistency,
    not security against a privileged operator.
    """
    prior = validate_chapter_10_4(
        summary_path, manifest_path
    )

    research = api.c1.load_verified_research()
    evidence = api.m9.validate_all()

    require(
        api.m9.make_final(evidence)
        == research["final"],
        "Module 9 final evidence inconsistent",
    )

    result = integration.run_integration()
    recorded = prior["summary"]
    contract = result["contract"]

    checks = {
        "experiments": contract["experiments"],
        "pending_lessons": contract["pending_lessons"],
        "verified_audit_components":
            contract["verified_audits"],
        "release_gates": contract["release_gates"],
        "detail_lookups_verified":
            contract["detail_lookups"],
        "paired_examples": contract["paired_examples"],
        "write_requests_rejected":
            result["rejected_write_requests"],
        "source_files_checked":
            result["source_files_checked"],
    }

    require(
        all(
            recorded[name] == current
            for name, current in checks.items()
        ),
        "Final integration differs from Chapter 10.4",
    )

    require(
        result["source_files_unchanged"] is True
        and contract["release_decision"]
        == "HOLD_FOR_REVIEW",
        "Unsafe integration state",
    )

    final = research["final"]
    gates = evidence["statistics"]["gates"]

    require(
        final["frozen_baseline_experiments"] == 2
        and final["candidate_detector_experiments"] == 3
        and final["module_8_lessons_pending"] == 8
        and final["module_8_lessons_approved"] == 0
        and final["module_8_lessons_active"] == 0
        and final["candidate_promoted"] is False
        and final["baseline_replaced"] is False
        and gates["passed"] == 3
        and gates["failed"] == 6,
        "Research release boundary altered",
    )

    return {
        "prior": prior,
        "research": research,
        "evidence": evidence,
        "integration": result,
        "source_fingerprints":
            integration.evidence_fingerprints(),
    }


def make_report(validated):
    final = validated["research"]["final"]
    cross = validated["integration"]
    gates = validated["evidence"]["statistics"]["gates"]

    return {
        "project": "Guardian AI",
        "module": 10,
        "chapter": "10.5",
        "evaluation_status": "FULL_RECORD_ANALYSIS",
        "platform_status": "LOCAL_RESEARCH_PROTOTYPE",
        "completed_modules": list(range(1, 11)),
        "frontend": {
            "framework": "React and Vite",
            "build_verified": True,
            "build_sha256":
                cross["frontend"]["build_sha256"],
            "automated_browser_tested": False,
        },
        "backend": {
            "framework": "FastAPI",
            "integration": "verified_read_only",
            "live_guardian_connected": False,
        },
        "research_evidence": {
            "frozen_baseline_experiments": 2,
            "candidate_detector_experiments": 3,
            "original_lessons_pending": 8,
            "original_lessons_approved": 0,
            "original_lessons_active": 0,
            "audit_components_verified": 4,
            "paired_examples": 48,
        },
        "detector_comparisons":
            final["detector_comparisons"],
        "release_gates": {
            "decision": gates["release_decision"],
            "passed": gates["passed"],
            "failed": gates["failed"],
            "candidate_promoted": False,
            "baseline_replaced": False,
        },
        "platform_verification": {
            "individual_detail_lookups":
                cross["contract"]["detail_lookups"],
            "rejected_write_requests":
                cross["rejected_write_requests"],
            "source_files_checked":
                cross["source_files_checked"],
            "source_files_unchanged":
                cross["source_files_unchanged"],
            "permission_changes": 0,
        },
        "provenance": {
            "module_9_final_sha256":
                api.c1.digest(final),
            "experiment_audit_heads":
                final["experiment_audit_heads"],
            "lesson_audit_heads":
                final["lesson_audit_heads"],
            "protected_sources_sha256":
                digest(validated["source_fingerprints"]),
            "chapter_10_4_summary_sha256":
                validated["prior"]["summary_sha256"],
        },
        "scientific_findings": [
            "Frozen v1 detector fits its original synthetic fixtures",
            "Frozen v1 performed poorly on changed synthetic wording",
            "Candidate v2 improved on later synthetic evaluations",
            "Candidate v2 regressed on original known-good cases",
            "Release gates correctly retained HOLD_FOR_REVIEW",
            "No causal live-agent safety improvement established",
        ],
        "research_limitations": [
            "Small hand-authored synthetic evaluation suites",
            "Additional assessment not independently blinded",
            "No representative real-world generalization estimate",
            "No production authentication or external audit anchor",
            "Single-writer local research registries",
            "No automated browser test in Chapter 10.4",
            "No production security certification",
            "No demonstrated durable model-weight learning",
        ],
        "automatic_lesson_publication": False,
        "candidate_promoted": False,
        "live_agent_permission_changes": 0,
        "network_collection_performed": False,
        "model_inference_performed_in_finalization": False,
    }


def make_architecture():
    return """# Guardian AI — Research Architecture

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
"""


def make_setup():
    return """# Guardian AI — Local Setup

## Requirements

- Python 3 with the repository's virtual environment
- Node.js and npm
- Python dependencies: fastapi, uvicorn, httpx, pytest
- Frontend dependencies from chapter_03/package.json

Ollama was used in previous research modules. It is
not required to view the completed Module 10 dashboard.

## 1. Start the backend

From the repository root:

```bash
source venv/bin/activate
cd module_10_platform/chapter_02
python3 -m uvicorn guardian_research_detail_api:app \\
  --host 127.0.0.1 --port 8000
```

Backend health: http://127.0.0.1:8000/health
API documentation: http://127.0.0.1:8000/docs

## 2. Start the frontend

In a second terminal:

```bash
cd ~/Downloads/guardian-ai/module_10_platform/chapter_03
npm install
npm run dev
```

Open: http://127.0.0.1:5173

The Vite development server proxies /api and /health
to local FastAPI on port 8000.

## 3. Build and test the frontend

```bash
cd ~/Downloads/guardian-ai/module_10_platform/chapter_03
npm test
npm run build
```

## 4. Test the APIs

```bash
cd ~/Downloads/guardian-ai/module_10_platform/chapter_02
python3 -m pytest test_guardian_research_detail_api.py -v
python3 -u c2_detail_api_benchmark.py
```

## 5. Run the final integration checks

```bash
cd ~/Downloads/guardian-ai/module_10_platform/chapter_04
python3 -m pytest test_guardian_integration.py -v
python3 -u c4_integration_benchmark.py
```

## 6. Final scientific evaluation

```bash
cd ~/Downloads/guardian-ai/module_10_platform/chapter_05
python3 -m pytest test_guardian_final_evaluation.py -v
python3 -u guardian_final_evaluation.py
```

## Troubleshooting

ECONNREFUSED 127.0.0.1:8000:
Start the Chapter 10.2 FastAPI backend first.

HTTP 503 EVIDENCE_VERIFICATION_FAILED:
Inspect the upstream research evidence and manifests.
Do not bypass verification by inventing results.

Frontend does not start:
Check that Node.js is installed and run npm install
inside chapter_03.

## Usage restrictions

- Local research prototype only
- No public-network API exposure
- No automatic candidate promotion
- No lesson activation through the dashboard
- No Guardian permission changes through the UI
"""


def make_findings(report):
    rows = report["detector_comparisons"]
    lines = [
        "# Guardian AI — Final Research Report",
        "",
        "## Executive summary",
        "",
        "Guardian AI is a local research prototype that",
        "combines independent agent-action enforcement,",
        "synthetic safety experiments, threat detection,",
        "durable experiment evidence, and a read-only",
        "research dashboard.",
        "",
        "The system does not establish autonomous",
        "self-improvement or real-world safety guarantees.",
        "",
        "## Platform evidence",
        "",
        "- Frozen baseline experiments: 2",
        "- Candidate detector experiments: 3",
        "- Lessons pending review: 8",
        "- Lessons approved or active: 0",
        "- Verified local audit chains: 4",
        "- Paired detector examples: 48",
        "- API write attempts rejected: 24",
        "- Protected source files changed: 0",
        "",
        "## Detector findings",
        "",
        "| Dataset | v1 accuracy | v2 accuracy | Delta |",
        "|---|---:|---:|---:|",
    ]

    for name in ("original", "development", "fresh"):
        item = rows[name]
        lines.append(
            f"| {name} | {item['v1_accuracy']:.4f}"
            f" | {item['v2_accuracy']:.4f}"
            f" | {item['accuracy_delta_pp']:+.2f} pp |"
        )

    lines.extend([
        "",
        "## Release decision",
        "",
        "**HOLD_FOR_REVIEW**",
        "",
        "The candidate passed 3 of 9 gates.",
        "It regressed on original known-good cases.",
        "Independent external validation and live-agent",
        "safety evaluation are still missing.",
        "",
        "## Integrity and safety",
        "",
        "The research registry and lesson journals",
        "support deterministic verification and",
        "conflict detection. The API does not expose",
        "operations to alter Guardian permissions.",
        "",
        "## Scientific limitations",
        "",
    ])

    for limitation in report["research_limitations"]:
        lines.append(f"- {limitation}")

    lines.extend([
        "",
        "## Future research",
        "",
        "Prepare independently collected evaluation data,",
        "evaluate false-positive regressions, run properly",
        "controlled live-agent safety experiments,",
        "authenticate reviewers, and add external",
        "audit anchoring before production deployment.",
        "",
        "## Status",
        "",
        "Research platform complete for the specified",
        "local prototype scope. Production readiness",
        "and causal safety improvement are unproven.",
    ])

    return "\n".join(lines) + "\n"


def run_final(
    output_dir=RESULTS,
    document_dir=HERE,
    summary_path=C4_SUMMARY,
    manifest_path=C4_MANIFEST,
):
    validated = validate_platform(
        summary_path, manifest_path
    )
    report = make_report(validated)

    require(
        report["release_gates"]["decision"]
        == "HOLD_FOR_REVIEW"
        and report["research_evidence"][
            "original_lessons_active"
        ] == 0
        and report["platform_verification"][
            "source_files_unchanged"
        ] is True
        and report["live_agent_permission_changes"] == 0,
        "Final release safety invariant failed",
    )

    output_dir = Path(output_dir)
    document_dir = Path(document_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    document_dir.mkdir(parents=True, exist_ok=True)

    report_path = (
        output_dir / "guardian_final_report.json"
    )
    architecture_path = (
        document_dir / "GUARDIAN_AI_ARCHITECTURE.md"
    )
    setup_path = (
        document_dir / "GUARDIAN_AI_SETUP.md"
    )
    findings_path = (
        document_dir / "GUARDIAN_AI_FINAL_REPORT.md"
    )

    write_json(report_path, report)

    architecture_path.write_text(
        make_architecture(), encoding="utf-8"
    )
    setup_path.write_text(
        make_setup(), encoding="utf-8"
    )
    findings_path.write_text(
        make_findings(report), encoding="utf-8"
    )

    manifest = {
        "module": 10,
        "chapter": "10.5",
        "report_sha256": sha256_file(report_path),
        "architecture_sha256":
            sha256_file(architecture_path),
        "setup_sha256": sha256_file(setup_path),
        "findings_sha256": sha256_file(findings_path),
        "chapter_10_4_summary_sha256":
            sha256_file(summary_path),
        "chapter_10_4_manifest_sha256":
            sha256_file(manifest_path),
        "module_9_final_sha256":
            report["provenance"][
                "module_9_final_sha256"
            ],
        "protected_sources_sha256":
            report["provenance"][
                "protected_sources_sha256"
            ],
    }

    manifest_path_out = (
        output_dir / "manifest.json"
    )
    write_json(manifest_path_out, manifest)

    summary = {
        "module": 10,
        "chapter": "10.5",
        "evaluation_status":
            report["evaluation_status"],
        "platform_status":
            report["platform_status"],
        "research_modules_in_scope":
            len(report["completed_modules"]),
        "frontend_build_verified":
            report["frontend"]["build_verified"],
        "experiments": 5,
        "pending_lessons": 8,
        "verified_audit_components": 4,
        "paired_examples": 48,
        "write_requests_rejected": 24,
        "protected_sources_unchanged": True,
        "release_decision": "HOLD_FOR_REVIEW",
        "release_gates_passed": 3,
        "release_gates_failed": 6,
        "candidate_promoted": False,
        "permissions_changed": 0,
        "final_manifest_verified":
            manifest["report_sha256"]
            == sha256_file(report_path),
        "production_ready": False,
    }

    print("\nFINAL MODULE 10 C5 SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nReport:", report_path)
    print("Architecture:", architecture_path)
    print("Setup:", setup_path)
    print("Findings:", findings_path)
    print("Manifest:", manifest_path_out)

    return {
        "summary": summary,
        "report": report,
        "manifest": manifest,
    }


if __name__ == "__main__":
    run_final()
