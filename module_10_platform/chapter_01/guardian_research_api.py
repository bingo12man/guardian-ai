
"""
Guardian AI — Module 10, Chapter 10.1

Verified, read-only FastAPI research backend.

Endpoints:
    GET /health
    GET /api/v1/overview
    GET /api/v1/detectors
    GET /api/v1/lessons
    GET /api/v1/release

Properties:
- Rebuilds Module 9 evidence before serving reports
- Checks Module 10 contract and final manifest
- Exposes minimized, allowlisted response fields
- No arbitrary filesystem access endpoint
- No lesson approval, promotion or capability API
- No live Guardian tool execution
- No outbound network calls

Research prototype, not a public production service.
"""

import hashlib
import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

M9 = ROOT / "module_9_durable_learning/chapter_05"
sys.path.insert(0, str(M9))

import m9_5_final_benchmark as m9


FINAL_PATH = M9 / "results/m9_5_final_results.json"
CONTRACT_PATH = M9 / "MODULE_10_INTERFACE.json"
MANIFEST_PATH = M9 / "results/manifest.json"

APP_VERSION = "0.1.0"

app = FastAPI(
    title="Guardian AI Research API",
    version=APP_VERSION,
    description=(
        "Verified read-only research summaries. "
        "No authorization or lesson-management operations."
    ),
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def digest(value):
    return hashlib.sha256(
        canonical(value).encode("utf-8")
    ).hexdigest()


def file_hash(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def load_verified_research():
    """
    Fail closed when scientific evidence changes.

    Revalidation is performed per request to avoid
    silently serving cached evidence after tampering.

    This has overhead; Chapter 10.4 may add a
    safely invalidated verified snapshot cache.
    """
    evidence = m9.validate_all()
    expected_final = m9.make_final(evidence)
    expected_contract = m9.make_module_10_contract(
        expected_final
    )

    require(
        FINAL_PATH.is_file()
        and CONTRACT_PATH.is_file()
        and MANIFEST_PATH.is_file(),
        "Missing final research artifacts",
    )

    actual_final = read_json(FINAL_PATH)
    contract = read_json(CONTRACT_PATH)
    manifest = read_json(MANIFEST_PATH)

    require(
        actual_final == expected_final,
        "Module 9 final report mismatch",
    )
    require(
        contract == expected_contract,
        "Module 10 interface mismatch",
    )

    require(
        contract["module_9_final_sha256"]
        == digest(actual_final),
        "Invalid final evidence fingerprint",
    )

    require(
        manifest["final_sha256"]
        == file_hash(FINAL_PATH)
        and manifest["contract_sha256"]
        == file_hash(CONTRACT_PATH),
        "Final manifest mismatch",
    )

    # Re-check the source files captured by the
    # Chapter 9.5 final manifest.
    for name, path in m9.SOURCES.items():
        require(
            manifest["source_file_sha256"][name]
            == file_hash(path),
            f"Source fingerprint mismatch: {name}",
        )

    require(
        contract["producer_module"] == 9
        and contract["consumer_module"] == 10
        and contract["integration_mode"]
        == "read_only_research",
        "Unsupported integration contract",
    )

    require(
        all(
            value is False
            for value in contract[
                "authorization_constraints"
            ].values()
        ),
        "Authorization boundary changed",
    )

    require(
        actual_final["release_decision"]
        == "HOLD_FOR_REVIEW"
        and actual_final["candidate_promoted"] is False
        and actual_final["baseline_replaced"] is False
        and actual_final["module_8_lessons_active"] == 0
        and actual_final["authorization_changes"] == 0,
        "Research state is not safe to expose",
    )

    return {
        "final": actual_final,
        "contract": contract,
        "manifest": manifest,
    }


def overview_view(research):
    final = research["final"]

    return {
        "project": "Guardian AI",
        "platform_version": APP_VERSION,
        "completed_research_modules": 9,
        "current_module": 10,
        "evaluation_status":
            final["evaluation_status"],
        "frozen_baseline_experiments":
            final["frozen_baseline_experiments"],
        "candidate_detector_experiments":
            final["candidate_detector_experiments"],
        "pending_lessons":
            final["module_8_lessons_pending"],
        "active_lessons":
            final["module_8_lessons_active"],
        "release_decision":
            final["release_decision"],
        "permission_changes":
            final["authorization_changes"],
        "research_only": True,
        "live_guardian_connected": False,
    }


def detectors_view(research):
    final = research["final"]

    comparisons = []

    for dataset in (
        "original",
        "development",
        "fresh",
    ):
        row = final["detector_comparisons"][dataset]

        comparisons.append({
            "dataset": dataset,
            "examples": row["examples"],
            "v1_accuracy": row["v1_accuracy"],
            "v2_accuracy": row["v2_accuracy"],
            "delta_accuracy_pp":
                row["accuracy_delta_pp"],
            "paired_gains": row["gains"],
            "paired_losses": row["losses"],
            "exact_p_two_sided":
                row["exact_p_two_sided"],
            "bootstrap_delta_pp_95":
                row["bootstrap_delta_pp_95"],
        })

    return {
        "baseline_detector":
            "m8_rule_detector_v1",
        "candidate_detector":
            "guardian_detector_v2_candidate_1",
        "candidate_status":
            "CANDIDATE_PENDING",
        "candidate_promoted": False,
        "comparison_results": comparisons,
        "real_world_generalization_established": False,
    }


def lessons_view(research):
    final = research["final"]

    return {
        "source": "Module 8 controller-authored candidates",
        "pending": final["module_8_lessons_pending"],
        "approved": final["module_8_lessons_approved"],
        "active": final["module_8_lessons_active"],
        "synthetic_fixture_rolled_back":
            final["synthetic_fixture_rolled_back"],
        "live_guardian_lessons_activated":
            final["live_guardian_lessons_activated"],
        "review_controls_exposed": False,
        "lesson_content_exposed": False,
    }


def release_view(research):
    final = research["final"]

    return {
        "candidate_status":
            final["candidate_detector_status"],
        "decision": final["release_decision"],
        "gates_passed":
            final["release_gates_passed"],
        "gates_failed":
            final["release_gates_failed"],
        "baseline_replaced": False,
        "candidate_promoted": False,
        "automatic_promotion_allowed": False,
        "independent_deployment_evidence": False,
        "real_world_generalization_established":
            final["real_world_generalization_established"],
    }


@app.exception_handler(ValueError)
async def evidence_error_handler(
    request: Request,
    exc: ValueError,
):
    """
    Don't expose file paths or internal audit
    structures in API error responses.
    """
    return JSONResponse(
        status_code=503,
        content={
            "error": "EVIDENCE_VERIFICATION_FAILED",
            "message": (
                "Verified research evidence is unavailable."
            ),
        },
    )


@app.get("/health")
def health():
    try:
        load_verified_research()
        return {
            "status": "ok",
            "evidence_verified": True,
            "read_only": True,
        }
    except (ValueError, OSError, KeyError, TypeError):
        raise HTTPException(
            status_code=503,
            detail="Research evidence unavailable",
        )


@app.get("/api/v1/overview")
def overview():
    return overview_view(load_verified_research())


@app.get("/api/v1/detectors")
def detectors():
    return detectors_view(load_verified_research())


@app.get("/api/v1/lessons")
def lessons():
    return lessons_view(load_verified_research())


@app.get("/api/v1/release")
def release():
    return release_view(load_verified_research())
