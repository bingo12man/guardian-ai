
"""
Guardian AI — Module 10, Chapter 10.2

Extended read-only research API.

Uses the verified Chapter 10.1 FastAPI app.

Adds:
  GET /api/v1/experiments
  GET /api/v1/experiments/{experiment_id}
  GET /api/v1/lesson-review
  GET /api/v1/lesson-review/{candidate_id}
  GET /api/v1/audits
  GET /api/v1/audits/{component}
  GET /api/v1/paired-outcomes/{dataset}
  GET /api/v1/release-gates

No POST/PUT/PATCH/DELETE routes are created.
No raw prompt-injection content is returned.
No publication, capability, or policy operations.
"""

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

sys.path.insert(
    0,
    str(ROOT / "module_10_platform/chapter_01"),
)

import guardian_research_api as c1

m9 = c1.m9
app = c1.app

router = APIRouter(
    prefix="/api/v1",
    tags=["Research Details"],
)

DATASETS = ("original", "development", "fresh")

AUDIT_COMPONENTS = (
    "baseline_experiments",
    "candidate_experiments",
    "pending_lessons",
    "fixture_lifecycle",
)


def verified_snapshot():
    """
    Perform Chapter 10.1 final evidence validation,
    then reconstruct lower-level records.

    Verification failures are handled by the
    existing fail-closed ValueError handler.
    """
    research = c1.load_verified_research()
    evidence = m9.validate_all()

    c1.require(
        m9.make_final(evidence) == research["final"],
        "Research snapshot inconsistency",
    )

    return research, evidence


def experiment_view(record):
    """
    Minimize a verified experiment record.

    Include full recorded evaluation metrics,
    but exclude any untrusted source documents.
    """
    return {
        "experiment_id": record["experiment_id"],
        "kind": record["kind"],
        "status": record["status"],
        "detector_version": record["detector_version"],
        "detector_sha256": record["detector_sha256"],
        "dataset": record["configuration"]["dataset"],
        "dataset_sha256": record["dataset_sha256"],
        "source_module": record["source_module"],
        "source_evidence_sha256":
            record["source_evidence_sha256"],
        "configuration": {
            "evaluation_mode":
                record["configuration"]["evaluation_mode"],
            "model_inference":
                record["configuration"]["model_inference"],
        },
        "result": record["result"],
        "editable": False,
        "promoted": False,
    }


def all_experiments(evidence):
    records = (
        evidence["baseline"]["records"]
        + evidence["candidate"]["records"]
    )

    c1.require(
        len(records) == 5
        and len({
            x["experiment_id"] for x in records
        }) == 5,
        "Invalid experiment collection",
    )

    return sorted(
        records,
        key=lambda row: row["experiment_id"],
    )


def lesson_views(evidence):
    """
    Lesson records come from the revalidated
    controller-authored Module 8 candidate source.

    No raw document content or lesson instructions
    are exposed by this API.
    """
    candidates = m9.c2.trusted_candidates()
    journal = evidence["lessons"]["pending"]

    c1.require(
        len(candidates) == 8
        and journal["pending_review"] == 8
        and journal["active"] == 0,
        "Unexpected pending lesson state",
    )

    output = []

    for candidate in candidates:
        lesson = m9.c2.safe_lesson(candidate)

        output.append({
            "candidate_id": candidate["candidate_id"],
            "template_id": candidate["template_id"],
            "category": candidate["category"],
            "snapshot_id": candidate["snapshot_id"],
            "content_sha256":
                candidate["content_sha256"],
            "lesson_sha256": m9.c2.digest(lesson),
            "review_state": "PENDING_REVIEW",
            "approved": False,
            "active": False,
            "published": False,
            "can_approve_via_api": False,
        })

    return sorted(
        output,
        key=lambda row: row["candidate_id"],
    )


def audit_views(evidence):
    """
    Expose only verified audit identities,
    counts and provenance fingerprints.

    Hash chain correctness is checked upstream;
    the API does not expose mutable journal files.
    """
    baseline = evidence["baseline"]
    candidate = evidence["candidate"]
    lessons = evidence["lessons"]

    rows = [
        {
            "component": "baseline_experiments",
            "source_module": 9,
            "record_count": len(baseline["records"]),
            "audit_events": len(baseline["records"]),
            "audit_head": baseline["head"],
            "verification": "VERIFIED",
        },
        {
            "component": "candidate_experiments",
            "source_module": 9,
            "record_count": len(candidate["records"]),
            "audit_events": len(candidate["records"]),
            "audit_head": candidate["head"],
            "verification": "VERIFIED",
        },
        {
            "component": "pending_lessons",
            "source_module": 9,
            "record_count":
                lessons["pending"]["lessons"],
            "audit_events":
                lessons["pending"]["audit_events"],
            "audit_head": lessons["pending_head"],
            "verification": "VERIFIED",
        },
        {
            "component": "fixture_lifecycle",
            "source_module": 9,
            "record_count":
                lessons["fixture"]["lessons"],
            "audit_events":
                lessons["fixture"]["audit_events"],
            "audit_head": lessons["fixture_head"],
            "verification": "VERIFIED",
        },
    ]

    c1.require(
        tuple(row["component"] for row in rows)
        == AUDIT_COMPONENTS,
        "Audit component mismatch",
    )

    return rows


def paired_dataset_view(evidence, dataset):
    if dataset not in DATASETS:
        raise HTTPException(
            status_code=404,
            detail="Unknown evaluation dataset",
        )

    paired = evidence["statistics"]["paired"][dataset]

    return {
        "dataset": dataset,
        "examples": paired["examples"],
        "attack_examples": paired["attack_examples"],
        "benign_examples": paired["benign_examples"],
        "gains": paired["completion_gains"],
        "losses": paired["completion_losses"],
        "both_correct": paired["both_correct"],
        "both_incorrect": paired["both_incorrect"],
        "accuracy_delta_pp":
            paired["accuracy_delta_pp"],
        "exact_p_two_sided":
            paired["exact_paired_p_two_sided"],
        "bootstrap_delta_pp_95":
            paired["paired_bootstrap_delta_pp_95"],
        "v1": paired["v1"],
        "v2": paired["v2"],
        "outcomes": [
            {
                "snapshot_id": row["snapshot_id"],
                "true_category": row["true_category"],
                "v1_predicted": row["v1_predicted"],
                "v2_predicted": row["v2_predicted"],
                "v1_correct": row["v1_correct"],
                "v2_correct": row["v2_correct"],
                "outcome": row["outcome"],
            }
            for row in paired["per_example"]
        ],
        "contains_raw_content": False,
        "independent_real_world_test": False,
    }


@router.get("/experiments")
def list_experiments(
    kind: str | None = None,
):
    _, evidence = verified_snapshot()

    if kind is not None and kind not in (
        "frozen_detector_baseline",
        "candidate_detector",
    ):
        raise HTTPException(
            status_code=422,
            detail="Unsupported experiment kind",
        )

    records = all_experiments(evidence)

    if kind is not None:
        records = [
            record for record in records
            if record["kind"] == kind
        ]

    return {
        "total": len(records),
        "experiments": [
            experiment_view(record)
            for record in records
        ],
        "read_only": True,
    }


@router.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str):
    _, evidence = verified_snapshot()

    records = {
        record["experiment_id"]: record
        for record in all_experiments(evidence)
    }

    if experiment_id not in records:
        raise HTTPException(
            status_code=404,
            detail="Experiment not found",
        )

    return experiment_view(records[experiment_id])


@router.get("/lesson-review")
def list_lesson_review():
    _, evidence = verified_snapshot()
    rows = lesson_views(evidence)

    return {
        "total": len(rows),
        "pending": len(rows),
        "approved": 0,
        "active": 0,
        "candidates": rows,
        "approval_endpoint_available": False,
    }


@router.get("/lesson-review/{candidate_id}")
def get_lesson_review(candidate_id: str):
    _, evidence = verified_snapshot()

    rows = {
        row["candidate_id"]: row
        for row in lesson_views(evidence)
    }

    if candidate_id not in rows:
        raise HTTPException(
            status_code=404,
            detail="Candidate not found",
        )

    return rows[candidate_id]


@router.get("/audits")
def list_audits():
    _, evidence = verified_snapshot()
    rows = audit_views(evidence)

    return {
        "components": len(rows),
        "audits": rows,
        "all_verified": True,
        "externally_anchored": False,
    }


@router.get("/audits/{component}")
def get_audit(component: str):
    _, evidence = verified_snapshot()

    rows = {
        row["component"]: row
        for row in audit_views(evidence)
    }

    if component not in rows:
        raise HTTPException(
            status_code=404,
            detail="Unknown audit component",
        )

    return rows[component]


@router.get("/paired-outcomes/{dataset}")
def paired_outcomes(dataset: str):
    _, evidence = verified_snapshot()
    return paired_dataset_view(evidence, dataset)


@router.get("/release-gates")
def release_gates():
    research, evidence = verified_snapshot()
    gates = evidence["statistics"]["gates"]

    return {
        "decision": gates["release_decision"],
        "passed": gates["passed"],
        "failed": gates["failed"],
        "gates": [
            {
                "gate": name,
                "passed": passed,
            }
            for name, passed in gates["gates"].items()
        ],
        "candidate_status":
            research["final"]["candidate_detector_status"],
        "candidate_promoted": False,
        "baseline_replaced": False,
        "activation_available": False,
        "research_only": True,
    }


# Extend the existing verified app.
app.include_router(router)
