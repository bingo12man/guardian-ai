
"""
Guardian AI — Module 7, Chapter 7.2

Deduplicated collective safety evidence memory.

Source:
    Module 7.1 verified observations
    Module 6.1 trusted exercise definitions

Aggregation:
    Level 1: model + session + lesson signature
    Level 2: shared lesson signature across models

Repeated observations are preserved but never
counted as independent model/session evidence.

Mixed correctness is retained as disagreement.

NO:
    raw model outputs
    new safety-rule generation
    capability issuance
    authorization changes
    automatic trust restoration

Research prototype, not a hardened security
boundary or causal evidence of learning.
"""

import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"

sys.path.insert(
    0, str(HERE.parent / "chapter_01")
)
sys.path.insert(
    0, str(
        ROOT / "module_6_curriculum/chapter_02"
    )
)

import guardian_observation_registry as registry
from guardian_adaptive_scheduler import (
    load_verified_dataset,
)

MODELS = registry.MODELS
SESSION_ID = "m6c3_live_budget8_r1"

SOURCE_DIR = HERE.parent / "chapter_01/results"

OBSERVATIONS_PATH = (
    SOURCE_DIR / "verified_observations.json"
)
AUDIT_PATH = SOURCE_DIR / "registry_audit.json"
SUMMARY_PATH = SOURCE_DIR / "m7_1_summary.json"
MANIFEST_PATH = SOURCE_DIR / "manifest.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(obj):
    return json.dumps(
        obj, sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def digest(obj):
    return hashlib.sha256(
        canonical(obj).encode("utf-8")
    ).hexdigest()


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def write_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2),
        encoding="utf-8",
    )


def load_verified_sources():
    """
    Verify stored Chapter 7.1 output against
    independently rebuilt source-record views.

    Does not authenticate original raw Ollama
    outputs that were not retained.
    """
    paths = (
        OBSERVATIONS_PATH,
        AUDIT_PATH,
        SUMMARY_PATH,
        MANIFEST_PATH,
    )
    require(
        all(path.is_file() for path in paths),
        "Missing Chapter 7.1 source",
    )

    source = registry.verify_sources()
    original = registry.source_events(source)

    expected_registry = registry.ObservationRegistry(
        original
    )
    for event in original:
        decision = expected_registry.submit(
            registry.make_claim(event)
        )
        require(
            decision["accepted"],
            "Source reconstruction failed",
        )

    stored = read_json(OBSERVATIONS_PATH)
    audit = read_json(AUDIT_PATH)
    summary = read_json(SUMMARY_PATH)
    manifest = read_json(MANIFEST_PATH)

    require(
        stored == expected_registry.accepted(),
        "Verified observation source mismatch",
    )
    require(
        audit == expected_registry.audit_log(),
        "Registry audit source mismatch",
    )
    registry.verify_audit(
        audit, expected_registry.summary()["audit_head"]
    )

    require(
        manifest["observations_sha256"]
        == digest(stored),
        "Observation checksum mismatch",
    )
    require(
        manifest["audit_sha256"] == digest(audit),
        "Audit checksum mismatch",
    )
    require(
        manifest["summary_sha256"] == digest(summary),
        "Summary checksum mismatch",
    )

    require(
        summary["module"] == 7
        and summary["chapter"] == "7.1"
        and summary["accepted"] == len(stored) == 30
        and summary["audit_entries"] == len(audit) == 30
        and summary["permission_changes"] == 0
        and summary["tool_executions"] == 0
        and summary["privilege_restorations"] == 0,
        "Registry summary invariant failed",
    )

    require(
        summary["by_model"]
        == expected_registry.summary()["by_model"],
        "Model summary does not match source",
    )

    data = load_verified_dataset()

    return {
        "observations": copy.deepcopy(stored),
        "exercises": copy.deepcopy(data["exercises"]),
        "dataset_sha256": data["dataset_sha256"],
        "registry_audit_head":
            expected_registry.summary()["audit_head"],
    }


def annotate_observations(observations, exercises):
    """
    Join each controller-scored observation
    to its trusted Chapter 6 lesson signature.

    The lesson ID never comes from an agent claim.
    """
    index = {
        exercise["exercise_id"]: exercise
        for exercise in exercises
    }

    require(
        len(index) == len(exercises),
        "Duplicate trusted exercise ID",
    )

    annotated = []
    seen = set()

    for event in observations:
        eid = event["observation_id"]

        require(eid not in seen, "Duplicate observation")
        seen.add(eid)

        require(
            event["exercise_id"] in index,
            "Unknown exercise in observation",
        )
        exercise = index[event["exercise_id"]]

        require(
            exercise["model"] == event["model"],
            "Model mismatch",
        )
        require(
            exercise["stage"] == event["workflow_stage"]
            and exercise["failure_category"]
            == event["failure_category"],
            "Lesson signature mismatch",
        )

        expected_split = (
            "practice"
            if event["phase"] == "practice"
            else "assessment"
        )

        require(
            exercise["split"] == expected_split,
            "Exercise/phase split mismatch",
        )
        require(
            event["permissions_changed"] is False
            and event["tool_executed"] is False
            and event["controller_scored"] is True,
            "Unsafe or unscored source",
        )

        unsigned = {
            k: v for k, v in event.items()
            if k != "evidence_fingerprint"
        }
        require(
            event["evidence_fingerprint"]
            == digest(unsigned),
            "Observation fingerprint mismatch",
        )

        # New fields are computed by controller code.
        annotated.append({
            "observation_id": eid,
            "model": event["model"],
            "session_id": SESSION_ID,
            "lesson_id": exercise["lesson_id"],
            "category": event["failure_category"],
            "stage": event["workflow_stage"],
            "phase": event["phase"],
            "correct": event["observed_correct"],
            "evidence_fingerprint":
                event["evidence_fingerprint"],
        })

    return sorted(
        annotated,
        key=lambda item: item["observation_id"],
    )


def evidence_unit_id(model, session, lesson):
    return "unit_" + digest({
        "model": model,
        "session_id": session,
        "lesson_id": lesson,
    })[:24]


def build_evidence_units(annotated):
    """
    Multiple outcomes for the same model/session/
    lesson form ONE correlated evidence unit.

    A unit with both correct and incorrect
    outcomes remains MIXED.
    """
    grouped = defaultdict(list)

    for item in annotated:
        key = (
            item["model"],
            item["session_id"],
            item["lesson_id"],
        )
        grouped[key].append(item)

    units = []

    for (model, session, lesson), rows in sorted(
        grouped.items()
    ):
        categories = {x["category"] for x in rows}
        stages = {x["stage"] for x in rows}

        require(
            len(categories) == 1 and len(stages) == 1,
            "Conflicting lesson metadata",
        )

        correct = sum(x["correct"] for x in rows)
        incorrect = len(rows) - correct

        if correct and incorrect:
            state = "MIXED"
        elif correct:
            state = "ALL_CORRECT"
        else:
            state = "ALL_INCORRECT"

        units.append({
            "unit_id": evidence_unit_id(
                model, session, lesson
            ),
            "model": model,
            "session_id": session,
            "lesson_id": lesson,
            "category": next(iter(categories)),
            "stage": next(iter(stages)),
            "observations": len(rows),
            "correct": correct,
            "incorrect": incorrect,
            "outcome_state": state,
            "observation_ids": sorted(
                x["observation_id"] for x in rows
            ),
            "phase_counts": dict(sorted(Counter(
                x["phase"] for x in rows
            ).items())),
            "independent_session_units": 1,
            "permissions_changed": False,
            "privileges_restored": False,
        })

    return units


def build_collective_memory(units):
    """
    Aggregate different donor model units under
    each original trusted lesson signature.

    Never merge Llama and Qwen performance into
    a single calibrated confidence score.
    """
    groups = defaultdict(list)

    for unit in units:
        require(
            unit["model"] in MODELS,
            "Unknown contributing model",
        )
        groups[unit["lesson_id"]].append(unit)

    collective = []

    for lesson_id, rows in sorted(groups.items()):
        models = sorted({
            row["model"] for row in rows
        })

        require(
            len(rows) == len({
                (row["model"], row["session_id"])
                for row in rows
            }),
            "Duplicate model/session lesson units",
        )

        require(
            len({row["category"] for row in rows}) == 1
            and len({row["stage"] for row in rows}) == 1,
            "Collective signature disagreement",
        )

        outcomes = sorted({
            row["outcome_state"] for row in rows
        })

        collective.append({
            "lesson_id": lesson_id,
            "category": rows[0]["category"],
            "stage": rows[0]["stage"],
            "contributing_models": models,
            "contributing_model_count": len(models),
            "correlated_units": len(rows),
            "observations": sum(
                row["observations"] for row in rows
            ),
            "correct": sum(
                row["correct"] for row in rows
            ),
            "incorrect": sum(
                row["incorrect"] for row in rows
            ),
            "mixed_units": sum(
                row["outcome_state"] == "MIXED"
                for row in rows
            ),
            "per_model": {
                row["model"]: {
                    "unit_id": row["unit_id"],
                    "session_id": row["session_id"],
                    "observations": row["observations"],
                    "correct": row["correct"],
                    "incorrect": row["incorrect"],
                    "outcome_state":
                        row["outcome_state"],
                }
                for row in rows
            },
            "observed_unit_states": outcomes,
            "confidence_calibrated": False,
            "behavioral_transfer_demonstrated": False,
            "safe_for_automatic_policy_update": False,
        })

    return collective


def validate_memory(annotated, units, collective):
    """
    Recompute exact canonical aggregates.
    Reject fabricated units and altered counts.
    """
    expected_units = build_evidence_units(annotated)
    expected_collective = build_collective_memory(
        expected_units
    )

    require(
        units == expected_units,
        "Evidence unit mismatch",
    )
    require(
        collective == expected_collective,
        "Collective memory mismatch",
    )

    require(
        sum(x["observations"] for x in units)
        == len(annotated),
        "Observation accounting mismatch",
    )
    require(
        sum(x["observations"] for x in collective)
        == len(annotated),
        "Collective observation accounting mismatch",
    )
    require(
        all(
            item["safe_for_automatic_policy_update"]
            is False
            for item in collective
        ),
        "Unsafe automatic-policy flag",
    )

    return True


def make_summary(annotated, units, collective):
    return {
        "module": 7,
        "chapter": "7.2",
        "verified_input_observations": len(annotated),
        "correlated_evidence_units": len(units),
        "unique_collective_lessons": len(collective),
        "units_by_model": dict(sorted(Counter(
            unit["model"] for unit in units
        ).items())),
        "unit_outcomes": dict(sorted(Counter(
            unit["outcome_state"] for unit in units
        ).items())),
        "mixed_outcome_units": sum(
            unit["outcome_state"] == "MIXED"
            for unit in units
        ),
        "cross_model_supported_lessons": sum(
            x["contributing_model_count"] > 1
            for x in collective
        ),
        "max_independent_sessions_per_model": 1,
        "calibrated_confidence_scores": 0,
        "automatic_policy_updates": 0,
        "permission_changes": 0,
        "privilege_restorations": 0,
        "new_behavioral_learning_tested": False,
    }
