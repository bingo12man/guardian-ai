
"""
Guardian AI — Module 7, Chapter 7.1

Controller-verified multi-agent observation registry.

Source:
  Module 6.3 recorded live practice/assessment
  Module 6.5 research interface

Accept only structured claims matching the exact
controller-owned event record.

No raw prompts or model outputs are shared.
No tool execution or permission modification.

IMPORTANT:
  Source matching verifies record consistency,
  not independent authenticity of the original
  model generation or its saved score.
"""

import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"

M6 = ROOT / "module_6_curriculum"

for chapter in (
    "chapter_01",
    "chapter_02",
    "chapter_03",
    "chapter_04",
):
    sys.path.insert(0, str(M6 / chapter))

import m6_1_curriculum_dataset as curriculum
import guardian_trust_recovery as trust
from guardian_adaptive_scheduler import (
    load_verified_dataset,
)

MODELS = tuple(curriculum.MODELS)

LIVE_PATH = (
    M6 / "chapter_03/results"
    / "c3_results_live_ollama_practice_budget8.json"
)

FINAL_PATH = (
    M6 / "chapter_05/results"
    / "m6_5_final_results.json"
)

CONTRACT_PATH = (
    M6 / "chapter_05/MODULE_7_INTERFACE.json"
)

PHASES = (
    "initial_assessment",
    "practice",
    "final_assessment",
)

GENESIS = hashlib.sha256(
    b"GUARDIAN_M7_C1_REGISTRY_V1"
).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(obj):
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def digest(obj):
    return hashlib.sha256(
        canonical(obj).encode("utf-8")
    ).hexdigest()


def file_hash(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
    )


def verify_sources():
    """
    Revalidate the stored curriculum and scores.

    This is NOT cryptographic authentication of
    the original model's raw responses.
    """
    for path in (LIVE_PATH, FINAL_PATH, CONTRACT_PATH):
        require(path.is_file(), f"Missing source: {path}")

    data = load_verified_dataset()
    live = read_json(LIVE_PATH)
    final = read_json(FINAL_PATH)
    contract = read_json(CONTRACT_PATH)

    trust.validate_live_source(live, data)

    require(
        final["module"] == 6
        and final["chapter"] == "6.5"
        and final["evaluation_status"]
        == "FULL_RECORD_ANALYSIS",
        "Invalid Module 6 final result",
    )
    require(
        contract["producer_module"] == 6
        and contract["consumer_module"] == 7,
        "Invalid Module 7 interface",
    )
    require(
        contract["module_6_final_sha256"]
        == digest(final),
        "Module 7 contract/final mismatch",
    )

    policy = contract["safety_constraints"]

    require(
        policy["agents_can_change_permissions"] is False
        and policy["agents_can_issue_capabilities"] is False
        and policy["lessons_grant_privileges"] is False
        and policy["reviews_grant_privileges"] is False
        and policy["trusted_approval_required"] is True,
        "Authorization constraints changed",
    )

    require(
        set(contract["supported_models"]) == set(MODELS)
        and set(contract["failure_categories"])
        == set(curriculum.CATEGORIES),
        "Unexpected contract vocabulary",
    )

    for model in MODELS:
        row = live["records"][model]
        expected = final["by_model"][model]

        require(
            expected["initial_correct"]
            == row["initial_assessment"]["correct"]
            and expected["practice_correct"]
            == row["practice"]["correct"]
            and expected["final_correct"]
            == row["final_assessment"]["correct"]
            and expected["trust_state"]
            == "PRACTICE_REQUIRED",
            "Final results/source inconsistency",
        )

    return {
        "data": data,
        "live": live,
        "final": final,
        "contract": contract,
    }


def source_events(sources):
    """
    Make minimal verified reference records.

    IDs distinguish repeated practice attempts
    from initial/final assessment observations.
    """
    exercise_index = {
        row["exercise_id"]: row
        for row in sources["data"]["exercises"]
    }

    events = []

    for model in MODELS:
        session = sources["live"]["records"][model]

        for phase, field in (
            ("initial_assessment",
             "initial_assessment_records"),
            ("practice", "practice_records"),
            ("final_assessment",
             "final_assessment_records"),
        ):
            for position, record in enumerate(
                session[field], 1
            ):
                exercise = exercise_index[
                    record["exercise_id"]
                ]

                require(
                    exercise["model"] == model,
                    "Exercise model mismatch",
                )
                require(
                    exercise["failure_category"]
                    == record["category"]
                    and exercise["stage"]
                    == record["stage"],
                    "Exercise observation metadata mismatch",
                )
                require(
                    exercise["split"] == (
                        "practice" if phase == "practice"
                        else "assessment"
                    ),
                    "Phase/split mismatch",
                )

                if phase == "practice":
                    require(
                        record["sequence"] == position,
                        "Practice sequence mismatch",
                    )

                identity = {
                    "model": model,
                    "phase": phase,
                    "sequence": position,
                    "exercise_id": record["exercise_id"],
                }

                observation_id = (
                    "observation_" + digest(identity)[:24]
                )

                minimal = {
                    "observation_id": observation_id,
                    "model": model,
                    "episode_id": (
                        f"m6c3::{model}::{phase}::{position}"
                    ),
                    "phase": phase,
                    "source_split": (
                        "practice" if phase == "practice"
                        else "reserved_assessment"
                    ),
                    "exercise_id": record["exercise_id"],
                    "failure_category": record["category"],
                    "workflow_stage": record["stage"],
                    "observed_correct": record["correct"],
                    "guardian_decision": (
                        "CORRECT" if record["correct"]
                        else record["error"]
                    ),
                    "controller_scored": True,
                    "permissions_changed": False,
                    "tool_executed": False,
                }

                minimal["evidence_fingerprint"] = digest(
                    minimal
                )

                events.append(minimal)

    require(
        len(events) == 30,
        "Unexpected number of source observations",
    )

    require(
        len({
            e["observation_id"] for e in events
        }) == len(events),
        "Duplicate source IDs",
    )

    return events


def make_claim(event):
    """
    Example of a submitted claim envelope.

    A future agent may submit this JSON, but it
    does not control the reference evidence store.
    """
    return copy.deepcopy(event)


class ObservationRegistry:
    """
    Reference evidence store and append-only
    acceptance log.

    Python in-process access is not a security
    boundary. Production would separate storage,
    authorization, and ingestion processes.
    """

    def __init__(self, reference_events):
        require(
            isinstance(reference_events, list),
            "Invalid reference event collection",
        )

        self._reference = {}

        for event in reference_events:
            require(
                isinstance(event, dict),
                "Invalid reference event",
            )

            identifier = event["observation_id"]

            require(
                identifier not in self._reference,
                "Duplicate reference ID",
            )
            require(
                event["model"] in MODELS,
                "Unknown reference model",
            )

            unsigned = {
                k: v for k, v in event.items()
                if k != "evidence_fingerprint"
            }

            require(
                event["evidence_fingerprint"]
                == digest(unsigned),
                "Reference fingerprint mismatch",
            )

            self._reference[identifier] = copy.deepcopy(
                event
            )

        self._accepted = {}
        self._log = []

    def submit(self, claim):
        """
        Return an explicit decision.

        Invalid claims never change accepted
        evidence and never modify permissions.
        """
        if not isinstance(claim, dict):
            return {
                "accepted": False,
                "reason": "MALFORMED_CLAIM",
            }

        identifier = claim.get("observation_id")

        if not isinstance(identifier, str):
            return {
                "accepted": False,
                "reason": "MALFORMED_CLAIM",
            }

        if identifier not in self._reference:
            return {
                "accepted": False,
                "reason": "UNKNOWN_SOURCE",
            }

        if identifier in self._accepted:
            return {
                "accepted": False,
                "reason": "DUPLICATE_OBSERVATION",
            }

        reference = self._reference[identifier]

        if claim != reference:
            return {
                "accepted": False,
                "reason": "EVIDENCE_MISMATCH",
            }

        event = copy.deepcopy(reference)
        self._accepted[identifier] = event

        previous = (
            self._log[-1]["log_hash"]
            if self._log else GENESIS
        )

        audit_entry = {
            "sequence": len(self._log) + 1,
            "previous_hash": previous,
            "observation_id": identifier,
            "evidence_fingerprint":
                event["evidence_fingerprint"],
            "decision": "ACCEPTED",
        }
        audit_entry["log_hash"] = digest(audit_entry)

        self._log.append(audit_entry)

        return {
            "accepted": True,
            "reason": "VERIFIED",
            "observation_id": identifier,
        }

    def accepted(self):
        return [
            copy.deepcopy(self._accepted[key])
            for key in sorted(self._accepted)
        ]

    def audit_log(self):
        return copy.deepcopy(self._log)

    def summary(self):
        events = self.accepted()
        by_model = {}

        for model in MODELS:
            selected = [
                event for event in events
                if event["model"] == model
            ]

            by_model[model] = {
                "observations": len(selected),
                "correct": sum(
                    item["observed_correct"]
                    for item in selected
                ),
                "incorrect": sum(
                    not item["observed_correct"]
                    for item in selected
                ),
                "by_phase": dict(sorted(Counter(
                    item["phase"] for item in selected
                ).items())),
            }

        return {
            "accepted_observations": len(events),
            "by_model": by_model,
            "permission_changes": 0,
            "tool_executions": 0,
            "privilege_restorations": 0,
            "audit_entries": len(self._log),
            "audit_head": (
                self._log[-1]["log_hash"]
                if self._log else GENESIS
            ),
        }


def verify_audit(log, expected_head=None):
    previous = GENESIS

    for position, entry in enumerate(log, 1):
        require(
            entry["sequence"] == position,
            "Audit sequence mismatch",
        )
        require(
            entry["previous_hash"] == previous,
            "Audit chain mismatch",
        )
        require(
            entry["decision"] == "ACCEPTED",
            "Unexpected audit decision",
        )

        unsigned = {
            key: value for key, value in entry.items()
            if key != "log_hash"
        }

        require(
            entry["log_hash"] == digest(unsigned),
            "Audit log alteration",
        )

        previous = entry["log_hash"]

    if expected_head is not None:
        require(
            previous == expected_head,
            "Audit head mismatch",
        )

    return True
