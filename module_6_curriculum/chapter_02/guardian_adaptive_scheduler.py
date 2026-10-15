
"""
Guardian AI — Module 6, Chapter 6.2

Adaptive, bounded synthetic safety curriculum.

The trusted controller:
  - loads verified Chapter 6.1 data
  - chooses PRACTICE exercises only
  - scores untrusted model proposals independently
  - updates performance-based skill priorities
  - enforces an immutable attempt budget
  - records hash-chained evidence

No assessment leakage.
No tool execution.
No capability issuance.
No automatic privilege restoration.

Priorities are scheduling heuristics, not
calibrated probabilities of future safety.
"""

import copy
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

sys.path.insert(
    0, str(HERE.parent / "chapter_01")
)

import m6_1_curriculum_dataset as c1


DATA_DIR = HERE.parent / "chapter_01/results"

GENESIS = hashlib.sha256(
    b"GUARDIAN_MODULE_6_C2_AUDIT_V1"
).hexdigest()


def require(value, reason):
    if not value:
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


def load_verified_dataset(data_dir=DATA_DIR):
    """
    Compare stored data against freshly rebuilt
    Chapter 6.1 training-only sources.

    This is an integrity/reproducibility check,
    not external authentication of the files.
    """
    data_dir = Path(data_dir)

    stored_profiles = c1.read_json(
        data_dir / "agent_risk_profiles.json"
    )
    stored_exercises = c1.read_json(
        data_dir / "curriculum_exercises.json"
    )
    stored_summary = c1.read_json(
        data_dir / "m6_1_summary.json"
    )
    manifest = c1.read_json(
        data_dir / "manifest.json"
    )

    sources = c1.load_verified_sources()

    expected_profiles = c1.make_risk_profiles(
        sources["evidence"],
        sources["donors"],
    )
    expected_exercises = c1.make_exercises(
        sources["donors"]
    )
    expected_summary = c1.summarize(
        expected_profiles, expected_exercises
    )

    require(
        stored_profiles == expected_profiles,
        "Risk profile/source mismatch",
    )
    require(
        stored_exercises == expected_exercises,
        "Curriculum/source mismatch",
    )
    require(
        stored_summary == expected_summary,
        "Summary/source mismatch",
    )

    for key, value in (
        ("profiles_sha256", digest(stored_profiles)),
        ("exercises_sha256", digest(stored_exercises)),
        ("summary_sha256", digest(stored_summary)),
    ):
        require(
            manifest[key] == value,
            f"Manifest mismatch: {key}",
        )

    c1.validate_exercises(
        stored_exercises, sources["donors"]
    )

    return {
        "profiles": copy.deepcopy(stored_profiles),
        "exercises": copy.deepcopy(stored_exercises),
        "summary": copy.deepcopy(stored_summary),
        "dataset_sha256": digest(stored_exercises),
    }


def skill_key(exercise):
    return (
        exercise["lesson_id"],
        exercise["failure_category"],
        exercise["stage"],
    )


@dataclass(frozen=True)
class Assignment:
    sequence: int
    exercise_id: str
    model: str
    category: str
    stage: str
    prompt: str
    priority: float

    # Gold answers are intentionally not
    # exposed through this assignment object.


class AdaptiveScheduler:
    def __init__(
        self,
        model,
        profile,
        exercises,
        max_attempts=8,
    ):
        require(
            model in c1.MODELS,
            "Unknown model",
        )
        require(
            type(max_attempts) is int
            and 1 <= max_attempts <= 100,
            "Invalid attempt budget",
        )
        require(
            isinstance(profile, dict)
            and profile["model"] == model,
            "Profile/model mismatch",
        )

        self._model = model
        self._budget = max_attempts

        self._practice = {
            row["exercise_id"]: copy.deepcopy(row)
            for row in exercises
            if row["model"] == model
            and row["split"] == "practice"
        }

        require(
            len(self._practice) > 0,
            "No training exercises",
        )

        self._skills = {}

        weights = {
            item["category"]: item["priority_weight"]
            for item in profile["category_priorities"]
        }

        for exercise in self._practice.values():
            key = skill_key(exercise)

            if key not in self._skills:
                # Minimum base priority for observed
                # but comparatively rare categories.
                self._skills[key] = {
                    "category": exercise["failure_category"],
                    "stage": exercise["stage"],
                    "base": max(
                        0.05,
                        weights[exercise["failure_category"]],
                    ),
                    "attempts": 0,
                    "correct": 0,
                    "incorrect": 0,
                }

        self._exercise_usage = Counter()
        self._events = []
        self._pending = None

    def _priority(self, key):
        state = self._skills[key]

        risk = 1 + 0.6 * min(
            state["incorrect"], 5
        )
        mastery = 1 + 0.4 * state["correct"]
        coverage = 1 + 0.22 * state["attempts"]

        return state["base"] * risk / (
            mastery * coverage
        )

    def _choose(self):
        """
        Highest priority skill first.

        Within a skill, choose the least-used
        practice exercise. Stable IDs break ties.
        """
        candidates = []

        for item in self._practice.values():
            key = skill_key(item)
            priority = self._priority(key)

            candidates.append((
                -priority,
                self._exercise_usage[item["exercise_id"]],
                item["exercise_id"],
                item,
            ))

        candidates.sort(
            key=lambda row: row[:3]
        )

        priority = -candidates[0][0]
        return candidates[0][3], priority

    def remaining(self):
        return self._budget - len(self._events)

    def next_exercise(self):
        if self._pending is not None:
            raise RuntimeError(
                "Complete the pending exercise first"
            )

        if self.remaining() == 0:
            return None

        item, priority = self._choose()

        self._pending = item["exercise_id"]

        return Assignment(
            sequence=len(self._events) + 1,
            exercise_id=item["exercise_id"],
            model=self._model,
            category=item["failure_category"],
            stage=item["stage"],
            prompt=item["prompt"],
            priority=round(priority, 6),
        )

    def submit(self, assignment, raw_proposal):
        """
        The only performance update path.

        The model cannot set correct=True or
        supply a self-reported mastery score.
        """
        require(
            type(assignment) is Assignment,
            "Invalid assignment",
        )

        if self._pending is None:
            raise RuntimeError(
                "No pending practice exercise"
            )

        require(
            assignment.exercise_id == self._pending
            and assignment.model == self._model
            and assignment.sequence == len(self._events) + 1,
            "Assignment does not match pending exercise",
        )

        exercise = self._practice[self._pending]

        require(
            assignment.category
            == exercise["failure_category"]
            and assignment.stage == exercise["stage"]
            and assignment.prompt == exercise["prompt"],
            "Assignment metadata mismatch",
        )

        score = c1.score_proposal(
            exercise, raw_proposal
        )

        key = skill_key(exercise)
        skill = self._skills[key]

        skill["attempts"] += 1

        if score["correct"]:
            skill["correct"] += 1
        else:
            skill["incorrect"] += 1

        self._exercise_usage[
            exercise["exercise_id"]
        ] += 1

        previous = (
            self._events[-1]["event_hash"]
            if self._events else GENESIS
        )

        event = {
            "sequence": assignment.sequence,
            "previous_hash": previous,
            "model": self._model,
            "exercise_id": exercise["exercise_id"],
            "lesson_id": exercise["lesson_id"],
            "split": "practice",
            "category": exercise["failure_category"],
            "stage": exercise["stage"],
            "correct": score["correct"],
            "error": score["error"],
            "priority_after": round(
                self._priority(key), 6
            ),
            "tool_executed": False,
            "privileges_changed": False,
        }

        event["event_hash"] = digest(event)

        self._events.append(event)
        self._pending = None

        return copy.deepcopy({
            "correct": score["correct"],
            "error": score["error"],
            "attempts_used": len(self._events),
            "attempts_remaining": self.remaining(),
            "priority_after": event["priority_after"],
            "tool_executed": False,
            "privileges_changed": False,
        })

    def events(self):
        return copy.deepcopy(self._events)

    def priorities(self):
        return [
            {
                "lesson_id": key[0],
                "category": key[1],
                "stage": key[2],
                "priority": round(
                    self._priority(key), 6
                ),
                **{
                    field: self._skills[key][field]
                    for field in (
                        "attempts",
                        "correct",
                        "incorrect",
                    )
                },
            }
            for key in sorted(self._skills)
        ]

    def summary(self):
        correct = sum(
            row["correct"] for row in self._events
        )

        return {
            "model": self._model,
            "budget": self._budget,
            "attempts": len(self._events),
            "correct": correct,
            "incorrect": len(self._events) - correct,
            "accuracy": round(
                correct / len(self._events), 4
            ) if self._events else None,
            "exercise_ids_used": len({
                event["exercise_id"]
                for event in self._events
            }),
            "assessment_exercises_used": 0,
            "tool_executions": 0,
            "privilege_restorations": 0,
            "final_audit_hash": (
                self._events[-1]["event_hash"]
                if self._events else GENESIS
            ),
        }


def verify_events(events, expected_hash=None):
    previous = GENESIS

    for sequence, event in enumerate(events, 1):
        require(
            event["sequence"] == sequence,
            "Incorrect audit sequence",
        )
        require(
            event["previous_hash"] == previous,
            "Broken audit chain",
        )

        unsigned = {
            k: v for k, v in event.items()
            if k != "event_hash"
        }

        require(
            event["event_hash"] == digest(unsigned),
            "Modified audit event",
        )
        require(
            event["split"] == "practice"
            and event["tool_executed"] is False
            and event["privileges_changed"] is False,
            "Unsafe audit event",
        )

        previous = event["event_hash"]

    if expected_hash is not None:
        require(
            previous == expected_hash,
            "External audit anchor mismatch",
        )

    return True
