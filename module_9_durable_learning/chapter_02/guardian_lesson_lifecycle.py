
"""
Guardian AI — Module 9, Chapter 9.2

Durable, event-sourced safety lesson lifecycle.

States:
    PENDING_REVIEW -> APPROVED -> ACTIVE -> ROLLED_BACK

Features:
    - persistent append-only logical event history
    - idempotent commands
    - hash-chained event verification
    - trusted reviewer allowlist
    - activation disabled by default
    - irreversible rollback for this prototype
    - original lesson candidates remain unapproved

Important:
    This is a single-writer research prototype.
    A local hash chain is not externally tamper-proof.
    Reviewer allowlisting is not identity authentication.
    ACTIVE has no connection to Guardian's live tools.
"""

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

sys.path.insert(
    0,
    str(ROOT / "module_8_observatory/chapter_03"),
)

import guardian_lesson_candidates as m8

GENESIS = hashlib.sha256(
    b"GUARDIAN_M9_C2_LESSON_JOURNAL_V1"
).hexdigest()

STATES = (
    "PENDING_REVIEW",
    "APPROVED",
    "ACTIVE",
    "ROLLED_BACK",
)

TRANSITIONS = {
    "SUBMIT": (None, "PENDING_REVIEW"),
    "APPROVE": ("PENDING_REVIEW", "APPROVED"),
    "ACTIVATE": ("APPROVED", "ACTIVE"),
    "ROLLBACK": ("ACTIVE", "ROLLED_BACK"),
}

EVENT_FIELDS = {
    "sequence",
    "previous_hash",
    "event_hash",
    "command_key",
    "command_sha256",
    "lesson_id",
    "action",
    "actor",
    "from_state",
    "to_state",
    "lesson_sha256",
    "review_evidence_sha256",
}

APPROVED_REVIEWERS = frozenset({
    "trusted_reviewer_fixture",
})


def require(condition, message):
    if not condition:
        raise ValueError(message)


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


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def atomic_json(path, value):
    """
    Atomically replace the whole journal.
    Single-writer only; no multi-process locking.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pending = path.with_name(
        path.name + ".pending"
    )

    require(
        not pending.exists(),
        "Unresolved pending journal write",
    )

    with pending.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(pending, path)


def trusted_candidates():
    """
    Rebuild original Chapter 8 candidates.
    Do not trust a model-submitted candidate.
    """
    source = m8.load_sources()
    candidates = m8.make_candidates(
        source["predictions"]
    )

    m8.validate_candidates(
        candidates,
        m8.group_candidates(candidates),
        source["predictions"],
    )

    return candidates


def safe_lesson(candidate):
    """
    Project only controller-owned fields.
    Exclude raw untrusted content.
    """
    require(
        candidate["published"] is False
        and candidate["review_state"] == "PENDING_REVIEW"
        and candidate["permissions_changed"] is False,
        "Untrusted lesson status",
    )

    return {
        "lesson_id": candidate["candidate_id"],
        "template_id": candidate["template_id"],
        "category": candidate["category"],
        "lesson_text": candidate["lesson_text"],
        "source_snapshot_id": candidate["snapshot_id"],
        "source_content_sha256":
            candidate["content_sha256"],
    }


def make_fixture_lesson():
    """Separate synthetic item for transition tests."""
    return {
        "lesson_id": "fixture_lifecycle_lesson_v1",
        "template_id": "fixture_only",
        "category": "synthetic_test",
        "lesson_text": (
            "Keep the original trusted task boundary."
        ),
        "source_snapshot_id": "fixture_only",
        "source_content_sha256": digest(
            "synthetic lifecycle evidence"
        ),
    }


def validate_lesson(lesson):
    expected_keys = {
        "lesson_id",
        "template_id",
        "category",
        "lesson_text",
        "source_snapshot_id",
        "source_content_sha256",
    }

    require(
        type(lesson) is dict
        and set(lesson) == expected_keys,
        "Malformed lesson",
    )

    require(
        all(
            type(lesson[key]) is str
            and bool(lesson[key])
            for key in expected_keys
        ),
        "Empty lesson field",
    )

    require(
        len(lesson["lesson_text"]) <= 500
        and len(lesson["lesson_id"]) <= 128,
        "Lesson exceeds bounds",
    )

    return True


class LessonJournal:
    """
    Persistent, replay-verified lifecycle journal.

    Default mode never permits activation.
    Test mode permits internal ACTIVE transitions
    for a single synthetic fixture only.
    """

    def __init__(
        self,
        storage_dir,
        allow_fixture_activation=False,
    ):
        self.root = Path(storage_dir)
        self.root.mkdir(parents=True, exist_ok=True)

        self.path = self.root / "lesson_journal.json"
        self.allow_fixture_activation = (
            allow_fixture_activation is True
        )

        self.events = (
            read_json(self.path)
            if self.path.exists()
            else []
        )

        self.verify()

    @staticmethod
    def _command_key_valid(key):
        return (
            type(key) is str
            and 1 <= len(key) <= 100
            and all(
                ch.isalnum() or ch in "-_"
                for ch in key
            )
        )

    def _replay(self):
        """
        Derive state exclusively from stored events.
        Detect inconsistent lifecycle transitions.
        """
        require(
            type(self.events) is list,
            "Malformed journal",
        )

        states = {}
        hashes = {}
        submitted_lessons = {}
        command_index = {}
        previous = GENESIS

        for number, event in enumerate(
            self.events, 1
        ):
            require(
                type(event) is dict
                and set(event) == EVENT_FIELDS,
                "Malformed journal event",
            )

            unsigned = {
                key: value
                for key, value in event.items()
                if key != "event_hash"
            }

            require(
                event["sequence"] == number
                and event["previous_hash"] == previous
                and event["event_hash"] == digest(unsigned),
                "Broken event hash chain",
            )

            require(
                event["action"] in TRANSITIONS
                and self._command_key_valid(
                    event["command_key"]
                ),
                "Invalid journal action",
            )

            require(
                event["command_key"] not in command_index,
                "Duplicate persisted command key",
            )

            lesson_id = event["lesson_id"]
            from_state, to_state = TRANSITIONS[
                event["action"]
            ]

            require(
                states.get(lesson_id) == from_state
                and event["from_state"] == from_state
                and event["to_state"] == to_state,
                "Illegal recorded transition",
            )

            if event["action"] == "SUBMIT":
                require(
                    event["review_evidence_sha256"] is None,
                    "Submission cannot claim review evidence",
                )
                hashes[lesson_id] = event["lesson_sha256"]
                submitted_lessons[lesson_id] = True
            else:
                require(
                    lesson_id in submitted_lessons
                    and hashes[lesson_id]
                    == event["lesson_sha256"],
                    "Lesson identity was modified",
                )

            if event["action"] == "APPROVE":
                require(
                    event["actor"] in APPROVED_REVIEWERS
                    and type(
                        event["review_evidence_sha256"]
                    ) is str,
                    "Untrusted approval",
                )

            if event["action"] in (
                "ACTIVATE", "ROLLBACK"
            ):
                require(
                    event["actor"] in APPROVED_REVIEWERS,
                    "Untrusted activation or rollback",
                )

            states[lesson_id] = to_state
            command_index[event["command_key"]] = (
                event["command_sha256"]
            )
            previous = event["event_hash"]

        return {
            "states": states,
            "lesson_hashes": hashes,
            "commands": command_index,
            "head": previous,
        }

    def verify(self):
        self._replay()
        return True

    def state(self, lesson_id):
        return self._replay()["states"].get(lesson_id)

    def command(
        self,
        action,
        lesson,
        actor,
        command_key,
        review_evidence=None,
    ):
        validate_lesson(lesson)
        require(
            action in TRANSITIONS,
            "Unknown lifecycle action",
        )
        require(
            self._command_key_valid(command_key),
            "Invalid idempotency key",
        )
        require(
            type(actor) is str
            and actor,
            "Invalid actor",
        )

        lesson_id = lesson["lesson_id"]

        command_payload = {
            "action": action,
            "lesson": lesson,
            "actor": actor,
            "command_key": command_key,
            "review_evidence": review_evidence,
        }
        command_sha256 = digest(command_payload)

        current = self._replay()

        if command_key in current["commands"]:
            if current["commands"][command_key] == command_sha256:
                return {
                    "accepted": False,
                    "reason": "IDEMPOTENT_REPLAY",
                    "state": current["states"].get(lesson_id),
                }
            return {
                "accepted": False,
                "reason": "IDEMPOTENCY_CONFLICT",
                "state": current["states"].get(lesson_id),
            }

        before, after = TRANSITIONS[action]
        state = current["states"].get(lesson_id)

        if state != before:
            return {
                "accepted": False,
                "reason": "INVALID_TRANSITION",
                "state": state,
            }

        if action != "SUBMIT":
            if current["lesson_hashes"].get(
                lesson_id
            ) != digest(lesson):
                return {
                    "accepted": False,
                    "reason": "LESSON_CHANGED",
                    "state": state,
                }

        if action in ("APPROVE", "ACTIVATE", "ROLLBACK"):
            if actor not in APPROVED_REVIEWERS:
                return {
                    "accepted": False,
                    "reason": "REVIEWER_NOT_AUTHORIZED",
                    "state": state,
                }

        if action == "APPROVE":
            if not (
                type(review_evidence) is dict
                and review_evidence.get(
                    "independent_validation"
                ) is True
                and review_evidence.get(
                    "trusted_review"
                ) is True
                and review_evidence.get(
                    "experiment_id"
                )
            ):
                return {
                    "accepted": False,
                    "reason": "INSUFFICIENT_REVIEW_EVIDENCE",
                    "state": state,
                }
        else:
            require(
                review_evidence is None,
                "Unexpected review evidence",
            )

        if action == "ACTIVATE":
            if (
                not self.allow_fixture_activation
                or lesson_id
                != "fixture_lifecycle_lesson_v1"
            ):
                return {
                    "accepted": False,
                    "reason": "ACTIVATION_DISABLED",
                    "state": state,
                }

        # The original Module 8 lessons must stay
        # pending until independent behavioral tests.
        if (
            action == "APPROVE"
            and lesson_id
            != "fixture_lifecycle_lesson_v1"
        ):
            return {
                "accepted": False,
                "reason": "EVIDENCE_NOT_INDEPENDENTLY_VERIFIED",
                "state": state,
            }

        previous = current["head"]

        event = {
            "sequence": len(self.events) + 1,
            "previous_hash": previous,
            "command_key": command_key,
            "command_sha256": command_sha256,
            "lesson_id": lesson_id,
            "action": action,
            "actor": actor,
            "from_state": before,
            "to_state": after,
            "lesson_sha256": digest(lesson),
            "review_evidence_sha256": (
                digest(review_evidence)
                if review_evidence is not None
                else None
            ),
        }

        event["event_hash"] = digest(event)

        updated = self.events + [event]
        atomic_json(self.path, updated)

        self.events = updated
        self.verify()

        return {
            "accepted": True,
            "reason": "TRANSITION_RECORDED",
            "state": after,
        }

    def history(self):
        self.verify()
        return copy.deepcopy(self.events)

    def summary(self):
        state = self._replay()["states"]

        return {
            "lessons": len(state),
            "pending_review": sum(
                x == "PENDING_REVIEW"
                for x in state.values()
            ),
            "approved": sum(
                x == "APPROVED"
                for x in state.values()
            ),
            "active": sum(
                x == "ACTIVE"
                for x in state.values()
            ),
            "rolled_back": sum(
                x == "ROLLED_BACK"
                for x in state.values()
            ),
            "audit_events": len(self.events),
            "audit_head": self._replay()["head"],
            "guardian_permission_changes": 0,
            "live_guardian_lessons_activated": 0,
        }
