
"""
Guardian AI — Module 4, Chapter 4.3

Safe, deterministic lesson memory and retrieval.

Source:
  Module 4 Chapter 4.2 approved corrections.

Security design:
  - Validate source against Chapter 4.1 evidence.
  - Verify library fingerprint.
  - Exact structural lookup, no semantic search.
  - No untrusted text in retrieval queries.
  - No raw attack text in stored corrections.
  - No permission changes or capability issuance.
  - Return immutable correction objects.
  - Avoid retrieval from evaluation examples.

No LLM calls or real tool execution.
"""

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
RESULTS = HERE / "results"

sys.path.insert(
    0, str(MODULE / "chapter_01")
)
sys.path.insert(
    0, str(MODULE / "chapter_02")
)

import m4_1_lesson_extraction as c1
import m4_2_lesson_validation as c2


LIBRARY_PATH = (
    MODULE / "chapter_02/results"
    / "approved_corrections.json"
)

SUMMARY_PATH = (
    MODULE / "chapter_02/results"
    / "lesson_quality_results.json"
)

DENIAL_TO_CATEGORY = dict(
    c1.DENIAL_CATEGORIES
)

TRUSTED_STAGES = frozenset(c2.STAGES)
TRUSTED_TOOLS = frozenset(
    c2.ALLOWED_PROPOSED_TOOLS
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class RetrievalRequest:
    """
    Constructed by a trusted Guardian controller.

    Do not create this from an LLM-written reason
    or from attacker-controlled tool output.
    """
    decision: str
    stage: str
    proposed_tool: str
    expected_tool: str


@dataclass(frozen=True)
class RetrievedCorrection:
    correction_id: str
    lesson_id: str
    category: str
    stage: str
    expected_tool: str
    instruction: str
    quality_score: int


def validate_request(request):
    require(
        type(request) is RetrievalRequest,
        "Invalid retrieval request type",
    )

    require(
        request.decision in DENIAL_TO_CATEGORY,
        "Unknown Guardian decision",
    )

    require(
        request.stage in TRUSTED_STAGES,
        "Unknown workflow stage",
    )

    require(
        request.proposed_tool in TRUSTED_TOOLS,
        "Unknown proposed tool",
    )

    require(
        request.expected_tool
        == c2.STAGES[request.stage],
        "Expected tool does not match stage",
    )

    category = DENIAL_TO_CATEGORY[
        request.decision
    ]

    if category == "schema_error":
        require(
            request.proposed_tool == "invalid_schema",
            "Schema error requires invalid-schema marker",
        )
    else:
        require(
            request.proposed_tool != "invalid_schema",
            "Unexpected invalid-schema marker",
        )

    if category == "target_mismatch":
        require(
            request.proposed_tool in (
                "search_documents",
                "read_document",
            ),
            "Invalid target-mismatch tool",
        )

    return category


class SafeLessonMemory:
    """
    Read-only, validated, controller-owned index.

    The object must be owned by trusted code.
    Its lookup API accepts structured decisions,
    not raw attacker-controlled strings.
    """

    def __init__(self, corrections):
        require(
            isinstance(corrections, list),
            "Invalid correction library",
        )

        self.__index = {}

        for item in corrections:
            require(
                isinstance(item, dict),
                "Invalid correction item",
            )

            category = item["category"]
            stage = item["stage"]

            require(
                category in c2.CORRECTION_TEMPLATES,
                "Unknown correction category",
            )

            require(
                stage in TRUSTED_STAGES,
                "Unknown correction stage",
            )

            require(
                item["expected_tool"]
                == c2.STAGES[stage],
                "Correction tool mismatch",
            )

            require(
                item["instruction"]
                == c2.CORRECTION_TEMPLATES[category],
                "Untrusted correction text",
            )

            require(
                item["authorization_change"] is False,
                "Permission-changing correction",
            )

            require(
                item["source_split"] == "train",
                "Held-out correction",
            )

            key = (
                category,
                stage,
                item["expected_tool"],
            )

            # Multiple lessons may cover a category
            # and stage with different observed tools.
            # Store them independently rather than
            # silently overwriting entries.
            self.__index.setdefault(
                key, []
            ).append({
                "correction_id": item["correction_id"],
                "lesson_id": item["lesson_id"],
                "category": category,
                "stage": stage,
                "expected_tool": item["expected_tool"],
                "proposed_tool": None,
                "instruction": item["instruction"],
                "quality_score": item["quality_score"],
            })

        # Ensure unique correction and lesson IDs.
        entries = [
            value
            for group in self.__index.values()
            for value in group
        ]

        require(
            len({
                e["correction_id"] for e in entries
            }) == len(entries),
            "Duplicate correction ID",
        )

        require(
            len({
                e["lesson_id"] for e in entries
            }) == len(entries),
            "Duplicate lesson ID",
        )

        self.__count = len(entries)

    def size(self):
        return self.__count

    def retrieve(self, request):
        """
        Return at most one correction.

        Prefer strongest evidence-supported lesson.
        Stable ID breaks ties deterministically.

        Note: Chapter 4.2 corrections do not include
        proposed_tool. Thus this version matches
        category + stage + expected_tool only.
        """
        category = validate_request(request)

        key = (
            category,
            request.stage,
            request.expected_tool,
        )

        candidates = self.__index.get(
            key, []
        )

        if not candidates:
            return None

        ranked = sorted(
            candidates,
            key=lambda x: (
                -x["quality_score"],
                x["correction_id"],
            ),
        )

        best = ranked[0]

        return RetrievedCorrection(
            correction_id=best["correction_id"],
            lesson_id=best["lesson_id"],
            category=best["category"],
            stage=best["stage"],
            expected_tool=best["expected_tool"],
            instruction=best["instruction"],
            quality_score=best["quality_score"],
        )


def load_verified_memory(
    library_path=LIBRARY_PATH,
    summary_path=SUMMARY_PATH,
):
    """
    Recompute Chapter 4.2 corrections from the
    original validated evidence.

    Reject local library modifications, even
    when an altered summary has a matching hash.
    """
    library = c2.read_json(library_path)
    saved_summary = c2.read_json(summary_path)

    evidence, lessons, _ = c2.load_sources()
    expected = c2.make_library(
        evidence, lessons
    )

    require(
        library == expected,
        "Correction library differs from source evidence",
    )

    require(
        saved_summary["correction_library_sha256"]
        == c1.digest(library),
        "Correction library fingerprint mismatch",
    )

    require(
        saved_summary["validated_corrections"]
        == len(library),
        "Correction count mismatch",
    )

    require(
        saved_summary["authorization_changes"] == 0
        and saved_summary["held_out_evidence_used"] == 0,
        "Invalid correction provenance",
    )

    return SafeLessonMemory(library)


def request_from_event(event, task):
    """
    A trusted adapter for an already-classified
    Guardian trace event.

    The decision label and trusted task/stage
    must come from the Guardian controller.
    Never call this on arbitrary model-generated
    JSON without separate verification.
    """
    require(
        isinstance(event, dict)
        and isinstance(task, dict),
        "Invalid event or task",
    )

    decision = event.get("status")
    stage = event.get("stage_before")

    if decision not in DENIAL_TO_CATEGORY:
        return None

    if stage not in TRUSTED_STAGES:
        return None

    expected = c1.expected_action(
        task, stage
    )

    if expected is None:
        return None

    action = event.get("action")

    proposed_tool = (
        action.get("tool")
        if isinstance(action, dict)
        else "invalid_schema"
    )

    request = RetrievalRequest(
        decision=decision,
        stage=stage,
        proposed_tool=proposed_tool,
        expected_tool=expected[0],
    )

    validate_request(request)

    return request


def correction_feedback(correction):
    """
    Bounded, controller-authored feedback.

    Deliberately excludes targets, file contents,
    raw model outputs and permissions.
    """
    if correction is None:
        return None

    require(
        isinstance(correction, RetrievedCorrection),
        "Unverified correction object",
    )

    require(
        correction.instruction
        == c2.CORRECTION_TEMPLATES[
            correction.category
        ],
        "Correction text mismatch",
    )

    require(
        correction.expected_tool
        == c2.STAGES[correction.stage],
        "Correction stage mismatch",
    )

    return (
        "Guardian correction "
        f"({correction.category}): "
        + correction.instruction
    )
