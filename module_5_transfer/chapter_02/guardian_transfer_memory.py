
"""
Guardian AI — Module 5, Chapter 5.2

Donor-aware, exact-match safety lesson retrieval.

- Verify donor libraries against source evidence.
- Bind each lesson to its actual donor model.
- Require explicit donor and recipient identities.
- Match failure category, stage and proposed tool.
- Return no correction on missing donor support.
- Keep correction text controller-authored.
- Never change Guardian authorization.

Research prototype, not a process-security boundary.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"

sys.path.insert(
    0, str(HERE.parent / "chapter_01")
)

import m5_1_transfer_dataset as c1


DONOR_PATH = (
    HERE.parent / "chapter_01/results/donor_libraries.json"
)
TRANSFER_PATH = (
    HERE.parent / "chapter_01/results/transfer_dataset.json"
)
MANIFEST_PATH = (
    HERE.parent / "chapter_01/results/manifest.json"
)

DENIAL_TO_CATEGORY = dict(
    c1.c1.DENIAL_CATEGORIES
)
STAGES = dict(c1.c2.STAGES)

VALID_PROPOSED_TOOLS = frozenset(
    c1.c2.ALLOWED_PROPOSED_TOOLS
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class TransferRequest:
    donor_model: str
    recipient_model: str
    decision: str
    stage: str
    proposed_tool: str
    expected_tool: str


@dataclass(frozen=True)
class TransferCorrection:
    transfer_id: str
    donor_model: str
    recipient_model: str
    direction: str
    lesson_id: str
    correction_id: str
    category: str
    stage: str
    proposed_tool: str
    expected_tool: str
    instruction: str
    donor_observations: int


def validate_request(request):
    require(
        type(request) is TransferRequest,
        "Invalid transfer request type",
    )
    require(
        request.donor_model in c1.MODELS,
        "Unknown donor model",
    )
    require(
        request.recipient_model in c1.MODELS,
        "Unknown recipient model",
    )
    require(
        request.decision in DENIAL_TO_CATEGORY,
        "Unknown Guardian denial",
    )
    require(
        request.stage in STAGES,
        "Invalid stage",
    )
    require(
        request.expected_tool == STAGES[request.stage],
        "Expected tool/stage mismatch",
    )
    require(
        request.proposed_tool in VALID_PROPOSED_TOOLS,
        "Unknown proposed tool",
    )

    category = DENIAL_TO_CATEGORY[request.decision]

    if category == "schema_error":
        require(
            request.proposed_tool == "invalid_schema",
            "Schema denial requires invalid marker",
        )
    else:
        require(
            request.proposed_tool != "invalid_schema",
            "Unexpected invalid-schema marker",
        )

    if category == "target_mismatch":
        require(
            request.proposed_tool in (
                "search_documents", "read_document"
            ),
            "Invalid target-mismatch proposal",
        )

    return category


class DonorAwareMemory:
    """
    Read-only model-specific correction index.

    Note: Python-level encapsulation does not
    prevent hostile same-process introspection.
    """

    def __init__(self, transfers):
        require(
            isinstance(transfers, list),
            "Invalid transfer dataset",
        )

        index = {}
        seen_ids = set()

        for row in transfers:
            require(
                row["donor_model"] in c1.MODELS
                and row["recipient_model"] in c1.MODELS,
                "Invalid transfer model",
            )
            require(
                row["authorization_change"] is False,
                "Permission-changing transfer",
            )
            require(
                row["status"] == "candidate_transfer",
                "Unapproved transfer status",
            )

            category = row["failure_category"]
            require(
                category in c1.c2.CORRECTION_TEMPLATES,
                "Unknown correction category",
            )
            require(
                row["correction_text"]
                == c1.c2.CORRECTION_TEMPLATES[category],
                "Untrusted correction text",
            )
            require(
                row["stage"] in STAGES
                and row["expected_tool"] == STAGES[row["stage"]],
                "Invalid stage/tool correspondence",
            )
            require(
                row["proposed_tool"] in VALID_PROPOSED_TOOLS,
                "Invalid proposed tool",
            )
            require(
                row["donor_observations"] >= 1,
                "Unsupported donor lesson",
            )

            direction = (
                "same_model"
                if row["donor_model"] == row["recipient_model"]
                else "cross_model"
            )
            require(
                row["direction"] == direction,
                "Incorrect transfer direction",
            )

            require(
                row["transfer_id"] not in seen_ids,
                "Duplicate transfer identifier",
            )
            seen_ids.add(row["transfer_id"])

            key = (
                row["donor_model"],
                row["recipient_model"],
                category,
                row["stage"],
                row["proposed_tool"],
                row["expected_tool"],
            )

            require(
                key not in index,
                "Ambiguous duplicate retrieval key",
            )

            index[key] = TransferCorrection(
                transfer_id=row["transfer_id"],
                donor_model=row["donor_model"],
                recipient_model=row["recipient_model"],
                direction=direction,
                lesson_id=row["lesson_id"],
                correction_id=row["correction_id"],
                category=category,
                stage=row["stage"],
                proposed_tool=row["proposed_tool"],
                expected_tool=row["expected_tool"],
                instruction=row["correction_text"],
                donor_observations=row["donor_observations"],
            )

        self.__index = index
        self.__count = len(index)

    def size(self):
        return self.__count

    def retrieve(self, request):
        category = validate_request(request)

        key = (
            request.donor_model,
            request.recipient_model,
            category,
            request.stage,
            request.proposed_tool,
            request.expected_tool,
        )

        return self.__index.get(key)


def load_verified_memory(
    donor_path=DONOR_PATH,
    transfer_path=TRANSFER_PATH,
    manifest_path=MANIFEST_PATH,
):
    """
    Verify exact correspondence with all Chapter 5.1
    sources, not merely user-editable JSON hashes.
    """
    libraries = c1.c2.read_json(donor_path)
    transfers = c1.c2.read_json(transfer_path)
    manifest = c1.c2.read_json(manifest_path)

    evidence, lessons, corrections, hashes = (
        c1.source_data()
    )

    expected_libraries = c1.make_donor_libraries(
        evidence, lessons, corrections
    )
    expected_transfers = c1.make_transfer_dataset(
        expected_libraries
    )

    require(
        libraries == expected_libraries,
        "Donor libraries do not match source evidence",
    )
    require(
        transfers == expected_transfers,
        "Transfers do not match donor evidence",
    )
    require(
        manifest["source_fingerprints"] == hashes,
        "Source fingerprint mismatch",
    )
    require(
        manifest["donor_libraries_sha256"]
        == c1.fingerprint(libraries),
        "Donor library checksum mismatch",
    )
    require(
        manifest["transfer_dataset_sha256"]
        == c1.fingerprint(transfers),
        "Transfer checksum mismatch",
    )

    c1.validate_transfer_dataset(
        transfers, libraries
    )

    return DonorAwareMemory(transfers)


def request_from_guardian_event(
    event,
    trusted_task,
    donor_model,
    recipient_model,
):
    """
    Adapter intended for trusted Guardian event
    objects only, never arbitrary model output.

    The donor and recipient are externally fixed.
    """
    require(
        isinstance(event, dict)
        and isinstance(trusted_task, dict),
        "Invalid event/task",
    )

    decision = event.get("status")
    stage = event.get("stage_before")

    if decision not in DENIAL_TO_CATEGORY:
        return None

    if stage not in STAGES:
        return None

    expected = c1.c1.expected_action(
        trusted_task, stage
    )

    if expected is None:
        return None

    action = event.get("action")
    proposed_tool = (
        action.get("tool")
        if isinstance(action, dict)
        else "invalid_schema"
    )

    request = TransferRequest(
        donor_model=donor_model,
        recipient_model=recipient_model,
        decision=decision,
        stage=stage,
        proposed_tool=proposed_tool,
        expected_tool=expected[0],
    )

    validate_request(request)
    return request


def transfer_feedback(correction):
    """
    The only agent-visible transfer feedback.

    Do not include donor raw failure content,
    scenario IDs, document names or permissions.
    """
    if correction is None:
        return None

    require(
        type(correction) is TransferCorrection,
        "Invalid transfer correction object",
    )
    require(
        correction.category in c1.c2.CORRECTION_TEMPLATES,
        "Unknown transfer category",
    )
    require(
        correction.instruction
        == c1.c2.CORRECTION_TEMPLATES[
            correction.category
        ],
        "Correction text mismatch",
    )

    return (
        f"Guardian lesson ({correction.category}): "
        + correction.instruction
    )
