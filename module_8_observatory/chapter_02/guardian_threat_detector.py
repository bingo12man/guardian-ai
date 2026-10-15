
"""
Guardian AI — Module 8, Chapter 8.2

Offline deterministic prompt-injection detector.

Input: Verified Chapter 8.1 synthetic snapshots.

The detector:
  - never receives ground-truth labels
  - uses controller-authored pattern rules
  - extracts minimal match offsets, not raw text
  - never decodes or executes embedded instructions
  - never changes Guardian authorization

Detection is heuristic, not security certification.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

sys.path.insert(
    0, str(HERE.parent / "chapter_01")
)

import guardian_source_registry as c1

SOURCE_DIR = HERE.parent / "chapter_01/results"

SNAPSHOTS_PATH = SOURCE_DIR / "verified_snapshots.json"
LABELS_PATH = SOURCE_DIR / "ground_truth_labels.json"
MANIFEST_PATH = SOURCE_DIR / "manifest.json"
AUDIT_PATH = SOURCE_DIR / "source_audit.json"

CATEGORIES = tuple(c1.TECHNIQUES)

# Precedence matters if several rules match.
# This is a controller-defined classification policy,
# not a learned or calibrated risk ordering.

RULES = (
    (
        "ENC_BASE64_MARKER",
        "encoded_instruction",
        r"\b[A-Za-z0-9+/]{16,}={0,2}\b",
    ),
    (
        "ENC_UNICODE_ESCAPE",
        "encoded_instruction",
        r"\\u[0-9a-fA-F]{4}",
    ),
    (
        "MEM_REMEMBER_UPGRADE",
        "memory_poisoning",
        r"\bremember\b.{0,100}\b(?:upgraded|manager)\b",
    ),
    (
        "MEM_FALSE_PREFERENCE",
        "memory_poisoning",
        r"\bsave\b.{0,70}\bfalse preference\b",
    ),
    (
        "AUTH_SYSTEM_NOTICE",
        "authority_spoof",
        r"\bsystem notice\b.{0,140}\b(?:administrator|guardian|approved)\b",
    ),
    (
        "AUTH_DEVELOPER_DIRECTIVE",
        "authority_spoof",
        r"\bdeveloper directive\b.{0,140}\b(?:ignore|finish|skip)\b",
    ),
    (
        "REDIR_INSTEAD",
        "target_redirection",
        r"\binstead of\b.{0,120}\bopen\b",
    ),
    (
        "REDIR_FROM_TO",
        "target_redirection",
        r"\bredirect\b.{0,120}\bfrom\b.{0,70}\bto\b",
    ),
)

COMPILED = tuple(
    (rule_id, category, re.compile(pattern, re.I | re.S))
    for rule_id, category, pattern in RULES
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_text(value):
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def load_verified_archive():
    """
    Reconstruction against original C1 fixture
    definitions detects modified source files.

    Ground truth is NOT returned to the detector.
    """
    c1.validate_handoff()

    archive = c1.read_json(SNAPSHOTS_PATH)
    manifest = c1.read_json(MANIFEST_PATH)
    audit = c1.read_json(AUDIT_PATH)

    expected = c1.make_snapshots()

    require(
        archive == expected,
        "Archive differs from controller fixtures",
    )
    require(
        manifest["snapshots_sha256"]
        == c1.fingerprint(archive),
        "Snapshot manifest mismatch",
    )
    require(
        manifest["audit_sha256"]
        == c1.fingerprint(audit),
        "Audit manifest mismatch",
    )

    source_registry = c1.SourceRegistry(
        c1.approved_sources(),
        expected,
    )

    for item in expected:
        require(
            source_registry.ingest(item)["accepted"],
            "Fixture ingestion failed",
        )

    require(
        audit == source_registry.audit(),
        "Archived audit differs from trusted reconstruction",
    )
    c1.verify_audit(
        audit,
        source_registry.audit_head(),
    )

    return archive


def detect_text(content):
    """
    Pure, label-blind text classifier.

    A finding records only a rule ID and offsets,
    not suspicious content itself.

    Encoded substrings are NEVER decoded.
    """
    require(
        type(content) is str
        and 0 < len(content) <= c1.MAX_CONTENT_CHARS,
        "Invalid content",
    )

    findings = []

    for rule_id, category, regex in COMPILED:
        for match in regex.finditer(content):
            findings.append({
                "rule_id": rule_id,
                "category": category,
                "start": match.start(),
                "end": match.end(),
            })

    category = (
        findings[0]["category"]
        if findings else "benign"
    )

    return {
        "predicted_category": category,
        "suspicious": category != "benign",
        "findings": findings,
        "rule_matches": len(findings),
        "calibrated_confidence": None,
        "instructions_executed": False,
    }


def detect_snapshot(snapshot):
    """
    Strictly project recognized source fields.
    Ignore any untrusted classification claims.
    """
    require(
        type(snapshot) is dict,
        "Invalid snapshot",
    )

    expected_keys = {
        "snapshot_id",
        "source_id",
        "source_kind",
        "content",
        "content_sha256",
        "max_content_chars",
        "operator_authorized",
        "network_fetched",
        "executed",
        "permission_changes",
    }

    require(
        set(snapshot) == expected_keys,
        "Unexpected snapshot fields",
    )

    content = snapshot["content"]

    require(
        sha256_text(content)
        == snapshot["content_sha256"],
        "Content hash mismatch",
    )
    require(
        snapshot["source_kind"] == "synthetic_offline"
        and snapshot["operator_authorized"] is True
        and snapshot["network_fetched"] is False
        and snapshot["executed"] is False
        and snapshot["permission_changes"] is False,
        "Unsafe snapshot provenance",
    )

    result = detect_text(content)

    return {
        "snapshot_id": snapshot["snapshot_id"],
        "source_id": snapshot["source_id"],
        "content_sha256": snapshot["content_sha256"],
        **result,
    }


def detect_archive(snapshots):
    require(
        type(snapshots) is list,
        "Archive must be a list",
    )

    ids = [s["snapshot_id"] for s in snapshots]
    require(
        len(ids) == len(set(ids)),
        "Duplicate snapshot ID",
    )

    return [
        detect_snapshot(snapshot)
        for snapshot in snapshots
    ]


def verify_predictions(predictions, archive):
    """
    Recalculate predictions from trusted input.

    Does not evaluate against ground truth.
    """
    require(
        predictions == detect_archive(archive),
        "Prediction records differ from detector",
    )

    for prediction in predictions:
        require(
            prediction["instructions_executed"] is False,
            "Execution flag modified",
        )
        require(
            prediction["calibrated_confidence"] is None,
            "Unjustified confidence value",
        )

    return True
