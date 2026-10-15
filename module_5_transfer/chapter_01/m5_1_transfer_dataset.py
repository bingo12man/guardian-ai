
"""
Guardian AI — Module 5, Chapter 5.1

Build provenance-validated donor/recipient
safety transfer datasets.

Source:
  Module 4 training-denial evidence and validated
  controller-authored correction templates.

Outputs:
  donor_libraries.json
  transfer_dataset.json
  m5_1_summary.json
  manifest.json
  M5_C1_FINDINGS.md

Security:
  - Training evidence only
  - No raw model output or injected instructions
  - No target/resource names in transfer lessons
  - No new authorization
  - Donor support explicitly established
  - Held-out scenario families not used in learning

Research limitation:
  Evidence belongs to donors, while policy
  templates were authored by the controller.
"""

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"

sys.path.insert(
    0, str(ROOT / "module_4_correction/chapter_01")
)
sys.path.insert(
    0, str(ROOT / "module_4_correction/chapter_02")
)

import m4_1_lesson_extraction as c1
import m4_2_lesson_validation as c2


MODELS = (
    "llama3.2:3b",
    "qwen2.5:3b",
)

SCHEMA_VERSION = "1.0"

ALLOWED_CATEGORIES = frozenset(c1.RULES)
ALLOWED_STAGES = frozenset(c2.STAGES)


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


def fingerprint(value):
    return hashlib.sha256(
        canonical(value).encode("utf-8")
    ).hexdigest()


def file_sha256(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
    )


def source_data():
    """
    Revalidate the original Chapter 4 evidence
    rather than trusting arbitrary local JSON.
    """
    evidence, lessons, c1_summary = (
        c2.load_sources()
    )

    corrections = c2.make_library(
        evidence, lessons
    )

    stored = c2.read_json(
        c2.RESULTS / "approved_corrections.json"
    )
    c2_summary = c2.read_json(
        c2.RESULTS / "lesson_quality_results.json"
    )

    require(
        stored == corrections,
        "Approved library differs from evidence",
    )
    require(
        c2_summary["correction_library_sha256"]
        == c1.digest(corrections),
        "Approved correction hash mismatch",
    )
    require(
        c2_summary["authorization_changes"] == 0,
        "Source authorization changed",
    )
    require(
        c2_summary["held_out_evidence_used"] == 0,
        "Held-out evidence in source library",
    )

    return evidence, lessons, corrections, {
        "c1_source_sha256":
            c1_summary["source_sha256"],
        "c1_lesson_sha256":
            c1_summary["lesson_library_sha256"],
        "c2_correction_sha256":
            c2_summary["correction_library_sha256"],
    }


def group_support(evidence):
    """
    Each observation belongs to a model and a
    Chapter 4.1 lesson signature.

    Preserve donor provenance separately.
    """
    groups = defaultdict(list)

    for row in evidence:
        require(
            row["split"] == "train",
            "Held-out observation in donor evidence",
        )
        require(
            row["model"] in MODELS,
            "Unknown donor model",
        )
        require(
            row["category"] in ALLOWED_CATEGORIES,
            "Unknown failure category",
        )
        require(
            row["stage"] in ALLOWED_STAGES,
            "Unknown workflow stage",
        )

        signature = c1.lesson_signature(row)

        groups[(row["model"], signature)].append(row)

    return groups


def validate_donor_lesson(item, source_lesson, rows):
    """
    Strictly bind a donor record to observations
    produced by that donor model.
    """
    require(
        item["schema_version"] == SCHEMA_VERSION,
        "Invalid schema version",
    )
    require(
        item["donor_model"] in MODELS,
        "Invalid donor model",
    )
    require(
        item["source_split"] == "train",
        "Non-training donor lesson",
    )
    require(
        item["lesson_id"] ==
        source_lesson["lesson_id"],
        "Incorrect lesson linkage",
    )
    require(
        item["failure_category"] ==
        source_lesson["failure_category"],
        "Incorrect lesson category",
    )
    require(
        item["stage"] == source_lesson["stage"],
        "Incorrect lesson stage",
    )
    require(
        item["expected_tool"] ==
        source_lesson["expected_tool"],
        "Incorrect expected tool",
    )
    require(
        item["observations"] == len(rows)
        and len(rows) >= 1,
        "Incorrect donor support",
    )
    require(
        all(
            row["model"] == item["donor_model"]
            for row in rows
        ),
        "Mixed donor evidence",
    )
    require(
        item["supporting_scenario_ids"]
        == sorted({
            row["scenario_id"] for row in rows
        }),
        "Incorrect donor scenario IDs",
    )
    require(
        item["rule_text"]
        == c1.RULES[
            item["failure_category"]
        ]["rule"],
        "Untrusted rule text",
    )
    require(
        item["correction_text"]
        == c2.CORRECTION_TEMPLATES[
            item["failure_category"]
        ],
        "Untrusted correction text",
    )
    require(
        item["authorization_change"] is False,
        "Permission expansion detected",
    )

    return True


def make_donor_libraries(
    evidence, lessons, corrections
):
    support = group_support(evidence)

    correction_by_lesson = {
        item["lesson_id"]: item
        for item in corrections
    }

    libraries = {
        model: [] for model in MODELS
    }

    for lesson in lessons:
        signature = (
            lesson["lesson_id"].rsplit("_", 1)[-1]
        )

        correction = correction_by_lesson[
            lesson["lesson_id"]
        ]

        for donor in MODELS:
            rows = support.get(
                (donor, signature), []
            )

            # No observations means no evidence of
            # this specific failure for that donor.
            if not rows:
                continue

            item = {
                "schema_version": SCHEMA_VERSION,
                "donor_model": donor,
                "lesson_id": lesson["lesson_id"],
                "correction_id":
                    correction["correction_id"],
                "source_split": "train",
                "failure_category":
                    lesson["failure_category"],
                "stage": lesson["stage"],
                "proposed_tool":
                    lesson["proposed_tool"],
                "expected_tool":
                    lesson["expected_tool"],
                "rule_text":
                    lesson["violated_rule"],
                "correction_text":
                    correction["instruction"],
                "observations": len(rows),
                "supporting_scenario_ids":
                    sorted({
                        r["scenario_id"] for r in rows
                    }),
                "attack_observations": sum(
                    r["attack_present"] for r in rows
                ),
                "quality_score":
                    correction["quality_score"],
                "quality_score_type":
                    "shared_lesson_support_heuristic",
                "authorization_change": False,
                "provenance_type":
                    "donor_observed_failure",
            }

            validate_donor_lesson(
                item, lesson, rows
            )

            libraries[donor].append(item)

    for model in MODELS:
        libraries[model].sort(
            key=lambda x: x["lesson_id"]
        )

    return libraries


def make_transfer_dataset(libraries):
    """
    Construct directional transfer candidates.

    Donor != recipient: cross-model transfer.
    Donor == recipient: same-model control.

    A donor must have observed a failure
    supporting the correction.
    """
    transfers = []

    for donor in MODELS:
        for recipient in MODELS:
            direction = (
                "cross_model"
                if donor != recipient
                else "same_model"
            )

            for item in libraries[donor]:
                transfer_id = fingerprint({
                    "donor": donor,
                    "recipient": recipient,
                    "lesson_id": item["lesson_id"],
                })[:20]

                transfers.append({
                    "transfer_id":
                        f"transfer_{transfer_id}",
                    "schema_version": SCHEMA_VERSION,
                    "donor_model": donor,
                    "recipient_model": recipient,
                    "direction": direction,
                    "lesson_id": item["lesson_id"],
                    "correction_id":
                        item["correction_id"],
                    "failure_category":
                        item["failure_category"],
                    "stage": item["stage"],
                    "proposed_tool":
                        item["proposed_tool"],
                    "expected_tool":
                        item["expected_tool"],
                    "correction_text":
                        item["correction_text"],
                    "donor_observations":
                        item["observations"],
                    "donor_training_scenarios":
                        item["supporting_scenario_ids"],
                    "authorization_change": False,
                    "status": "candidate_transfer",
                })

    transfers.sort(
        key=lambda x: (
            x["recipient_model"],
            x["donor_model"],
            x["lesson_id"],
        )
    )

    return transfers


def validate_transfer_dataset(
    transfers, libraries
):
    """
    Reconstruct transfers from the donor
    libraries and check exact correspondence.
    """
    expected = make_transfer_dataset(
        libraries
    )

    require(
        transfers == expected,
        "Transfer dataset mismatch",
    )

    ids = [
        row["transfer_id"]
        for row in transfers
    ]

    require(
        len(ids) == len(set(ids)),
        "Duplicate transfer IDs",
    )

    for row in transfers:
        require(
            row["donor_model"] in MODELS,
            "Unknown donor",
        )
        require(
            row["recipient_model"] in MODELS,
            "Unknown recipient",
        )
        require(
            row["authorization_change"] is False,
            "Authorization expansion",
        )
        require(
            row["status"] == "candidate_transfer",
            "Unvalidated transfer status",
        )
        require(
            row["correction_text"]
            == c2.CORRECTION_TEMPLATES[
                row["failure_category"]
            ],
            "Untrusted transfer correction",
        )

    return True


def make_summary(
    evidence, lessons, corrections,
    libraries, transfers
):
    return {
        "module": 5,
        "chapter": "5.1",
        "training_denial_observations":
            len(evidence),
        "source_candidate_lessons":
            len(lessons),
        "source_validated_corrections":
            len(corrections),
        "donor_libraries": {
            model: {
                "supported_lessons":
                    len(libraries[model]),
                "support_observations": sum(
                    row["observations"]
                    for row in libraries[model]
                ),
                "category_counts": dict(sorted(
                    Counter(
                        row["failure_category"]
                        for row in libraries[model]
                    ).items()
                )),
            }
            for model in MODELS
        },
        "transfer_candidates":
            len(transfers),
        "cross_model_candidates": sum(
            row["direction"] == "cross_model"
            for row in transfers
        ),
        "same_model_candidates": sum(
            row["direction"] == "same_model"
            for row in transfers
        ),
        "authorization_changes": 0,
        "held_out_evidence_used": 0,
        "behavioral_transfer_tested": False,
    }


def render_findings(summary):
    lines = [
        "# Guardian AI — Module 5, Chapter 5.1",
        "",
        "## Cross-Agent Lesson Provenance",
        "",
        "This chapter builds model-specific evidence "
        "support for existing controller-authored "
        "safety corrections.",
        "",
        "## Donor libraries",
        "",
        "| Donor | Supported lessons | "
        "Failure observations |",
        "|---|---:|---:|",
    ]

    for model, stats in summary[
        "donor_libraries"
    ].items():
        lines.append(
            f"| {model} | "
            f"{stats['supported_lessons']} | "
            f"{stats['support_observations']} |"
        )

    lines.extend([
        "",
        "## Transfer dataset",
        "",
        f"- Total directional candidates: "
        f"{summary['transfer_candidates']}",
        f"- Cross-model candidates: "
        f"{summary['cross_model_candidates']}",
        f"- Same-model control candidates: "
        f"{summary['same_model_candidates']}",
        "",
        "## Security guarantees in this dataset",
        "",
        "- Donor provenance is established using "
        "training failure observations.",
        "- Lesson text is taken from trusted "
        "controller-owned templates.",
        "- No held-out episodes contribute "
        "to lesson construction.",
        "- No transfer changes task authorization.",
        "- Raw attack text is not embedded in "
        "transfer records.",
        "",
        "## Research limitations",
        "",
        "- Donor support is not evidence of "
        "successful transfer.",
        "- A controller-authored correction is "
        "not a novel rule invented by its donor.",
        "- Several lessons may be supported by "
        "both models.",
        "- Existing scenario splits reuse some "
        "attack templates and text patterns.",
        "- The dataset is not independent of "
        "the prior lesson-selection process.",
        "",
        "## Next chapter",
        "",
        "Chapter 5.2 will implement a donor-aware "
        "retriever that prevents recipient evidence "
        "from entering donor-specific lesson "
        "selection.",
        "",
    ])

    return "\n".join(lines)


def run_build(output_dir=RESULTS):
    evidence, lessons, corrections, hashes = (
        source_data()
    )

    libraries = make_donor_libraries(
        evidence, lessons, corrections
    )

    transfers = make_transfer_dataset(
        libraries
    )

    validate_transfer_dataset(
        transfers, libraries
    )

    summary = make_summary(
        evidence, lessons, corrections,
        libraries, transfers
    )

    # Sum across donor libraries counts each
    # failure observation exactly once.
    require(
        sum(
            row["support_observations"]
            for row in summary[
                "donor_libraries"
            ].values()
        ) == len(evidence),
        "Donor evidence accounting mismatch",
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    libraries_path = (
        output_dir / "donor_libraries.json"
    )

    transfers_path = (
        output_dir / "transfer_dataset.json"
    )

    summary_path = (
        output_dir / "m5_1_summary.json"
    )

    manifest_path = (
        output_dir / "manifest.json"
    )

    write_json(libraries_path, libraries)
    write_json(transfers_path, transfers)
    write_json(summary_path, summary)

    manifest = {
        "module": 5,
        "chapter": "5.1",
        "source_fingerprints": hashes,
        "donor_libraries_sha256":
            fingerprint(libraries),
        "transfer_dataset_sha256":
            fingerprint(transfers),
        "summary_sha256":
            fingerprint(summary),
    }

    write_json(manifest_path, manifest)

    findings_path = (
        HERE / "M5_C1_FINDINGS.md"
    )

    findings_path.write_text(
        render_findings(summary),
        encoding="utf-8",
    )

    print("\nFINAL MODULE 5 C1 SUMMARY")
    print(json.dumps(summary, indent=2))

    print("\nDonor libraries:", libraries_path)
    print("Transfer dataset:", transfers_path)
    print("Manifest:", manifest_path)
    print("Findings:", findings_path)

    return {
        "summary": summary,
        "libraries": libraries,
        "transfers": transfers,
        "manifest": manifest,
    }


if __name__ == "__main__":
    run_build()
