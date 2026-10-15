
"""
Module 5 Chapter 5.2 donor-aware retrieval benchmark.

Training-only diagnostic.
No LLM calls or held-out lesson construction.

Measures:
- Donor-specific exact matches
- Donor-specific misses
- Same-model and cross-model lookup
- Verification of all directional transfers
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

sys.path.insert(
    0, str(HERE.parent / "chapter_01")
)

import m5_1_transfer_dataset as c1

from guardian_transfer_memory import (
    DonorAwareMemory,
    TransferRequest,
    load_verified_memory,
    transfer_feedback,
)


CATEGORY_TO_DECISION = {
    category: decision
    for decision, category
    in c1.c1.DENIAL_CATEGORIES.items()
}


def make_request(evidence, donor, recipient):
    return TransferRequest(
        donor_model=donor,
        recipient_model=recipient,
        decision=CATEGORY_TO_DECISION[
            evidence["category"]
        ],
        stage=evidence["stage"],
        proposed_tool=evidence["proposed_tool"],
        expected_tool=evidence["expected_tool"],
    )


def assess(memory, evidence, libraries):
    supported = {
        (
            donor,
            lesson["failure_category"],
            lesson["stage"],
            lesson["proposed_tool"],
            lesson["expected_tool"],
        )
        for donor, lessons in libraries.items()
        for lesson in lessons
    }

    records = []

    for observation in evidence:
        if observation["split"] != "train":
            raise ValueError("Held-out retrieval diagnostic")

        for donor in c1.MODELS:
            for recipient in c1.MODELS:
                request = make_request(
                    observation, donor, recipient
                )
                correction = memory.retrieve(request)

                signature = (
                    donor,
                    observation["category"],
                    observation["stage"],
                    observation["proposed_tool"],
                    observation["expected_tool"],
                )

                expected_hit = signature in supported
                actual_hit = correction is not None

                if actual_hit != expected_hit:
                    raise AssertionError(
                        "Donor-specific lookup incorrect"
                    )

                if correction is not None:
                    if (
                        correction.donor_model != donor
                        or correction.recipient_model != recipient
                    ):
                        raise AssertionError(
                            "Cross-donor contamination"
                        )

                    if not transfer_feedback(correction):
                        raise AssertionError(
                            "Empty correction feedback"
                        )

                records.append({
                    "source_model": observation["model"],
                    "donor_model": donor,
                    "recipient_model": recipient,
                    "direction": (
                        "same_model" if donor == recipient
                        else "cross_model"
                    ),
                    "category": observation["category"],
                    "matched": actual_hit,
                    "expected_hit": expected_hit,
                    "transfer_id": (
                        correction.transfer_id
                        if correction else None
                    ),
                })

    return records


def summarize(records, memory):
    by_direction = {}

    for direction in ("same_model", "cross_model"):
        rows = [
            r for r in records
            if r["direction"] == direction
        ]

        by_direction[direction] = {
            "queries": len(rows),
            "hits": sum(r["matched"] for r in rows),
            "misses": sum(not r["matched"] for r in rows),
            "incorrect_matches": sum(
                r["matched"] != r["expected_hit"]
                for r in rows
            ),
        }

    by_donor = {}

    for donor in c1.MODELS:
        rows = [
            r for r in records
            if r["donor_model"] == donor
        ]

        by_donor[donor] = {
            "queries": len(rows),
            "hits": sum(r["matched"] for r in rows),
            "misses": sum(not r["matched"] for r in rows),
        }

    return {
        "transfer_entries": memory.size(),
        "training_observations": len(records) // 4,
        "retrieval_queries": len(records),
        "retrieval_hits": sum(
            r["matched"] for r in records
        ),
        "retrieval_misses": sum(
            not r["matched"] for r in records
        ),
        "incorrect_matches": sum(
            r["matched"] != r["expected_hit"]
            for r in records
        ),
        "by_direction": by_direction,
        "by_donor": by_donor,
        "held_out_queries": 0,
        "authorization_changes": 0,
        "behavioral_transfer_tested": False,
    }


def run_benchmark(output_dir=RESULTS):
    memory = load_verified_memory()

    evidence, lessons, corrections, _ = (
        c1.source_data()
    )
    libraries = c1.make_donor_libraries(
        evidence, lessons, corrections
    )

    transfers = c1.make_transfer_dataset(
        libraries
    )

    if memory.size() != len(transfers):
        raise AssertionError(
            "Transfer index size mismatch"
        )

    # Check every transfer entry using its exact
    # donor, recipient and lesson signature.
    for item in transfers:
        request = TransferRequest(
            donor_model=item["donor_model"],
            recipient_model=item["recipient_model"],
            decision=CATEGORY_TO_DECISION[
                item["failure_category"]
            ],
            stage=item["stage"],
            proposed_tool=item["proposed_tool"],
            expected_tool=item["expected_tool"],
        )

        found = memory.retrieve(request)

        if (
            found is None
            or found.transfer_id != item["transfer_id"]
        ):
            raise AssertionError(
                "Directional transfer record missing"
            )

    records = assess(memory, evidence, libraries)
    summary = summarize(records, memory)

    if summary["incorrect_matches"] != 0:
        raise AssertionError(
            "Donor retrieval invariant failed"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True, exist_ok=True
    )

    report = {
        "module": 5,
        "chapter": "5.2",
        "mode": "train_donor_retrieval_diagnostic",
        "summary": summary,
        "records": records,
        "limitations": [
            "Queries use known training denial signatures",
            "No recipient model was actually prompted",
            "Donor attribution refers to failure evidence",
            "Lessons remain controller-authored templates",
            "No evidence of behavioral transfer yet",
        ],
    }

    result_path = (
        output_dir / "c2_retrieval_results.json"
    )

    result_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    checksum = hashlib.sha256(
        result_path.read_bytes()
    ).hexdigest()

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({
            "module": 5,
            "chapter": "5.2",
            "results_sha256": checksum,
            "summary": summary,
        }, indent=2),
        encoding="utf-8",
    )

    print("\nFINAL MODULE 5 C2 SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nSaved:", result_path)

    return report


if __name__ == "__main__":
    run_benchmark()
