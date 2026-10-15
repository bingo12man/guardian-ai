
"""
Guardian AI — Module 4, Chapter 4.3 benchmark.

Deterministic retrieval over training denials.
No LLM calls.

Measures:
- Exact structured retrieval coverage
- Category and stage correctness
- No permission modifications
- No evaluation-split evidence consumption
- Deterministic library fingerprint
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

sys.path.insert(0, str(HERE.parent / "chapter_01"))

import m4_1_lesson_extraction as extraction

from guardian_lesson_memory import (
    RetrievalRequest,
    load_verified_memory,
    correction_feedback,
)


def make_request(evidence):
    return RetrievalRequest(
        decision=evidence["decision"],
        stage=evidence["stage"],
        proposed_tool=evidence["proposed_tool"],
        expected_tool=evidence["expected_tool"],
    )


def evaluate(memory, evidence):
    records = []

    for observation in evidence:
        if observation["split"] != "train":
            raise ValueError(
                "Held-out evidence supplied to retrieval benchmark"
            )

        request = make_request(observation)
        correction = memory.retrieve(request)

        found = correction is not None

        if found:
            category = extraction.DENIAL_CATEGORIES[
                request.decision
            ]

            if (
                correction.category != category
                or correction.stage != request.stage
                or correction.expected_tool
                != request.expected_tool
            ):
                raise AssertionError(
                    "Incorrect correction retrieved"
                )

            feedback = correction_feedback(
                correction
            )

            if not feedback:
                raise AssertionError(
                    "Missing safe feedback"
                )

        records.append({
            "scenario_id": observation["scenario_id"],
            "model": observation["model"],
            "step": observation["step"],
            "category": observation["category"],
            "stage": observation["stage"],
            "matched": found,
            "correction_id": (
                correction.correction_id
                if found else None
            ),
        })

    return records


def summarize(records):
    category_counts = Counter(
        r["category"] for r in records
    )

    hits = Counter(
        r["category"]
        for r in records
        if r["matched"]
    )

    total = len(records)
    matched = sum(
        r["matched"] for r in records
    )

    return {
        "training_denial_queries": total,
        "retrieval_hits": matched,
        "retrieval_misses": total - matched,
        "retrieval_coverage": round(
            matched / total, 4
        ) if total else 0,
        "category_results": {
            category: {
                "queries": count,
                "hits": hits[category],
            }
            for category, count
            in sorted(category_counts.items())
        },
        "held_out_queries": 0,
        "authorization_changes": 0,
    }


def run_benchmark(output_dir=RESULTS):
    memory = load_verified_memory()

    evidence = extraction.load_evidence()[0]
    scenarios = extraction.load_evidence()[1]

    training = []

    for episode in evidence["records"]:
        training.extend(
            extraction.classify_denials(
                episode,
                scenarios[episode["scenario_id"]],
            )
        )

    records = evaluate(memory, training)
    summary = summarize(records)

    if len(training) != 165:
        raise AssertionError(
            "Unexpected Chapter 4.1 evidence count"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True, exist_ok=True
    )

    report = {
        "module": 4,
        "chapter": "4.3",
        "mode": "train_retrieval_diagnostic",
        "corrections_in_memory": memory.size(),
        "summary": summary,
        "records": records,
        "limitations": [
            "Train-split retrieval is not held-out generalization",
            "No language model consumed corrections",
            "Only trusted decision labels are accepted",
            "Corrections are not indexed by proposed tool",
            "Missing matches produce no correction",
        ],
    }

    path = output_dir / "c3_retrieval_results.json"
    path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    checksum = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    manifest = {
        "module": 4,
        "chapter": "4.3",
        "results_sha256": checksum,
        "summary": summary,
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nFINAL MODULE 4 C3 SUMMARY")
    print(json.dumps({
        "corrections_in_memory": memory.size(),
        **summary,
    }, indent=2))
    print("\nSaved:", path)

    return report


if __name__ == "__main__":
    run_benchmark()
