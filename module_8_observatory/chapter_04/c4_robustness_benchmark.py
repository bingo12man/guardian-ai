
"""
Guardian AI — Chapter 8.4 robustness benchmark.

Frozen detector + new synthetic test suite.
Evaluation labels are read only after prediction.

No automatic promotion of candidate lessons.
"""

import json
import sys
from pathlib import Path

import guardian_decoy_monitor as monitor

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

sys.path.insert(
    0, str(HERE.parent / "chapter_02")
)
import c2_detection_benchmark as evaluator


def run_benchmark(
    output_dir=RESULTS,
    findings_dir=HERE,
):
    rule_hash = monitor.load_frozen_detector()

    decoys = monitor.make_decoys()
    monitor.validate_decoys(decoys)

    # Detector does not receive labels.
    predictions = monitor.detect_decoys(decoys)
    monitored = monitor.monitor_findings(
        decoys, predictions
    )
    monitor.validate_monitor_output(
        decoys, predictions, monitored
    )

    # Evaluation starts after inference.
    labels = monitor.make_labels()
    evaluation = evaluator.evaluate(
        predictions, labels
    )

    outcomes = []
    false_positive_ids = []
    false_negative_ids = []
    wrong_family_ids = []

    for prediction in predictions:
        identifier = prediction["snapshot_id"]
        actual = labels[identifier]
        predicted = prediction["predicted_category"]

        correct = actual == predicted

        outcomes.append({
            "snapshot_id": identifier,
            "ground_truth": actual,
            "predicted": predicted,
            "correct": correct,
        })

        if actual == "benign" and predicted != "benign":
            false_positive_ids.append(identifier)
        elif actual != "benign" and predicted == "benign":
            false_negative_ids.append(identifier)
        elif (
            actual != "benign"
            and predicted != "benign"
            and actual != predicted
        ):
            wrong_family_ids.append(identifier)

    binary = evaluation["binary_attack_detection"]

    monitor.require(
        binary["false_positive"] == len(false_positive_ids)
        and binary["false_negative"]
        == len(false_negative_ids),
        "Detection error accounting mismatch",
    )

    # Any attack prediction creates a candidate.
    # Some may be false positives in ground truth.
    monitor.require(
        monitored["candidate_count"]
        == monitored["detected_count"],
        "Candidate/alert accounting mismatch",
    )

    summary = {
        "module": 8,
        "chapter": "8.4",
        "mode": "frozen_rule_synthetic_robustness",
        "decoys": len(decoys),
        "attack_decoys": sum(
            category != "benign"
            for category in labels.values()
        ),
        "hard_benign_decoys": sum(
            category == "benign"
            for category in labels.values()
        ),
        "ground_truth_distribution":
            monitor.category_counts(labels),
        "multiclass_accuracy": evaluation["accuracy"],
        "macro_f1": evaluation["macro_f1"],
        "binary_detection": binary,
        "false_positive_ids": false_positive_ids,
        "false_negative_ids": false_negative_ids,
        "wrong_attack_family_ids": wrong_family_ids,
        "alerts_generated": len(monitored["alerts"]),
        "pending_candidates":
            monitored["candidate_count"],
        "unique_pending_templates":
            len(monitored["groups"]),
        "false_candidate_alerts":
            len(false_positive_ids),
        "candidate_publications": 0,
        "detector_rules_frozen": True,
        "frozen_rule_sha256": rule_hash,
        "network_requests": 0,
        "model_calls": 0,
        "instructions_executed": 0,
        "authorization_changes": 0,
        "generalization_demonstrated": False,
        "real_world_performance_established": False,
    }

    output_dir = Path(output_dir)
    findings_dir = Path(findings_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    findings_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "decoys": output_dir / "synthetic_decoys.json",
        "labels": output_dir / "decoy_labels.json",
        "predictions":
            output_dir / "decoy_predictions.json",
        "outcomes":
            output_dir / "detection_outcomes.json",
        "evaluation":
            output_dir / "robustness_evaluation.json",
        "alerts":
            output_dir / "threat_alerts.json",
        "candidates":
            output_dir / "pending_decoy_candidates.json",
        "groups":
            output_dir / "pending_decoy_groups.json",
        "summary":
            output_dir / "m8_4_summary.json",
        "manifest": output_dir / "manifest.json",
    }

    payloads = {
        "decoys": decoys,
        "labels": labels,
        "predictions": predictions,
        "outcomes": outcomes,
        "evaluation": evaluation,
        "alerts": monitored["alerts"],
        "candidates": monitored["candidates"],
        "groups": monitored["groups"],
        "summary": summary,
    }

    for name, value in payloads.items():
        monitor.write_json(files[name], value)

    manifest = {
        "module": 8,
        "chapter": "8.4",
        "frozen_rule_sha256": rule_hash,
        "artifacts_sha256": {
            name: monitor.digest(value)
            for name, value in payloads.items()
        },
    }

    monitor.write_json(
        files["manifest"], manifest
    )

    findings = [
        "# Guardian AI — Chapter 8.4 Findings",
        "",
        "## Synthetic robustness evaluation",
        "",
        f"- Decoys: {len(decoys)}",
        f"- Attack variants: "
        f"{summary['attack_decoys']}",
        f"- Hard benign negatives: "
        f"{summary['hard_benign_decoys']}",
        f"- Multiclass accuracy: "
        f"{evaluation['accuracy']:.4f}",
        f"- Macro F1: "
        f"{evaluation['macro_f1']:.4f}",
        f"- Binary precision: "
        f"{binary['precision']:.4f}",
        f"- Binary recall: "
        f"{binary['recall']:.4f}",
        f"- False-positive alerts: "
        f"{len(false_positive_ids)}",
        f"- Missed attacks: "
        f"{len(false_negative_ids)}",
        "",
        "## Interpretation",
        "",
        "The detector was frozen before evaluation.",
        "Hard negatives include benign quotations,",
        "security explanations and encoded examples.",
        "Any false detections remain visible in the",
        "candidate-review queue.",
        "",
        "## Limitations",
        "",
        "- Synthetic hand-authored examples only",
        "- No independent real-world collection",
        "- One fixed robustness suite",
        "- Patterns may still overlap training design",
        "- Labels reflect author-defined intent",
        "- No LLM or downstream behavior evaluated",
        "- No autonomous lesson publication",
        "",
        "## Chapter 8.5",
        "",
        "Consolidate the evidence and prepare",
        "the Module 9 durability contract.",
    ]

    findings_path = (
        findings_dir / "M8_C4_FINDINGS.md"
    )
    findings_path.write_text(
        "\n".join(findings) + "\n",
        encoding="utf-8",
    )

    print("\nFINAL MODULE 8 C4 SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nSaved:", files["summary"])
    print("Findings:", findings_path)

    return {
        "summary": summary,
        "predictions": predictions,
        "evaluation": evaluation,
        "monitored": monitored,
        "manifest": manifest,
    }


if __name__ == "__main__":
    run_benchmark()
