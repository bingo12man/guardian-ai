
"""
Guardian AI — Module 8.2 detector evaluation.

The classifier is label-blind.
Ground truth is loaded only by the evaluator.

Small synthetic fixture benchmark.
Not an independent real-world accuracy claim.
"""

import json
from collections import Counter
from pathlib import Path

import guardian_threat_detector as detector

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

CATEGORIES = detector.CATEGORIES


def require(value, reason):
    if not value:
        raise ValueError(reason)


def safe_div(numerator, denominator):
    return (
        numerator / denominator
        if denominator else 0.0
    )


def metrics(tp, fp, fn):
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(
        2 * precision * recall,
        precision + recall,
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": tp + fn,
    }


def evaluate(predictions, ground_truth):
    ids = [row["snapshot_id"] for row in predictions]

    require(
        len(ids) == len(set(ids))
        and set(ids) == set(ground_truth),
        "Prediction/ground truth identity mismatch",
    )

    require(
        set(ground_truth.values()).issubset(CATEGORIES),
        "Unknown ground truth category",
    )

    confusion = {
        true: {
            predicted: 0 for predicted in CATEGORIES
        }
        for true in CATEGORIES
    }

    correct = 0
    binary_tp = binary_fp = binary_fn = binary_tn = 0

    for row in predictions:
        expected = ground_truth[row["snapshot_id"]]
        predicted = row["predicted_category"]

        require(
            predicted in CATEGORIES,
            "Unknown predicted category",
        )

        confusion[expected][predicted] += 1
        correct += int(expected == predicted)

        real_attack = expected != "benign"
        predicted_attack = predicted != "benign"

        if real_attack and predicted_attack:
            binary_tp += 1
        elif not real_attack and predicted_attack:
            binary_fp += 1
        elif real_attack and not predicted_attack:
            binary_fn += 1
        else:
            binary_tn += 1

    per_category = {}

    for category in CATEGORIES:
        tp = confusion[category][category]
        fp = sum(
            confusion[other][category]
            for other in CATEGORIES
            if other != category
        )
        fn = sum(
            confusion[category][other]
            for other in CATEGORIES
            if other != category
        )
        per_category[category] = metrics(tp, fp, fn)

    count = len(predictions)

    return {
        "examples": count,
        "correct": correct,
        "accuracy": round(safe_div(correct, count), 4),
        "binary_attack_detection": {
            "true_positive": binary_tp,
            "false_positive": binary_fp,
            "false_negative": binary_fn,
            "true_negative": binary_tn,
            **metrics(binary_tp, binary_fp, binary_fn),
            "specificity": round(
                safe_div(
                    binary_tn,
                    binary_tn + binary_fp,
                ), 4
            ),
        },
        "per_category": per_category,
        "macro_f1": round(
            sum(
                x["f1"] for x in per_category.values()
            ) / len(CATEGORIES),
            4,
        ),
        "confusion_matrix": confusion,
    }


def load_evaluation_labels():
    """
    The evaluator, not the detector, can read
    the stored fixture labels.
    """
    labels = detector.c1.read_json(
        detector.LABELS_PATH
    )
    manifest = detector.c1.read_json(
        detector.MANIFEST_PATH
    )

    require(
        labels == detector.c1.make_ground_truth(),
        "Ground-truth fixture mismatch",
    )
    require(
        detector.c1.fingerprint(labels)
        == manifest["ground_truth_sha256"],
        "Ground-truth manifest mismatch",
    )

    return labels


def run_benchmark(
    output_dir=RESULTS,
    findings_dir=HERE,
):
    archive = detector.load_verified_archive()

    # Predict BEFORE accessing reference labels.
    predictions = detector.detect_archive(archive)
    detector.verify_predictions(predictions, archive)

    ground_truth = load_evaluation_labels()
    evaluation = evaluate(predictions, ground_truth)

    predicted_distribution = dict(sorted(
        Counter(
            row["predicted_category"]
            for row in predictions
        ).items()
    ))

    summary = {
        "module": 8,
        "chapter": "8.2",
        "mode": "offline_rule_based_detection",
        "evaluated_snapshots": len(predictions),
        "predicted_distribution":
            predicted_distribution,
        "accuracy": evaluation["accuracy"],
        "macro_f1": evaluation["macro_f1"],
        "binary_attack_detection":
            evaluation["binary_attack_detection"],
        "per_category": evaluation["per_category"],
        "rule_matches": sum(
            row["rule_matches"]
            for row in predictions
        ),
        "unmatched_snapshots": sum(
            row["rule_matches"] == 0
            for row in predictions
        ),
        "prediction_integrity_verified": True,
        "ground_truth_separate_from_detector": True,
        "model_calls": 0,
        "network_requests": 0,
        "instructions_executed": 0,
        "authorization_changes": 0,
        "behavioral_safety_improvement_tested": False,
        "generalization_demonstrated": False,
    }

    output_dir = Path(output_dir)
    findings_dir = Path(findings_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    findings_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = (
        output_dir / "threat_predictions.json"
    )
    evaluation_path = (
        output_dir / "detection_evaluation.json"
    )
    summary_path = (
        output_dir / "m8_2_summary.json"
    )
    manifest_path = output_dir / "manifest.json"

    detector.c1.write_json(
        predictions_path, predictions
    )
    detector.c1.write_json(
        evaluation_path, evaluation
    )
    detector.c1.write_json(
        summary_path, summary
    )

    manifest = {
        "module": 8,
        "chapter": "8.2",
        "archive_sha256":
            detector.c1.fingerprint(archive),
        "predictions_sha256":
            detector.c1.fingerprint(predictions),
        "evaluation_sha256":
            detector.c1.fingerprint(evaluation),
        "summary_sha256":
            detector.c1.fingerprint(summary),
        "detector_rule_sha256":
            detector.c1.fingerprint(
                detector.RULES
            ),
    }

    detector.c1.write_json(
        manifest_path, manifest
    )

    findings_path = (
        findings_dir / "M8_C2_FINDINGS.md"
    )

    lines = [
        "# Guardian AI — Module 8.2 Findings",
        "",
        "## Offline threat classification",
        "",
        f"- Synthetic examples: {len(predictions)}",
        f"- Multiclass accuracy: "
        f"{evaluation['accuracy']:.4f}",
        f"- Macro F1: {evaluation['macro_f1']:.4f}",
        f"- Binary attack F1: "
        f"{evaluation['binary_attack_detection']['f1']:.4f}",
        "",
        "## Method",
        "",
        "Controller-authored regular-expression rules",
        "operate only on verified offline fixture text.",
        "Ground-truth labels are read after prediction.",
        "",
        "## Scientific limitations",
        "",
        "- Only 12 synthetic examples.",
        "- Detector rules are tailored to known patterns.",
        "- No independent unseen-template test set.",
        "- No real-world false-positive estimate.",
        "- Encoded strings are not decoded or executed.",
        "- A benign classification is not proof of safety.",
        "- No LLM behavioral improvement has been tested.",
        "",
        "## Chapter 8.3 handoff",
        "",
        "Use rule IDs, evidence offsets and separately",
        "validated labels to propose controller-reviewed",
        "lessons without treating content as instructions.",
    ]

    findings_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\nFINAL MODULE 8 C2 SUMMARY")
    print(json.dumps(summary, indent=2))

    print("\nPredictions:", predictions_path)
    print("Evaluation:", evaluation_path)
    print("Manifest:", manifest_path)
    print("Findings:", findings_path)

    return {
        "summary": summary,
        "predictions": predictions,
        "evaluation": evaluation,
        "manifest": manifest,
    }


if __name__ == "__main__":
    run_benchmark()
