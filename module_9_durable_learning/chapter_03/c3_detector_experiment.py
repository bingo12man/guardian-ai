
"""
Guardian AI — Module 9.3 controlled comparison.

Compare frozen v1 and candidate v2 across:
  original fixtures
  inspected development decoys
  fresh synthetic assessment

Register versioned results in a new store.

No candidate activation or Guardian updates.
"""

import json
import sys
from pathlib import Path

import guardian_detector_v2 as v2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"

sys.path.insert(
    0, str(ROOT / "module_8_observatory/chapter_02")
)
sys.path.insert(
    0, str(ROOT / "module_9_durable_learning/chapter_01")
)

import c2_detection_benchmark as metrics
import guardian_experiment_registry as registry


def require(condition, message):
    if not condition:
        raise ValueError(message)


def evaluate_one(archive, labels, detector_fn):
    # Explicitly run classification before evaluation.
    predictions = detector_fn(archive)
    result = metrics.evaluate(predictions, labels)
    return predictions, result


def make_candidate_record(
    dataset_name, archive, evaluation,
    baseline_handoff,
):
    """
    Result identity references actual dataset
    contents rather than a synthetic name hash.
    """
    base = {
        "schema_version": "1.0",
        "kind": "candidate_detector",
        "detector_version": v2.VERSION,
        "detector_sha256": v2.detector_hash(),
        "dataset_sha256": v2.digest(archive),
        "configuration": {
            "dataset": dataset_name,
            "evaluation_mode": "offline_comparison",
            "candidate_rules_frozen": True,
            "model_inference": False,
        },
        "result": evaluation,
        "status": "CANDIDATE_PENDING",
        "source_module": 9,
        "source_evidence_sha256":
            baseline_handoff["final_sha256"],
    }
    base["experiment_id"] = registry.experiment_id(base)
    registry.validate_record(base)
    return base


def run_benchmark(
    output_dir=RESULTS,
    findings_dir=HERE,
):
    output_dir = Path(output_dir)
    findings_dir = Path(findings_dir)

    storage = output_dir / "candidate_experiment_store"

    require(
        not storage.exists(),
        "Existing experiment store: use a new output directory",
    )

    baseline = registry.validate_module_8_handoff()
    datasets = v2.validated_datasets()
    store = registry.ExperimentRegistry(storage)

    scores = {}
    candidate_records = []

    for name, dataset in datasets.items():
        archive = dataset["archive"]
        labels = dataset["labels"]

        v1_predictions, v1_eval = evaluate_one(
            archive, labels,
            v2.v1.detect_archive,
        )
        v2_predictions, v2_eval = evaluate_one(
            archive, labels,
            v2.detect_archive,
        )

        scores[name] = {
            "dataset_role": dataset["role"],
            "examples": len(archive),
            "actual_dataset_sha256": v2.digest(archive),
            "v1": {
                "accuracy": v1_eval["accuracy"],
                "macro_f1": v1_eval["macro_f1"],
                "attack_precision":
                    v1_eval["binary_attack_detection"][
                        "precision"
                    ],
                "attack_recall":
                    v1_eval["binary_attack_detection"][
                        "recall"
                    ],
                "false_positives":
                    v1_eval["binary_attack_detection"][
                        "false_positive"
                    ],
                "false_negatives":
                    v1_eval["binary_attack_detection"][
                        "false_negative"
                    ],
            },
            "v2": {
                "accuracy": v2_eval["accuracy"],
                "macro_f1": v2_eval["macro_f1"],
                "attack_precision":
                    v2_eval["binary_attack_detection"][
                        "precision"
                    ],
                "attack_recall":
                    v2_eval["binary_attack_detection"][
                        "recall"
                    ],
                "false_positives":
                    v2_eval["binary_attack_detection"][
                        "false_positive"
                    ],
                "false_negatives":
                    v2_eval["binary_attack_detection"][
                        "false_negative"
                    ],
            },
            "delta_accuracy_pp": round(
                100 * (
                    v2_eval["accuracy"]
                    - v1_eval["accuracy"]
                ), 2
            ),
            "delta_macro_f1": round(
                v2_eval["macro_f1"] - v1_eval["macro_f1"],
                4,
            ),
        }

        # Register complete, recomputable v2
        # results without changing v1 records.
        record = make_candidate_record(
            name, archive, v2_eval, baseline
        )
        require(
            store.register(record)["accepted"],
            "Candidate registration failed",
        )
        candidate_records.append(record)

        # Store individual prediction outcomes for
        # future reproducibility and error analysis.
        v2.v1.c1.write_json(
            output_dir / f"{name}_predictions.json",
            {
                "v1": v1_predictions,
                "v2": v2_predictions,
            },
        )

    reopened = registry.ExperimentRegistry(storage)
    require(
        reopened.verify()
        and reopened.list_records() == candidate_records,
        "Persisted experiments failed verification",
    )

    summary = {
        "module": 9,
        "chapter": "9.3",
        "mode": "versioned_detector_comparison",
        "baseline_detector_version":
            "m8_rule_detector_v1",
        "candidate_detector_version": v2.VERSION,
        "baseline_rule_sha256": baseline["rule_sha256"],
        "candidate_rule_sha256": v2.detector_hash(),
        "evaluated_datasets": len(datasets),
        "evaluated_examples": sum(
            len(d["archive"])
            for d in datasets.values()
        ),
        "scores": scores,
        "candidate_experiments_registered":
            len(candidate_records),
        "registry_verified_after_restart": True,
        "candidate_status": "CANDIDATE_PENDING",
        "candidate_promoted": False,
        "baseline_replaced": False,
        "lesson_approvals": 0,
        "permissions_changed": 0,
        "network_requests": 0,
        "model_calls": 0,
        "real_world_generalization_established": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    findings_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "m9_3_summary.json"
    manifest_path = output_dir / "manifest.json"

    v2.v1.c1.write_json(summary_path, summary)
    v2.v1.c1.write_json(
        manifest_path,
        {
            "module": 9,
            "chapter": "9.3",
            "summary_sha256": v2.digest(summary),
            "candidate_rule_sha256": v2.detector_hash(),
            "candidate_experiment_ids": [
                x["experiment_id"]
                for x in candidate_records
            ],
            "audit_head": reopened.audit_head(),
        },
    )

    lines = [
        "# Guardian AI — Module 9.3 Findings",
        "",
        "## Detector comparison",
        "",
        "| Dataset | v1 accuracy | v2 accuracy | Delta (pp) |",
        "|---|---:|---:|---:|",
    ]
    for name, row in scores.items():
        lines.append(
            f"| {name} | "
            f"{row['v1']['accuracy']:.4f} | "
            f"{row['v2']['accuracy']:.4f} | "
            f"{row['delta_accuracy_pp']:+.2f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "The previously inspected 24-case robustness",
        "dataset is development evidence, not a blind test.",
        "The separate 12-case synthetic assessment",
        "provides a preliminary additional comparison.",
        "",
        "## Limitations",
        "",
        "- Heuristic detector, not learned model",
        "- All examples are synthetic",
        "- The new suite is still hand-authored",
        "- No real-world prevalence estimate",
        "- No live agent behavioral evaluation",
        "- Future detector revisions need a new",
        "  independently prepared test suite",
        "- Candidate remains unpromoted",
        "",
    ]

    (findings_dir / "M9_C3_FINDINGS.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\nFINAL MODULE 9 C3 SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nSaved:", summary_path)
    print("Registry:", storage)

    return {
        "summary": summary,
        "records": candidate_records,
    }


if __name__ == "__main__":
    run_benchmark()
