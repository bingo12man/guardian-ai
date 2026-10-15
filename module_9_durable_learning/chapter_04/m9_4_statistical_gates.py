
"""
Guardian AI — Module 9, Chapter 9.4

Paired statistical evaluation of frozen v1 and
candidate v2 with conservative release gates.

No model calls, network access, lesson publication,
Guardian capability issuance or permission changes.

The fresh synthetic suite is small and authored
within the project. Statistical intervals do not
turn it into real-world generalization evidence.
"""

import hashlib
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
C3 = HERE.parent / "chapter_03"
RESULTS = HERE / "results"

sys.path.insert(0, str(C3))
sys.path.insert(
    0, str(ROOT / "module_9_durable_learning/chapter_01")
)

import guardian_detector_v2 as v2
import c3_detector_experiment as experiment
import guardian_experiment_registry as registry

DATASETS = ("original", "development", "fresh")
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 9042026


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(
        value, sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def digest(value):
    return hashlib.sha256(
        canonical(value).encode("utf-8")
    ).hexdigest()


def file_hash(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
    )


def wilson_interval(successes, total, z=1.96):
    """Approximate two-sided 95% Wilson interval."""
    require(
        type(successes) is int
        and type(total) is int
        and 0 <= successes <= total,
        "Invalid binomial counts",
    )
    require(total > 0, "No observations")

    p = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    radius = (
        z / denominator
        * math.sqrt(
            p * (1 - p) / total
            + z2 / (4 * total * total)
        )
    )
    return [
        round(max(0.0, center - radius), 4),
        round(min(1.0, center + radius), 4),
    ]


def exact_paired_p(gains, losses):
    """
    Exact two-sided paired discordance test.

    Equivalent to exact McNemar conditional test
    on discordant correctness pairs.
    """
    require(
        type(gains) is int and gains >= 0
        and type(losses) is int and losses >= 0,
        "Invalid discordance counts",
    )

    n = gains + losses
    if n == 0:
        return 1.0

    tail = sum(
        math.comb(n, k)
        for k in range(min(gains, losses) + 1)
    ) / (2 ** n)

    return round(min(1.0, 2 * tail), 6)


def percentile(sorted_values, q):
    position = (len(sorted_values) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    return (
        sorted_values[lo] * (hi - position)
        + sorted_values[hi] * (position - lo)
        if lo != hi else sorted_values[lo]
    )


def paired_bootstrap_interval(
    paired_deltas,
    samples=BOOTSTRAP_SAMPLES,
    seed=BOOTSTRAP_SEED,
):
    """
    Resample paired example differences, not
    independent v1 and v2 samples.

    This conditions on the selected synthetic
    examples; it cannot address selection bias.
    """
    require(
        type(paired_deltas) is list
        and len(paired_deltas) > 0
        and all(value in (-1, 0, 1)
                for value in paired_deltas),
        "Invalid paired differences",
    )
    require(
        type(samples) is int and samples >= 100,
        "Too few bootstrap samples",
    )

    rng = random.Random(seed)
    n = len(paired_deltas)
    estimates = []

    for _ in range(samples):
        draw = [
            paired_deltas[rng.randrange(n)]
            for _ in range(n)
        ]
        estimates.append(100 * sum(draw) / n)

    estimates.sort()

    return [
        round(percentile(estimates, 0.025), 2),
        round(percentile(estimates, 0.975), 2),
    ]


def paired_evaluation(v1_predictions, v2_predictions, labels):
    """Align examples by ID and compare exact correctness."""
    a = {row["snapshot_id"]: row
         for row in v1_predictions}
    b = {row["snapshot_id"]: row
         for row in v2_predictions}

    require(
        len(a) == len(v1_predictions)
        and len(b) == len(v2_predictions)
        and set(a) == set(b) == set(labels),
        "Paired example identity mismatch",
    )

    gains = losses = both = neither = 0
    delta = []
    records = []
    counts = {
        "v1": {"correct": 0, "attack_tp": 0,
               "attack_fn": 0, "benign_fp": 0},
        "v2": {"correct": 0, "attack_tp": 0,
               "attack_fn": 0, "benign_fp": 0},
    }

    attacks = sum(value != "benign"
                  for value in labels.values())
    benign = len(labels) - attacks

    for sid in sorted(labels):
        truth = labels[sid]
        pred1 = a[sid]["predicted_category"]
        pred2 = b[sid]["predicted_category"]

        require(
            a[sid]["content_sha256"]
            == b[sid]["content_sha256"],
            "Compared different content",
        )

        old = pred1 == truth
        new = pred2 == truth

        gains += int(not old and new)
        losses += int(old and not new)
        both += int(old and new)
        neither += int(not old and not new)
        delta.append(int(new) - int(old))

        for label, predicted, correct in (
            ("v1", pred1, old),
            ("v2", pred2, new),
        ):
            counts[label]["correct"] += int(correct)

            if truth != "benign":
                counts[label]["attack_tp"] += int(
                    predicted != "benign"
                )
                counts[label]["attack_fn"] += int(
                    predicted == "benign"
                )
            elif predicted != "benign":
                counts[label]["benign_fp"] += 1

        records.append({
            "snapshot_id": sid,
            "true_category": truth,
            "v1_predicted": pred1,
            "v2_predicted": pred2,
            "v1_correct": old,
            "v2_correct": new,
            "outcome": (
                "GAIN" if not old and new
                else "LOSS" if old and not new
                else "BOTH_CORRECT" if old
                else "BOTH_INCORRECT"
            ),
        })

    n = len(labels)

    return {
        "examples": n,
        "attack_examples": attacks,
        "benign_examples": benign,
        "v1": {
            **counts["v1"],
            "accuracy": round(counts["v1"]["correct"] / n, 4),
            "accuracy_wilson95": wilson_interval(
                counts["v1"]["correct"], n
            ),
            "attack_recall_wilson95": wilson_interval(
                counts["v1"]["attack_tp"], attacks
            ) if attacks else None,
        },
        "v2": {
            **counts["v2"],
            "accuracy": round(counts["v2"]["correct"] / n, 4),
            "accuracy_wilson95": wilson_interval(
                counts["v2"]["correct"], n
            ),
            "attack_recall_wilson95": wilson_interval(
                counts["v2"]["attack_tp"], attacks
            ) if attacks else None,
        },
        "completion_gains": gains,
        "completion_losses": losses,
        "both_correct": both,
        "both_incorrect": neither,
        "net_correct_delta": gains - losses,
        "accuracy_delta_pp": round(
            100 * (gains - losses) / n, 2
        ),
        "exact_paired_p_two_sided":
            exact_paired_p(gains, losses),
        "paired_bootstrap_delta_pp_95":
            paired_bootstrap_interval(delta),
        "per_example": records,
    }


def validate_sources():
    """
    Validate saved C9.3 experiment against
    independently recomputed v1/v2 predictions.
    """
    handoff = registry.validate_module_8_handoff()
    datasets = v2.validated_datasets()

    summary_path = C3 / "results/m9_3_summary.json"
    manifest_path = C3 / "results/manifest.json"
    store_path = C3 / "results/candidate_experiment_store"

    for path in (summary_path, manifest_path, store_path):
        require(path.exists(), f"Missing source: {path}")

    summary = read_json(summary_path)
    manifest = read_json(manifest_path)
    store = registry.ExperimentRegistry(store_path)

    require(
        store.verify()
        and len(store.list_records()) == 3,
        "Invalid candidate experiment registry",
    )
    require(
        manifest["summary_sha256"] == digest(summary)
        and manifest["candidate_rule_sha256"]
        == v2.detector_hash()
        and manifest["audit_head"] == store.audit_head(),
        "Experiment manifest mismatch",
    )

    records = {
        row["configuration"]["dataset"]: row
        for row in store.list_records()
    }
    require(
        set(records) == set(DATASETS),
        "Missing candidate evaluations",
    )

    verified = {}

    for name in DATASETS:
        data = datasets[name]
        archive = data["archive"]
        labels = data["labels"]

        file_path = C3 / f"results/{name}_predictions.json"
        require(file_path.is_file(), "Missing predictions")

        saved = read_json(file_path)
        predicted_v1 = v2.v1.detect_archive(archive)
        predicted_v2 = v2.detect_archive(archive)

        require(
            saved == {"v1": predicted_v1, "v2": predicted_v2},
            f"Altered predictions: {name}",
        )

        _, evaluated_v1 = experiment.evaluate_one(
            archive, labels, v2.v1.detect_archive
        )
        _, evaluated_v2 = experiment.evaluate_one(
            archive, labels, v2.detect_archive
        )

        require(
            records[name]["result"] == evaluated_v2
            and records[name]["dataset_sha256"]
            == v2.digest(archive)
            and records[name]["detector_sha256"]
            == v2.detector_hash()
            and records[name]["status"]
            == "CANDIDATE_PENDING",
            "Registered candidate result mismatch",
        )

        result = summary["scores"][name]
        require(
            result["actual_dataset_sha256"]
            == v2.digest(archive)
            and result["v1"]["accuracy"]
            == evaluated_v1["accuracy"]
            and result["v2"]["accuracy"]
            == evaluated_v2["accuracy"]
            and result["v1"]["false_positives"]
            == evaluated_v1["binary_attack_detection"][
                "false_positive"
            ]
            and result["v2"]["false_positives"]
            == evaluated_v2["binary_attack_detection"][
                "false_positive"
            ],
            "Saved score summary mismatch",
        )

        verified[name] = paired_evaluation(
            predicted_v1, predicted_v2, labels
        )

    require(
        summary["candidate_promoted"] is False
        and summary["baseline_replaced"] is False
        and summary["permissions_changed"] == 0,
        "Unsafe experiment state",
    )

    return {
        "paired": verified,
        "summary": summary,
        "source_sha256": file_hash(summary_path),
        "registry_head": store.audit_head(),
        "baseline_rule_sha256":
            handoff["rule_sha256"],
        "candidate_rule_sha256":
            v2.detector_hash(),
    }


def release_gates(paired):
    """
    Research release policy, not real authorization.

    Strictly block promotion if the candidate loses
    original known-good attacks or the sample is
    too small for independent deployment evidence.
    """
    original = paired["original"]
    development = paired["development"]
    fresh = paired["fresh"]

    gates = {
        "original_accuracy_non_regression": (
            original["v2"]["correct"]
            >= original["v1"]["correct"]
        ),
        "original_attack_recall_non_regression": (
            original["v2"]["attack_fn"]
            <= original["v1"]["attack_fn"]
        ),
        "fresh_no_new_false_positives": (
            fresh["v2"]["benign_fp"]
            <= fresh["v1"]["benign_fp"]
        ),
        "fresh_attack_recall_not_worse": (
            fresh["v2"]["attack_fn"]
            <= fresh["v1"]["attack_fn"]
        ),
        "development_attack_recall_not_worse": (
            development["v2"]["attack_fn"]
            <= development["v1"]["attack_fn"]
        ),
        "minimum_independent_test_size": False,
        "independent_external_test_available": False,
        "live_agent_safety_evaluated": False,
        "authenticated_review_complete": False,
    }

    require(
        fresh["examples"] == 12,
        "Unexpected assessment size",
    )

    return {
        "gates": gates,
        "passed": sum(gates.values()),
        "failed": sum(not x for x in gates.values()),
        "release_decision": (
            "HOLD_FOR_REVIEW"
            if not all(gates.values())
            else "ELIGIBLE_FOR_SEPARATE_REVIEW"
        ),
        "candidate_promoted": False,
        "baseline_replaced": False,
        "permission_changes": 0,
    }


def run_analysis(
    output_dir=RESULTS,
    findings_dir=HERE,
):
    source = validate_sources()
    paired = source["paired"]
    gates = release_gates(paired)

    summary = {
        "module": 9,
        "chapter": "9.4",
        "evaluation_mode": "paired_synthetic_regression_analysis",
        "datasets": {
            name: {
                key: value
                for key, value in result.items()
                if key != "per_example"
            }
            for name, result in paired.items()
        },
        "release_gates": gates,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "candidate_status": "CANDIDATE_PENDING",
        "causal_safety_benefit_established": False,
        "real_world_generalization_established": False,
        "network_requests": 0,
        "model_calls": 0,
        "authorization_changes": 0,
    }

    output_dir = Path(output_dir)
    findings_dir = Path(findings_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    findings_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        output_dir / "m9_4_analysis.json", summary
    )
    write_json(
        output_dir / "paired_example_outcomes.json",
        {name: value["per_example"]
         for name, value in paired.items()},
    )

    write_json(
        output_dir / "manifest.json",
        {
            "module": 9,
            "chapter": "9.4",
            "source_experiment_sha256":
                source["source_sha256"],
            "source_registry_head":
                source["registry_head"],
            "baseline_rule_sha256":
                source["baseline_rule_sha256"],
            "candidate_rule_sha256":
                source["candidate_rule_sha256"],
            "analysis_sha256": digest(summary),
            "paired_outcomes_sha256": digest({
                name: value["per_example"]
                for name, value in paired.items()
            }),
        },
    )

    lines = [
        "# Guardian AI — Module 9.4 Findings",
        "",
        "| Dataset | Gains | Losses | Delta pp | Exact p |",
        "|---|---:|---:|---:|---:|",
    ]

    for name, row in paired.items():
        lines.append(
            f"| {name} | {row['completion_gains']} "
            f"| {row['completion_losses']} "
            f"| {row['accuracy_delta_pp']:+.2f} "
            f"| {row['exact_paired_p_two_sided']:.4f} |"
        )

    lines += [
        "",
        "## Release recommendation",
        "",
        gates["release_decision"],
        "",
        "Candidate v2 regressed on original fixtures.",
        "Improved synthetic scores do not substitute",
        "for a large independent evaluation or live",
        "agent-safety evidence.",
        "",
        "## Limitations",
        "",
        "- Small hand-authored synthetic sample",
        "- Data-dependent heuristic development",
        "- Conditional paired bootstrap intervals",
        "- No multiple-testing claims",
        "- No real-world performance inference",
        "- No independent reviewer authentication",
        "",
    ]

    findings = findings_dir / "M9_C4_FINDINGS.md"
    findings.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\nFINAL MODULE 9 C4 SUMMARY")
    print(json.dumps({
        "module": 9,
        "chapter": "9.4",
        "comparisons": {
            name: {
                "gains": row["completion_gains"],
                "losses": row["completion_losses"],
                "delta_accuracy_pp":
                    row["accuracy_delta_pp"],
                "exact_p_two_sided":
                    row["exact_paired_p_two_sided"],
                "bootstrap_delta_95":
                    row["paired_bootstrap_delta_pp_95"],
            }
            for name, row in paired.items()
        },
        "release_decision": gates["release_decision"],
        "gates_passed": gates["passed"],
        "gates_failed": gates["failed"],
        "candidate_promoted": False,
        "baseline_replaced": False,
        "authorization_changes": 0,
    }, indent=2))

    print("\nSaved:", output_dir / "m9_4_analysis.json")
    return summary


if __name__ == "__main__":
    run_analysis()
