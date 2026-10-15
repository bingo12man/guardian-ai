
"""
Guardian AI — Module 8, Chapter 8.5

Final offline observatory benchmark.

Reconstructs and validates:
    8.1 Synthetic source registry
    8.2 Detector and original evaluation
    8.3 Pending lesson candidates
    8.4 Frozen-detector robustness suite

Writes:
    results/m8_5_final_results.json
    results/manifest.json
    MODULE_8_RESEARCH_FINDINGS.md
    MODULE_9_INTERFACE.json

No network access, model inference, permission
changes, or automatic lesson publication.
"""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
M8 = ROOT / "module_8_observatory"
RESULTS = HERE / "results"

for chapter in (
    "chapter_01",
    "chapter_02",
    "chapter_03",
    "chapter_04",
):
    sys.path.insert(0, str(M8 / chapter))

import guardian_source_registry as c1
import guardian_threat_detector as c2
import c2_detection_benchmark as evaluation
import guardian_lesson_candidates as c3
import guardian_decoy_monitor as c4
import c4_robustness_benchmark as robustness


SOURCES = {
    "c1_archive": M8 / "chapter_01/results/verified_snapshots.json",
    "c1_labels": M8 / "chapter_01/results/ground_truth_labels.json",
    "c1_audit": M8 / "chapter_01/results/source_audit.json",
    "c1_summary": M8 / "chapter_01/results/m8_1_summary.json",
    "c1_manifest": M8 / "chapter_01/results/manifest.json",
    "c2_predictions": M8 / "chapter_02/results/threat_predictions.json",
    "c2_evaluation": M8 / "chapter_02/results/detection_evaluation.json",
    "c2_summary": M8 / "chapter_02/results/m8_2_summary.json",
    "c2_manifest": M8 / "chapter_02/results/manifest.json",
    "c3_candidates": M8 / "chapter_03/results/lesson_candidates.json",
    "c3_groups": M8 / "chapter_03/results/deduplicated_lesson_groups.json",
    "c3_queue": M8 / "chapter_03/results/pending_review_queue.json",
    "c3_summary": M8 / "chapter_03/results/m8_3_summary.json",
    "c3_manifest": M8 / "chapter_03/results/manifest.json",
    "c4_decoys": M8 / "chapter_04/results/synthetic_decoys.json",
    "c4_labels": M8 / "chapter_04/results/decoy_labels.json",
    "c4_predictions": M8 / "chapter_04/results/decoy_predictions.json",
    "c4_outcomes": M8 / "chapter_04/results/detection_outcomes.json",
    "c4_evaluation": M8 / "chapter_04/results/robustness_evaluation.json",
    "c4_alerts": M8 / "chapter_04/results/threat_alerts.json",
    "c4_candidates": M8 / "chapter_04/results/pending_decoy_candidates.json",
    "c4_groups": M8 / "chapter_04/results/pending_decoy_groups.json",
    "c4_summary": M8 / "chapter_04/results/m8_4_summary.json",
    "c4_manifest": M8 / "chapter_04/results/manifest.json",
}


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
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def validate_sources(paths=SOURCES):
    """Recompute trusted evidence, not just file hashes."""
    for name, path in paths.items():
        require(
            Path(path).is_file(),
            f"Missing required source: {name}",
        )

    # 8.1 — original synthetic source registry
    c1.validate_handoff()

    archive = c1.make_snapshots()
    labels = c1.make_ground_truth()
    registry = c1.SourceRegistry(
        c1.approved_sources(), archive
    )

    for snapshot in archive:
        require(
            registry.ingest(snapshot)["accepted"],
            "Chapter 8.1 reconstruction failed",
        )

    require(
        read_json(paths["c1_archive"]) == archive,
        "Original archive mismatch",
    )
    require(
        read_json(paths["c1_labels"]) == labels,
        "Original fixture labels mismatch",
    )
    require(
        read_json(paths["c1_audit"]) == registry.audit(),
        "Original source audit mismatch",
    )

    c1.verify_audit(
        registry.audit(), registry.audit_head()
    )

    c1_manifest = read_json(paths["c1_manifest"])
    c1_summary = read_json(paths["c1_summary"])

    require(
        c1_manifest["snapshots_sha256"] == c1.fingerprint(archive)
        and c1_manifest["ground_truth_sha256"] == c1.fingerprint(labels)
        and c1_manifest["audit_sha256"]
        == c1.fingerprint(registry.audit())
        and c1_manifest["summary_sha256"]
        == c1.fingerprint(c1_summary),
        "Chapter 8.1 manifest mismatch",
    )
    require(
        c1_summary["accepted_snapshots"] == 12
        and c1_summary["network_requests"] == 0
        and c1_summary["permission_changes"] == 0,
        "Chapter 8.1 invariant mismatch",
    )

    # 8.2 — frozen detector and original metrics
    original_predictions = c2.detect_archive(archive)
    original_eval = evaluation.evaluate(
        original_predictions, labels
    )

    c2.verify_predictions(
        original_predictions, archive
    )

    require(
        read_json(paths["c2_predictions"])
        == original_predictions,
        "Original predictions mismatch",
    )
    require(
        read_json(paths["c2_evaluation"])
        == original_eval,
        "Original evaluation mismatch",
    )

    c2_manifest = read_json(paths["c2_manifest"])
    c2_summary = read_json(paths["c2_summary"])

    require(
        c2_manifest["archive_sha256"]
        == c1.fingerprint(archive)
        and c2_manifest["predictions_sha256"]
        == c1.fingerprint(original_predictions)
        and c2_manifest["evaluation_sha256"]
        == c1.fingerprint(original_eval)
        and c2_manifest["summary_sha256"]
        == c1.fingerprint(c2_summary)
        and c2_manifest["detector_rule_sha256"]
        == c1.fingerprint(c2.RULES),
        "Chapter 8.2 manifest mismatch",
    )
    require(
        c2_summary["accuracy"] == original_eval["accuracy"]
        and c2_summary["macro_f1"] == original_eval["macro_f1"]
        and c2_summary["evaluated_snapshots"] == 12
        and c2_summary["authorization_changes"] == 0,
        "Original evaluation summary mismatch",
    )

    # 8.3 — controller-owned candidate generation
    candidates = c3.make_candidates(original_predictions)
    groups = c3.group_candidates(candidates)
    queue = c3.review_queue(candidates)

    c3.validate_candidates(
        candidates, groups, original_predictions
    )
    c3.validate_against_archive(candidates, archive)

    for key, expected in (
        ("c3_candidates", candidates),
        ("c3_groups", groups),
        ("c3_queue", queue),
    ):
        require(
            read_json(paths[key]) == expected,
            f"Chapter 8.3 artifact mismatch: {key}",
        )

    c3_manifest = read_json(paths["c3_manifest"])
    c3_summary = read_json(paths["c3_summary"])

    require(
        c3_manifest["candidates_sha256"] == digest(candidates)
        and c3_manifest["groups_sha256"] == digest(groups)
        and c3_manifest["review_queue_sha256"] == digest(queue)
        and c3_manifest["summary_sha256"] == digest(c3_summary)
        and c3_manifest["source_predictions_sha256"]
        == digest(original_predictions),
        "Chapter 8.3 manifest mismatch",
    )
    require(
        c3_summary["generated_candidates"] == len(candidates) == 8
        and c3_summary["unique_lesson_templates"] == len(groups) == 4
        and c3_summary["approved_lessons"] == 0
        and c3_summary["published_lessons"] == 0
        and c3_summary["automatic_lesson_promotion"] is False,
        "Chapter 8.3 publication invariant mismatch",
    )

    # 8.4 — genuinely separate fixed synthetic suite
    decoys = c4.make_decoys()
    decoy_labels = c4.make_labels()

    c4.validate_decoys(decoys)

    predictions = c4.detect_decoys(decoys)
    monitored = c4.monitor_findings(
        decoys, predictions
    )
    c4.validate_monitor_output(
        decoys, predictions, monitored
    )

    robust_eval = evaluation.evaluate(
        predictions, decoy_labels
    )

    outcomes = []
    false_positives = []
    false_negatives = []
    wrong_families = []

    for prediction in predictions:
        sid = prediction["snapshot_id"]
        actual = decoy_labels[sid]
        predicted = prediction["predicted_category"]

        outcomes.append({
            "snapshot_id": sid,
            "ground_truth": actual,
            "predicted": predicted,
            "correct": actual == predicted,
        })

        if actual == "benign" and predicted != "benign":
            false_positives.append(sid)
        elif actual != "benign" and predicted == "benign":
            false_negatives.append(sid)
        elif (
            actual != "benign"
            and predicted != "benign"
            and actual != predicted
        ):
            wrong_families.append(sid)

    reconstructed = {
        "c4_decoys": decoys,
        "c4_labels": decoy_labels,
        "c4_predictions": predictions,
        "c4_outcomes": outcomes,
        "c4_evaluation": robust_eval,
        "c4_alerts": monitored["alerts"],
        "c4_candidates": monitored["candidates"],
        "c4_groups": monitored["groups"],
    }

    for key, expected in reconstructed.items():
        require(
            read_json(paths[key]) == expected,
            f"Chapter 8.4 artifact mismatch: {key}",
        )

    c4_manifest = read_json(paths["c4_manifest"])
    c4_summary = read_json(paths["c4_summary"])

    require(
        c4_manifest["frozen_rule_sha256"]
        == c1.fingerprint(c2.RULES),
        "Robustness evaluation used modified rules",
    )
    for key, expected in reconstructed.items():
        artifact = key[3:]
        require(
            c4_manifest["artifacts_sha256"][artifact]
            == digest(expected),
            f"Robustness checksum mismatch: {artifact}",
        )

    require(
        c4_manifest["artifacts_sha256"]["summary"]
        == digest(c4_summary),
        "Robustness summary checksum mismatch",
    )

    binary = robust_eval["binary_attack_detection"]

    require(
        c4_summary["mode"] == "frozen_rule_synthetic_robustness"
        and c4_summary["decoys"] == 24
        and c4_summary["attack_decoys"] == 16
        and c4_summary["hard_benign_decoys"] == 8
        and c4_summary["multiclass_accuracy"]
        == robust_eval["accuracy"]
        and c4_summary["macro_f1"] == robust_eval["macro_f1"]
        and c4_summary["binary_detection"] == binary
        and c4_summary["false_positive_ids"] == false_positives
        and c4_summary["false_negative_ids"] == false_negatives
        and c4_summary["wrong_attack_family_ids"] == wrong_families
        and c4_summary["alerts_generated"]
        == len(monitored["alerts"])
        and c4_summary["pending_candidates"]
        == len(monitored["candidates"])
        and c4_summary["false_candidate_alerts"]
        == len(false_positives)
        and c4_summary["candidate_publications"] == 0
        and c4_summary["detector_rules_frozen"] is True
        and c4_summary["authorization_changes"] == 0,
        "Robustness summary does not match recomputation",
    )

    return {
        "original": original_eval,
        "robustness": robust_eval,
        "original_candidates": candidates,
        "robustness_candidates": monitored["candidates"],
        "robustness_alerts": monitored["alerts"],
        "false_positive_ids": false_positives,
        "false_negative_ids": false_negatives,
        "wrong_family_ids": wrong_families,
        "rule_sha256": c1.fingerprint(c2.RULES),
    }


def make_final(evidence):
    original = evidence["original"]
    robust = evidence["robustness"]
    binary_original = original["binary_attack_detection"]
    binary_robust = robust["binary_attack_detection"]

    return {
        "module": 8,
        "chapter": "8.5",
        "evaluation_status": "FULL_RECORD_ANALYSIS",
        "research_verdict": (
            "Frozen rules fit original synthetic fixtures "
            "but performed poorly on new synthetic variants "
            "and difficult benign examples."
        ),
        "original_dataset": {
            "examples": original["examples"],
            "correct": original["correct"],
            "accuracy": original["accuracy"],
            "macro_f1": original["macro_f1"],
            "attack_precision": binary_original["precision"],
            "attack_recall": binary_original["recall"],
            "attack_f1": binary_original["f1"],
        },
        "robustness_dataset": {
            "examples": robust["examples"],
            "correct": robust["correct"],
            "accuracy": robust["accuracy"],
            "macro_f1": robust["macro_f1"],
            "attack_precision": binary_robust["precision"],
            "attack_recall": binary_robust["recall"],
            "attack_f1": binary_robust["f1"],
            "attack_false_positives":
                binary_robust["false_positive"],
            "attack_false_negatives":
                binary_robust["false_negative"],
            "true_positives":
                binary_robust["true_positive"],
            "true_negatives":
                binary_robust["true_negative"],
        },
        "performance_gap": {
            "accuracy_drop_pp": round(
                100 * (
                    original["accuracy"]
                    - robust["accuracy"]
                ), 2
            ),
            "macro_f1_drop_pp": round(
                100 * (
                    original["macro_f1"]
                    - robust["macro_f1"]
                ), 2
            ),
            "binary_recall_drop_pp": round(
                100 * (
                    binary_original["recall"]
                    - binary_robust["recall"]
                ), 2
            ),
        },
        "lesson_pipeline": {
            "original_pending_candidates":
                len(evidence["original_candidates"]),
            "robustness_pending_candidates":
                len(evidence["robustness_candidates"]),
            "robustness_false_candidate_alerts":
                len(evidence["false_positive_ids"]),
            "approved": 0,
            "published": 0,
        },
        "false_positive_ids":
            evidence["false_positive_ids"],
        "false_negative_ids":
            evidence["false_negative_ids"],
        "wrong_family_ids":
            evidence["wrong_family_ids"],
        "frozen_detector_rule_sha256":
            evidence["rule_sha256"],
        "authorization_changes": 0,
        "network_requests": 0,
        "model_calls": 0,
        "instructions_executed": 0,
        "real_world_generalization_established": False,
        "model_behavioral_improvement_established": False,
        "scientific_limitations": [
            "Both datasets are synthetic and hand-authored",
            "Original rules were designed for original patterns",
            "Robustness variants are not a random attack sample",
            "Small sample size",
            "No independently authenticated public collection",
            "No live agent response experiments",
            "No independent model-weight learning",
        ],
    }


def make_module_9_contract(final):
    """Interface specification, not a capability grant."""
    return {
        "schema_version": "1.0",
        "producer_module": 8,
        "consumer_module": 9,
        "research_status": "robustness_failure_observed",
        "source_module_8_sha256": digest(final),
        "detector_rule_sha256":
            final["frozen_detector_rule_sha256"],
        "durable_record_schema": {
            "record_id": "controller_generated_unique_id",
            "experiment_id": "immutable_experiment_identifier",
            "source_fingerprint": "sha256",
            "detector_version": "frozen_version",
            "source_split": "train_validation_or_test",
            "prediction": "allowlisted_detection_category",
            "trusted_ground_truth": "separate_evaluation_only",
            "audit_head": "verified_controller_audit_hash",
        },
        "required_controls": [
            "Immutable dataset and experiment fingerprints",
            "Explicit train and evaluation separation",
            "Versioned detector and lesson templates",
            "Idempotent event ingestion",
            "Correlated-evidence deduplication",
            "Replayable deterministic benchmark fixtures",
            "Separate blind evaluation",
            "False-positive and false-negative monitoring",
            "Rollback of unapproved learning changes",
            "Trusted review before lesson activation",
        ],
        "research_priorities": [
            "Improve recall on unfamiliar attack wording",
            "Reduce benign quotation false positives",
            "Evaluate independent synthetic families",
            "Compare candidate detectors against frozen baseline",
            "Avoid test-set tuning and data leakage",
        ],
        "authorization_constraints": {
            "automated_lesson_approval": False,
            "collected_text_changes_policy": False,
            "agents_issue_capabilities": False,
            "agents_change_permissions": False,
            "network_collection_enabled_by_default": False,
            "raw_collected_code_execution": False,
        },
        "prohibited_record_fields": [
            "credentials",
            "capability_token",
            "permission_override",
            "private_document_contents",
        ],
    }


def make_findings(final):
    a = final["original_dataset"]
    b = final["robustness_dataset"]
    gap = final["performance_gap"]

    return "\n".join([
        "# Guardian AI — Module 8 Research Findings",
        "",
        "## Research question",
        "",
        "Can an offline observatory detect synthetic",
        "prompt-injection threats, preserve evidence,",
        "and generate safe lesson candidates?",
        "",
        "## Detection results",
        "",
        "| Metric | Original fixtures | Robustness decoys |",
        "|---|---:|---:|",
        f"| Examples | {a['examples']} | {b['examples']} |",
        f"| Correct | {a['correct']} | {b['correct']} |",
        f"| Accuracy | {a['accuracy']:.4f} | {b['accuracy']:.4f} |",
        f"| Macro F1 | {a['macro_f1']:.4f} | {b['macro_f1']:.4f} |",
        f"| Attack recall | {a['attack_recall']:.4f} "
        f"| {b['attack_recall']:.4f} |",
        "",
        f"Accuracy dropped {gap['accuracy_drop_pp']:.2f}"
        " percentage points on the robustness suite.",
        "",
        "## Candidate lessons",
        "",
        "All candidates remain pending review.",
        "False-positive detections can produce",
        "false candidate alerts, so detection is",
        "not sufficient for lesson approval.",
        "",
        "## Scientific conclusion",
        "",
        "Perfect performance on matching fixtures",
        "did not transfer to new synthetic variants.",
        "The frozen rule detector missed most new",
        "attack phrasings and flagged benign examples.",
        "",
        "## Limitations",
        "",
        "Both datasets are hand-authored synthetic suites.",
        "These results cannot establish a population",
        "accuracy estimate or live-agent safety benefit.",
        "",
        "## Module 9 handoff",
        "",
        "Use versioned, immutable evidence; separate",
        "train/evaluation splits; compare candidate",
        "detectors against this frozen baseline;",
        "and require independent validation before",
        "publishing any lesson.",
        "",
    ])


def run_final(
    paths=SOURCES,
    output_dir=RESULTS,
    findings_dir=HERE,
):
    evidence = validate_sources(paths)
    final = make_final(evidence)
    contract = make_module_9_contract(final)

    require(
        final["original_dataset"]["correct"] == 12
        and final["robustness_dataset"]["correct"] == 5
        and final["robustness_dataset"][
            "attack_false_negatives"
        ] == 15
        and final["robustness_dataset"][
            "attack_false_positives"
        ] == 4
        and final["lesson_pipeline"]["published"] == 0,
        "Unexpected Module 8 benchmark result",
    )

    output_dir = Path(output_dir)
    findings_dir = Path(findings_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    findings_dir.mkdir(parents=True, exist_ok=True)

    final_path = output_dir / "m8_5_final_results.json"
    manifest_path = output_dir / "manifest.json"
    contract_path = findings_dir / "MODULE_9_INTERFACE.json"
    findings_path = (
        findings_dir / "MODULE_8_RESEARCH_FINDINGS.md"
    )

    write_json(final_path, final)
    write_json(contract_path, contract)
    findings_path.write_text(
        make_findings(final),
        encoding="utf-8",
    )

    manifest = {
        "module": 8,
        "chapter": "8.5",
        "source_file_sha256": {
            name: file_hash(path)
            for name, path in sorted(paths.items())
        },
        "final_sha256": file_hash(final_path),
        "contract_sha256": file_hash(contract_path),
        "findings_sha256": file_hash(findings_path),
    }
    write_json(manifest_path, manifest)

    summary = {
        "evaluation_status": final["evaluation_status"],
        "original_accuracy":
            final["original_dataset"]["accuracy"],
        "robustness_accuracy":
            final["robustness_dataset"]["accuracy"],
        "accuracy_drop_pp":
            final["performance_gap"]["accuracy_drop_pp"],
        "robustness_macro_f1":
            final["robustness_dataset"]["macro_f1"],
        "robustness_attack_precision":
            final["robustness_dataset"]["attack_precision"],
        "robustness_attack_recall":
            final["robustness_dataset"]["attack_recall"],
        "robustness_false_positives":
            final["robustness_dataset"]["attack_false_positives"],
        "robustness_false_negatives":
            final["robustness_dataset"]["attack_false_negatives"],
        "original_pending_candidates":
            final["lesson_pipeline"][
                "original_pending_candidates"
            ],
        "robustness_pending_candidates":
            final["lesson_pipeline"][
                "robustness_pending_candidates"
            ],
        "published_lessons": 0,
        "network_requests": 0,
        "instructions_executed": 0,
        "authorization_changes": 0,
        "real_world_generalization_established": False,
        "module_9_contract_valid": (
            contract["producer_module"] == 8
            and contract["consumer_module"] == 9
            and contract["authorization_constraints"][
                "automated_lesson_approval"
            ] is False
        ),
    }

    print("\nFINAL MODULE 8 C5 SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nResults:", final_path)
    print("Findings:", findings_path)
    print("Contract:", contract_path)
    print("Manifest:", manifest_path)

    return {
        "summary": summary,
        "final": final,
        "contract": contract,
        "manifest": manifest,
    }


if __name__ == "__main__":
    run_final()
