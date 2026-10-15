
"""
Guardian AI — Module 9, Chapter 9.5

Final durability and scientific evidence benchmark.

Validates:
  9.1 Immutable frozen baseline experiment registry
  9.2 Durable pending lessons and fixture rollback
  9.3 Versioned v2 candidate experiment registry
  9.4 Paired statistics and conservative release gates

Outputs:
  results/m9_5_final_results.json
  results/manifest.json
  MODULE_9_RESEARCH_FINDINGS.md
  MODULE_10_INTERFACE.json

No model calls, network access, lesson activation,
capability issuance, or Guardian permission changes.

Local hash chains detect ordinary corruption but are
not tamper-proof against a privileged filesystem owner.
"""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
M9 = ROOT / "module_9_durable_learning"
RESULTS = HERE / "results"

for chapter in (
    "chapter_01",
    "chapter_02",
    "chapter_03",
    "chapter_04",
):
    sys.path.insert(0, str(M9 / chapter))

import guardian_experiment_registry as c1
import guardian_lesson_lifecycle as c2
import guardian_detector_v2 as detector
import c3_detector_experiment as c3
import m9_4_statistical_gates as c4


SOURCES = {
    "baseline_summary":
        M9 / "chapter_01/results/m9_1_summary.json",
    "baseline_manifest":
        M9 / "chapter_01/results/manifest.json",
    "lesson_summary":
        M9 / "chapter_02/results/m9_2_summary.json",
    "lesson_manifest":
        M9 / "chapter_02/results/manifest.json",
    "candidate_summary":
        M9 / "chapter_03/results/m9_3_summary.json",
    "candidate_manifest":
        M9 / "chapter_03/results/manifest.json",
    "statistical_analysis":
        M9 / "chapter_04/results/m9_4_analysis.json",
    "paired_outcomes":
        M9 / "chapter_04/results/paired_example_outcomes.json",
    "statistical_manifest":
        M9 / "chapter_04/results/manifest.json",
}

BASELINE_STORE = (
    M9 / "chapter_01/results/experiment_store"
)
PENDING_STORE = (
    M9 / "chapter_02/results/pending_lesson_store"
)
FIXTURE_STORE = (
    M9 / "chapter_02/results/fixture_lifecycle_store"
)
CANDIDATE_STORE = (
    M9 / "chapter_03/results/candidate_experiment_store"
)


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
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )


def validate_baseline(paths):
    handoff = c1.validate_module_8_handoff()
    expected = c1.make_baseline_records(handoff)

    store = c1.ExperimentRegistry(BASELINE_STORE)
    require(store.verify(), "Baseline store invalid")

    records = store.list_records()
    require(
        records == expected,
        "Frozen baseline differs from Module 8",
    )

    summary = read_json(paths["baseline_summary"])
    manifest = read_json(paths["baseline_manifest"])

    require(
        summary["baseline_experiments"] == 2
        and summary["registrations_accepted"] == 2
        and summary["duplicate_rejections"] == 2
        and summary["conflicting_result_rejections"] == 1
        and summary["record_tampering_detected"] is True
        and summary["restart_verification"] is True
        and summary["baseline_replaced"] is False
        and summary["candidate_models_activated"] == 0
        and summary["permission_changes"] == 0,
        "Baseline benchmark safety invariant failed",
    )

    require(
        summary["audit_head"] == store.audit_head()
        and summary["frozen_detector_sha256"]
        == handoff["rule_sha256"]
        and summary["source_module_8_sha256"]
        == handoff["final_sha256"],
        "Baseline provenance differs from source",
    )

    require(
        manifest["record_ids"] == [
            record["experiment_id"]
            for record in records
        ]
        and manifest["registry_head"] == store.audit_head()
        and manifest["summary_sha256"] == digest(summary),
        "Baseline manifest mismatch",
    )

    return {
        "records": records,
        "head": store.audit_head(),
        "summary": summary,
        "handoff": handoff,
    }


def validate_lessons(paths):
    expected = [
        c2.safe_lesson(candidate)
        for candidate in c2.trusted_candidates()
    ]
    require(
        len(expected) == 8,
        "Unexpected Module 8 candidate count",
    )

    pending = c2.LessonJournal(PENDING_STORE)
    fixture = c2.LessonJournal(
        FIXTURE_STORE,
        allow_fixture_activation=True,
    )

    require(
        pending.verify() and fixture.verify(),
        "Lesson journal verification failed",
    )

    pending_events = pending.history()
    fixture_events = fixture.history()
    expected_hashes = {
        lesson["lesson_id"]: c2.digest(lesson)
        for lesson in expected
    }

    require(
        len(pending_events) == len(expected) == 8
        and all(
            row["action"] == "SUBMIT"
            and row["to_state"] == "PENDING_REVIEW"
            for row in pending_events
        ),
        "Module 8 lessons were modified or promoted",
    )

    observed_hashes = {
        event["lesson_id"]: event["lesson_sha256"]
        for event in pending_events
    }

    require(
        observed_hashes == expected_hashes,
        "Pending lesson content hash mismatch",
    )

    pending_stats = pending.summary()
    fixture_stats = fixture.summary()

    require(
        pending_stats["lessons"] == 8
        and pending_stats["pending_review"] == 8
        and pending_stats["approved"] == 0
        and pending_stats["active"] == 0
        and pending_stats["rolled_back"] == 0,
        "Original lesson lifecycle changed",
    )

    fixture_lesson = c2.make_fixture_lesson()

    require(
        [event["action"] for event in fixture_events]
        == ["SUBMIT", "APPROVE", "ACTIVATE", "ROLLBACK"]
        and all(
            row["lesson_id"] == fixture_lesson["lesson_id"]
            and row["lesson_sha256"]
            == c2.digest(fixture_lesson)
            for row in fixture_events
        )
        and fixture.state(fixture_lesson["lesson_id"])
        == "ROLLED_BACK",
        "Synthetic fixture rollback invalid",
    )

    summary = read_json(paths["lesson_summary"])
    manifest = read_json(paths["lesson_manifest"])

    require(
        summary["imported_candidates"] == 8
        and summary["pending_review"] == 8
        and summary["approved_module_8_lessons"] == 0
        and summary["active_module_8_lessons"] == 0
        and summary["idempotent_replays_detected"] == 1
        and summary["idempotency_conflicts_rejected"] == 1
        and summary["unverified_approvals_rejected"] == 1
        and summary["synthetic_fixture_transitions"] == 4
        and summary["synthetic_fixture_final_state"]
        == "ROLLED_BACK"
        and summary["fixture_rollback_verified"] is True
        and summary["journal_tampering_detected"] is True
        and summary["live_guardian_lessons_activated"] == 0
        and summary["guardian_permissions_changed"] == 0,
        "Lesson benchmark mismatch",
    )

    require(
        manifest["summary_sha256"] == c2.digest(summary)
        and manifest["pending_audit_head"]
        == pending_stats["audit_head"]
        and manifest["fixture_audit_head"]
        == fixture_stats["audit_head"],
        "Lesson journal manifest mismatch",
    )

    return {
        "pending": pending_stats,
        "fixture": fixture_stats,
        "pending_head": pending_stats["audit_head"],
        "fixture_head": fixture_stats["audit_head"],
        "summary": summary,
    }


def validate_candidate_experiments(paths):
    source = c4.validate_sources()
    datasets = detector.validated_datasets()
    handoff = c1.validate_module_8_handoff()

    stored = c1.ExperimentRegistry(CANDIDATE_STORE)
    require(
        stored.verify(),
        "Candidate experiment store invalid",
    )

    records = stored.list_records()

    require(
        len(records) == 3,
        "Incorrect candidate experiment count",
    )

    for name in c4.DATASETS:
        data = datasets[name]
        archive = data["archive"]
        labels = data["labels"]

        _, score = c3.evaluate_one(
            archive,
            labels,
            detector.detect_archive,
        )

        expected = c3.make_candidate_record(
            name, archive, score, handoff
        )

        require(
            expected in records,
            f"Candidate experiment differs: {name}",
        )

    summary = read_json(paths["candidate_summary"])
    manifest = read_json(paths["candidate_manifest"])

    require(
        summary == source["summary"]
        and summary["evaluated_datasets"] == 3
        and summary["evaluated_examples"] == 48
        and summary["candidate_experiments_registered"] == 3
        and summary["candidate_status"] == "CANDIDATE_PENDING"
        and summary["candidate_promoted"] is False
        and summary["baseline_replaced"] is False
        and summary["permissions_changed"] == 0,
        "Candidate experiment summary mismatch",
    )

    require(
        manifest["summary_sha256"] == digest(summary)
        and manifest["candidate_rule_sha256"]
        == detector.detector_hash()
        and manifest["candidate_experiment_ids"]
        == [record["experiment_id"] for record in records]
        and manifest["audit_head"] == stored.audit_head(),
        "Candidate experiment manifest mismatch",
    )

    return {
        "records": records,
        "head": stored.audit_head(),
        "summary": summary,
        "paired": source["paired"],
    }


def validate_statistics(paths, paired, candidate_head):
    stored_analysis = read_json(
        paths["statistical_analysis"]
    )
    stored_outcomes = read_json(
        paths["paired_outcomes"]
    )
    manifest = read_json(
        paths["statistical_manifest"]
    )

    derived_outcomes = {
        name: record["per_example"]
        for name, record in paired.items()
    }

    derived_gates = c4.release_gates(paired)

    derived_analysis = {
        "module": 9,
        "chapter": "9.4",
        "evaluation_mode":
            "paired_synthetic_regression_analysis",
        "datasets": {
            name: {
                key: value
                for key, value in record.items()
                if key != "per_example"
            }
            for name, record in paired.items()
        },
        "release_gates": derived_gates,
        "bootstrap_samples": c4.BOOTSTRAP_SAMPLES,
        "bootstrap_seed": c4.BOOTSTRAP_SEED,
        "candidate_status": "CANDIDATE_PENDING",
        "causal_safety_benefit_established": False,
        "real_world_generalization_established": False,
        "network_requests": 0,
        "model_calls": 0,
        "authorization_changes": 0,
    }

    require(
        stored_outcomes == derived_outcomes,
        "Paired outcome artifact differs from model results",
    )
    require(
        stored_analysis == derived_analysis,
        "Statistical analysis differs from recomputation",
    )

    source_path = (
        M9 / "chapter_03/results/m9_3_summary.json"
    )

    require(
        manifest["source_experiment_sha256"]
        == file_hash(source_path)
        and manifest["source_registry_head"]
        == candidate_head
        and manifest["baseline_rule_sha256"]
        == c1.validate_module_8_handoff()["rule_sha256"]
        and manifest["candidate_rule_sha256"]
        == detector.detector_hash()
        and manifest["analysis_sha256"]
        == digest(stored_analysis)
        and manifest["paired_outcomes_sha256"]
        == digest(stored_outcomes),
        "Statistical manifest mismatch",
    )

    require(
        derived_gates["release_decision"]
        == "HOLD_FOR_REVIEW"
        and derived_gates["passed"] == 3
        and derived_gates["failed"] == 6
        and derived_gates["candidate_promoted"] is False
        and derived_gates["baseline_replaced"] is False,
        "Conservative release decision changed",
    )

    return {
        "analysis": stored_analysis,
        "gates": derived_gates,
        "paired": paired,
    }


def validate_all(paths=SOURCES):
    for name, path in paths.items():
        require(
            Path(path).is_file(),
            f"Missing evidence file: {name}",
        )

    baseline = validate_baseline(paths)
    lessons = validate_lessons(paths)
    candidate = validate_candidate_experiments(paths)
    stats = validate_statistics(
        paths,
        candidate["paired"],
        candidate["head"],
    )

    return {
        "baseline": baseline,
        "lessons": lessons,
        "candidate": candidate,
        "statistics": stats,
    }


def make_final(source):
    paired = source["statistics"]["paired"]
    gates = source["statistics"]["gates"]

    comparisons = {}

    for name in c4.DATASETS:
        row = paired[name]
        comparisons[name] = {
            "examples": row["examples"],
            "v1_accuracy": row["v1"]["accuracy"],
            "v2_accuracy": row["v2"]["accuracy"],
            "gains": row["completion_gains"],
            "losses": row["completion_losses"],
            "accuracy_delta_pp": row["accuracy_delta_pp"],
            "exact_p_two_sided":
                row["exact_paired_p_two_sided"],
            "bootstrap_delta_pp_95":
                row["paired_bootstrap_delta_pp_95"],
        }

    return {
        "module": 9,
        "chapter": "9.5",
        "evaluation_status": "FULL_RECORD_ANALYSIS",
        "scientific_verdict": (
            "Detector v2 improved on development and "
            "additional synthetic cases but regressed on "
            "original known-good cases; release remains on hold."
        ),
        "frozen_baseline_experiments": len(
            source["baseline"]["records"]
        ),
        "candidate_detector_experiments": len(
            source["candidate"]["records"]
        ),
        "module_8_lessons_pending":
            source["lessons"]["pending"]["pending_review"],
        "module_8_lessons_approved": 0,
        "module_8_lessons_active": 0,
        "synthetic_fixture_rolled_back": True,
        "experiment_audit_heads": {
            "baseline": source["baseline"]["head"],
            "candidate": source["candidate"]["head"],
        },
        "lesson_audit_heads": {
            "pending": source["lessons"]["pending_head"],
            "fixture": source["lessons"]["fixture_head"],
        },
        "detector_comparisons": comparisons,
        "release_decision": gates["release_decision"],
        "release_gates_passed": gates["passed"],
        "release_gates_failed": gates["failed"],
        "candidate_detector_status": "CANDIDATE_PENDING",
        "candidate_promoted": False,
        "baseline_replaced": False,
        "live_guardian_lessons_activated": 0,
        "authorization_changes": 0,
        "model_calls": 0,
        "network_requests": 0,
        "durable_model_learning_demonstrated": False,
        "real_world_generalization_established": False,
        "reproducibility_note": (
            "Deterministic local record reconstruction "
            "with non-externally-anchored audit chains."
        ),
    }


def make_module_10_contract(final):
    return {
        "schema_version": "1.0",
        "producer_module": 9,
        "consumer_module": 10,
        "integration_mode": "read_only_research",
        "module_9_final_sha256": digest(final),
        "research_status": "candidate_held_for_review",
        "supported_views": [
            "experiment_provenance",
            "frozen_detector_comparison",
            "paired_statistical_outcomes",
            "lesson_review_queue",
            "audit_verification_status",
            "release_gate_summary",
        ],
        "required_components": [
            "Read-only artifact loader",
            "Source and manifest verification",
            "Provenance-aware experiment views",
            "Research limitations and uncertainty",
            "Fail-closed missing-evidence handling",
            "Separated candidate and baseline states",
            "No raw attack text in public summaries",
        ],
        "authorization_constraints": {
            "automatic_detector_promotion": False,
            "automatic_lesson_approval": False,
            "lesson_activation_via_dashboard": False,
            "agents_change_permissions": False,
            "agents_issue_capabilities": False,
            "network_access_enabled_by_default": False,
            "execute_untrusted_content": False,
        },
        "source_registry_heads": final[
            "experiment_audit_heads"
        ],
        "source_lesson_heads": final[
            "lesson_audit_heads"
        ],
        "limitations": [
            "Synthetic evidence only",
            "No independent real-world validation",
            "No externally anchored audit",
            "No production reviewer authentication",
            "No demonstrated durable model learning",
            "No causal improvement in live agent safety",
        ],
    }


def make_findings(final):
    lines = [
        "# Guardian AI — Module 9 Research Findings",
        "",
        "## Research question",
        "",
        "Can Guardian preserve research evidence and",
        "evaluate candidate improvements without",
        "silently overwriting the baseline or lessons?",
        "",
        "## Experiment registry",
        "",
        "- Frozen baseline experiments: 2",
        "- Candidate detector experiments: 3",
        "- Candidate state: CANDIDATE_PENDING",
        "- Original baseline replaced: No",
        "",
        "## Lesson lifecycle",
        "",
        "- Original candidates pending review: 8",
        "- Original candidates activated: 0",
        "- Synthetic rollback fixture: Verified",
        "",
        "## Detector comparisons",
        "",
        "| Dataset | v1 | v2 | Change (pp) |",
        "|---|---:|---:|---:|",
    ]

    for name, row in final["detector_comparisons"].items():
        lines.append(
            f"| {name} | {row['v1_accuracy']:.4f} "
            f"| {row['v2_accuracy']:.4f} "
            f"| {row['accuracy_delta_pp']:+.2f} |"
        )

    lines.extend([
        "",
        "## Release decision",
        "",
        f"**{final['release_decision']}**",
        "",
        "The candidate improved on two synthetic",
        "datasets but regressed on the original suite.",
        "Six of nine release requirements remain unmet.",
        "",
        "## Limitations",
        "",
        "- No real-world generalization established",
        "- Development examples were previously inspected",
        "- Additional assessment is small and project-authored",
        "- Local audit chains are not tamper-proof",
        "- No independently authenticated reviewer",
        "- No live agent improvement demonstrated",
        "",
        "## Module 10 handoff",
        "",
        "Integrate validated research summaries through",
        "a read-only interface. Never allow dashboard",
        "controls or model output to activate a lesson,",
        "promote a detector or change Guardian policy.",
        "",
    ])

    return "\n".join(lines)


def run_final(
    paths=SOURCES,
    output_dir=RESULTS,
    findings_dir=HERE,
):
    evidence = validate_all(paths)
    final = make_final(evidence)
    contract = make_module_10_contract(final)

    require(
        final["release_decision"] == "HOLD_FOR_REVIEW"
        and final["frozen_baseline_experiments"] == 2
        and final["candidate_detector_experiments"] == 3
        and final["module_8_lessons_pending"] == 8
        and final["module_8_lessons_active"] == 0
        and final["release_gates_failed"] == 6
        and final["authorization_changes"] == 0,
        "Final Module 9 safety invariant failed",
    )

    output_dir = Path(output_dir)
    findings_dir = Path(findings_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    findings_dir.mkdir(parents=True, exist_ok=True)

    final_path = output_dir / "m9_5_final_results.json"
    findings_path = (
        findings_dir / "MODULE_9_RESEARCH_FINDINGS.md"
    )
    contract_path = (
        findings_dir / "MODULE_10_INTERFACE.json"
    )
    manifest_path = output_dir / "manifest.json"

    write_json(final_path, final)
    write_json(contract_path, contract)

    findings_path.write_text(
        make_findings(final) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "module": 9,
        "chapter": "9.5",
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
        "baseline_experiments":
            final["frozen_baseline_experiments"],
        "candidate_experiments":
            final["candidate_detector_experiments"],
        "lessons_pending":
            final["module_8_lessons_pending"],
        "lessons_approved": 0,
        "lessons_active": 0,
        "fixture_rollback_verified": True,
        "detector_comparisons":
            final["detector_comparisons"],
        "release_decision": final["release_decision"],
        "release_gates_passed":
            final["release_gates_passed"],
        "release_gates_failed":
            final["release_gates_failed"],
        "baseline_replaced": False,
        "candidate_promoted": False,
        "permission_changes": 0,
        "model_calls": 0,
        "module_10_contract_valid": (
            contract["producer_module"] == 9
            and contract["consumer_module"] == 10
            and all(
                value is False
                for value in contract[
                    "authorization_constraints"
                ].values()
            )
        ),
    }

    print("\nFINAL MODULE 9 C5 SUMMARY")
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
