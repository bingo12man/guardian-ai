
"""
Guardian AI — Module 9.1 registry benchmark.

Creates two immutable Module 8 baseline records.

Tests:
  - initial registration
  - duplicate re-registration
  - conflicting results
  - restart and audit verification
  - altered stored record detection

No detector retraining or network activity.
"""

import copy
import json
import shutil
from pathlib import Path

import guardian_experiment_registry as core

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def run_benchmark(
    output_dir=RESULTS,
    findings_dir=HERE,
):
    handoff = core.validate_module_8_handoff()
    records = core.make_baseline_records(handoff)

    output_dir = Path(output_dir)
    findings_dir = Path(findings_dir)

    # Keep the benchmark reproducible: create
    # a fresh dedicated sample registry.
    storage = output_dir / "experiment_store"

    if storage.exists():
        raise ValueError(
            "Existing registry detected; do not delete "
            "or overwrite immutable experiment history. "
            "Use a new output directory."
        )

    registry = core.ExperimentRegistry(storage)

    registrations = [
        registry.register(record)
        for record in records
    ]

    duplicates = [
        registry.register(record)
        for record in records
    ]

    conflict = copy.deepcopy(records[0])
    conflict["result"]["accuracy"] = 0.0

    conflict_decision = registry.register(conflict)

    restarted = core.ExperimentRegistry(storage)

    core.require(
        all(x["accepted"] for x in registrations),
        "Frozen baseline registration failed",
    )
    core.require(
        all(
            x["reason"] == "DUPLICATE_EXPERIMENT"
            for x in duplicates
        ),
        "Duplicate experiment was accepted",
    )
    core.require(
        conflict_decision["reason"]
        == "EXPERIMENT_ID_CONFLICT",
        "Experiment result conflict was accepted",
    )
    core.require(
        restarted.verify()
        and restarted.list_records() == records,
        "Registry recovery failed",
    )

    # Tamper test on a COPY of the entire store.
    # Original baseline evidence remains unchanged.
    tamper_store = output_dir / "tamper_probe"
    shutil.copytree(storage, tamper_store)

    altered_path = (
        tamper_store / "records"
        / f"{records[0]['experiment_id']}.json"
    )
    altered = core.read_json(altered_path)
    altered["result"]["accuracy"] = 0.5
    altered_path.write_text(
        json.dumps(altered, indent=2),
        encoding="utf-8",
    )

    tamper_detected = False
    try:
        core.ExperimentRegistry(tamper_store)
    except ValueError:
        tamper_detected = True

    core.require(
        tamper_detected,
        "Modified experiment record passed verification",
    )

    summary = {
        "module": 9,
        "chapter": "9.1",
        "mode": "durable_experiment_registry",
        "baseline_experiments": len(records),
        "registrations_accepted": sum(
            x["accepted"] for x in registrations
        ),
        "duplicate_rejections": sum(
            x["reason"] == "DUPLICATE_EXPERIMENT"
            for x in duplicates
        ),
        "conflicting_result_rejections": int(
            conflict_decision["reason"]
            == "EXPERIMENT_ID_CONFLICT"
        ),
        "record_tampering_detected": tamper_detected,
        "restart_verification": True,
        "audit_entries": len(restarted.audit),
        "frozen_detector_sha256": handoff["rule_sha256"],
        "registered_dataset_names": [
            r["configuration"]["dataset"]
            for r in records
        ],
        "source_module_8_sha256":
            handoff["final_sha256"],
        "audit_head": restarted.audit_head(),
        "baseline_replaced": False,
        "candidate_models_activated": 0,
        "permission_changes": 0,
        "network_requests": 0,
        "model_calls": 0,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    findings_dir.mkdir(parents=True, exist_ok=True)

    core.write_new_json(
        output_dir / "m9_1_summary.json",
        summary,
    )
    core.write_new_json(
        output_dir / "manifest.json",
        {
            "module": 9,
            "chapter": "9.1",
            "registry_head": restarted.audit_head(),
            "record_ids": [
                r["experiment_id"] for r in records
            ],
            "summary_sha256": core.digest(summary),
        },
    )

    findings = [
        "# Guardian AI — Module 9.1 Findings",
        "",
        "## Immutable baseline registration",
        "",
        "- Frozen detector baseline experiments: 2",
        "- Original fixture dataset and robustness",
        "  dataset are registered separately.",
        "- Replay registrations are rejected.",
        "- Conflicting results cannot replace records.",
        "- Reopening verifies the persisted audit chain.",
        "",
        "## Security and durability limitations",
        "",
        "- Single-writer local-file research prototype",
        "- No external audit anchoring",
        "- No transactionally atomic multi-file commit",
        "- An operator controlling the filesystem",
        "  can rewrite the entire local history",
        "- Source evidence remains synthetic",
        "",
        "## Module 9.2",
        "",
        "Build idempotent durable lesson memory with",
        "explicit lifecycle states and rollback.",
    ]

    findings_path = (
        findings_dir / "M9_C1_FINDINGS.md"
    )
    findings_path.write_text(
        "\n".join(findings) + "\n",
        encoding="utf-8",
    )

    print("\nFINAL MODULE 9 C1 SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nRegistry:", storage)
    print("Findings:", findings_path)

    return {
        "summary": summary,
        "records": records,
    }


if __name__ == "__main__":
    run_benchmark()
