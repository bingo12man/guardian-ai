
"""
Guardian AI — Module 8.1 offline observatory benchmark.

12 synthetic snapshots.
12 accepted claims.
12 replay attempts.
6 deliberately invalid submissions.

No network or model calls.
"""

import copy
import json
from collections import Counter
from pathlib import Path

import guardian_source_registry as core

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def invalid_submissions(snapshots):
    changed_content = copy.deepcopy(snapshots[0])
    changed_content["content"] += " Altered text."

    changed_hash = copy.deepcopy(snapshots[1])
    changed_hash["content_sha256"] = "0" * 64

    forged_source = copy.deepcopy(snapshots[2])
    forged_source["source_id"] = "unknown_external_source"

    unauthorized_flag = copy.deepcopy(snapshots[3])
    unauthorized_flag["operator_authorized"] = False

    forbidden_field = copy.deepcopy(snapshots[4])
    forbidden_field["capability_token"] = "fake_token"

    unknown = copy.deepcopy(snapshots[5])
    unknown["snapshot_id"] = "snapshot_unknown"

    return [
        changed_content,
        changed_hash,
        forged_source,
        unauthorized_flag,
        forbidden_field,
        unknown,
    ]


def run_benchmark(
    output_dir=RESULTS,
    findings_dir=HERE,
):
    handoff = core.validate_handoff()
    sources = core.approved_sources()
    snapshots = core.make_snapshots()
    ground_truth = core.make_ground_truth()

    # Submit forged claims first to ensure
    # they're not merely rejected as duplicates.
    probe = core.SourceRegistry(sources, snapshots)
    attacks = [
        probe.ingest(item)
        for item in invalid_submissions(snapshots)
    ]

    registry = core.SourceRegistry(sources, snapshots)

    accepted = [
        registry.ingest(copy.deepcopy(row))
        for row in snapshots
    ]

    replays = [
        registry.ingest(copy.deepcopy(row))
        for row in snapshots
    ]

    core.require(
        all(item["accepted"] for item in accepted),
        "Expected source rejected",
    )
    core.require(
        all(not item["accepted"] for item in attacks),
        "Forged source accepted",
    )
    core.require(
        all(not item["accepted"] for item in replays),
        "Replay accepted",
    )

    core.verify_audit(
        registry.audit(),
        registry.audit_head(),
    )

    distribution = dict(sorted(
        Counter(ground_truth.values()).items()
    ))
    rejection_reasons = dict(sorted(
        Counter(
            item["reason"]
            for item in attacks + replays
        ).items()
    ))

    summary = {
        "module": 8,
        "chapter": "8.1",
        "mode": "offline_synthetic_observatory",
        "registered_sources": len(sources),
        "synthetic_snapshots": len(snapshots),
        "accepted_snapshots": len(registry.accepted()),
        "replay_rejections": sum(
            not item["accepted"] for item in replays
        ),
        "altered_or_forbidden_rejections": sum(
            not item["accepted"] for item in attacks
        ),
        "rejection_reasons": rejection_reasons,
        "ground_truth_distribution": distribution,
        "audit_entries": len(registry.audit()),
        "audit_verified": True,
        "network_requests": 0,
        "model_calls": 0,
        "instructions_executed": 0,
        "permission_changes": 0,
        "privilege_restorations": 0,
        "threat_detection_tested": False,
        "behavioral_safety_improvement_tested": False,
    }

    output_dir = Path(output_dir)
    findings_dir = Path(findings_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    findings_dir.mkdir(parents=True, exist_ok=True)

    archive_path = (
        output_dir / "verified_snapshots.json"
    )
    labels_path = (
        output_dir / "ground_truth_labels.json"
    )
    sources_path = (
        output_dir / "approved_sources.json"
    )
    audit_path = output_dir / "source_audit.json"
    summary_path = (
        output_dir / "m8_1_summary.json"
    )
    manifest_path = output_dir / "manifest.json"

    core.write_json(
        archive_path, registry.accepted()
    )
    core.write_json(labels_path, ground_truth)
    core.write_json(
        sources_path,
        [
            {
                "source_id": x.source_id,
                "source_type": x.source_type,
                "description": x.description,
                "authorization": x.authorization,
                "allowed_snapshot_ids":
                    list(x.allowed_snapshot_ids),
                "network_enabled": x.network_enabled,
            }
            for x in sources
        ],
    )
    core.write_json(audit_path, registry.audit())
    core.write_json(summary_path, summary)

    manifest = {
        "module": 8,
        "chapter": "8.1",
        "module_7_contract_sha256":
            handoff["contract_hash"],
        "module_7_final_sha256":
            handoff["final_hash"],
        "snapshots_sha256":
            core.fingerprint(registry.accepted()),
        "ground_truth_sha256":
            core.fingerprint(ground_truth),
        "audit_sha256":
            core.fingerprint(registry.audit()),
        "summary_sha256":
            core.fingerprint(summary),
    }
    core.write_json(manifest_path, manifest)

    findings = [
        "# Guardian AI — Module 8.1",
        "",
        "## Offline source registry",
        "",
        f"Registered sources: {len(sources)}",
        f"Accepted synthetic snapshots: "
        f"{len(registry.accepted())}",
        "Network requests: 0",
        "Permission changes: 0",
        "",
        "## Scope",
        "",
        "All data is locally generated and fictional.",
        "The archive records inert text and provenance.",
        "Labels are stored separately from content.",
        "",
        "## Research limitations",
        "",
        "- All examples are synthetic fixtures.",
        "- Source authorization is hardcoded for fixtures.",
        "- Fingerprints ensure reproducibility,",
        "  not authenticity of arbitrary external sources.",
        "- No detection accuracy has been measured.",
        "- No agent behavioral benefit has been tested.",
        "",
        "## Next chapter",
        "",
        "Classify verified snapshot content offline,",
        "record prediction provenance and compare",
        "predictions against withheld fixture labels.",
    ]

    findings_path = findings_dir / "M8_C1_FINDINGS.md"
    findings_path.write_text(
        "\n".join(findings) + "\n",
        encoding="utf-8",
    )

    print("\nFINAL MODULE 8 C1 SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nArchive:", archive_path)
    print("Labels:", labels_path)
    print("Manifest:", manifest_path)
    print("Findings:", findings_path)

    return {
        "summary": summary,
        "snapshots": registry.accepted(),
        "labels": ground_truth,
        "manifest": manifest,
    }


if __name__ == "__main__":
    run_benchmark()
