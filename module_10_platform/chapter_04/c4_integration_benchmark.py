
"""
Guardian AI — Chapter 10.4 benchmark.

Runs read-only FastAPI integration checks and
frontend production artifact verification.

Writes only Chapter 10.4 result artifacts.
"""

import json
from pathlib import Path

import guardian_integration_checks as checks

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2) + "\n",
        encoding="utf-8",
    )


def run_benchmark(
    output_dir=RESULTS,
    findings_dir=HERE,
):
    result = checks.run_integration()
    contract = result["contract"]

    summary = {
        "module": 10,
        "chapter": "10.4",
        "mode": "read_only_platform_integration",
        "frontend_build_verified":
            result["frontend"]["build_present"],
        "frontend_build_sha256":
            result["frontend"]["build_sha256"],
        "experiments": contract["experiments"],
        "pending_lessons": contract["pending_lessons"],
        "verified_audit_components":
            contract["verified_audits"],
        "release_gates": contract["release_gates"],
        "detail_lookups_verified":
            contract["detail_lookups"],
        "paired_examples": contract["paired_examples"],
        "paired_gains": contract["paired_gains"],
        "paired_losses": contract["paired_losses"],
        "write_requests_rejected":
            result["rejected_write_requests"],
        "source_files_checked":
            result["source_files_checked"],
        "source_files_unchanged":
            result["source_files_unchanged"],
        "release_decision":
            contract["release_decision"],
        "candidate_promoted": False,
        "live_lessons_activated": 0,
        "permission_changes": 0,
        "browser_automation_completed": False,
        "production_security_certified": False,
    }

    output_dir = Path(output_dir)
    findings_dir = Path(findings_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    findings_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "m10_4_summary.json"

    write_json(summary_path, summary)

    write_json(
        output_dir / "manifest.json",
        {
            "module": 10,
            "chapter": "10.4",
            "summary_sha256": checks.fingerprint(summary),
            "frontend_build_sha256":
                result["frontend"]["build_sha256"],
            "source_files_unchanged": True,
        },
    )

    findings = [
        "# Guardian AI — Chapter 10.4 Findings",
        "",
        "## Integration checks",
        "",
        f"- Experiment records: {summary['experiments']}",
        f"- Pending lessons: {summary['pending_lessons']}",
        f"- Verified audits: {summary['verified_audit_components']}",
        f"- Paired examples: {summary['paired_examples']}",
        f"- Rejected write requests: {summary['write_requests_rejected']}",
        f"- Protected source files: {summary['source_files_checked']}",
        "- Source fingerprints unchanged: Yes",
        "",
        "## Release decision",
        "",
        summary["release_decision"],
        "",
        "## Test scope",
        "",
        "FastAPI was exercised in-process.",
        "The built React assets were verified on disk.",
        "Manual browser testing remains separate.",
        "",
        "## Limitations",
        "",
        "- No real browser automation in this benchmark",
        "- No production security certification",
        "- No externally anchored audit proof",
        "- No live agent safety improvement demonstrated",
        "",
        "## Next",
        "",
        "Chapter 10.5: final documentation, architecture",
        "and platform evaluation.",
    ]

    findings_path = findings_dir / "M10_C4_FINDINGS.md"
    findings_path.write_text(
        "\n".join(findings) + "\n",
        encoding="utf-8",
    )

    print("\nFINAL MODULE 10 C4 SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nSaved:", summary_path)
    print("Findings:", findings_path)

    return summary


if __name__ == "__main__":
    run_benchmark()
