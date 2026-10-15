
"""
Guardian AI — Module 9.2 lifecycle benchmark.

Production-derived candidates:
    8 submitted
    8 remain PENDING_REVIEW
    0 activated

Synthetic fixture:
    PENDING_REVIEW -> APPROVED
    -> ACTIVE -> ROLLED_BACK

No live Guardian memory or permissions change.
"""

import copy
import json
import shutil
from pathlib import Path

import guardian_lesson_lifecycle as core

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def run_benchmark(
    output_dir=RESULTS,
    findings_dir=HERE,
):
    output_dir = Path(output_dir)
    findings_dir = Path(findings_dir)

    primary = output_dir / "pending_lesson_store"
    fixture_path = output_dir / "fixture_lifecycle_store"
    tamper_path = output_dir / "tamper_probe"

    core.require(
        not any(p.exists() for p in (
            primary, fixture_path, tamper_path
        )),
        "Benchmark stores already exist; use a new output directory",
    )

    candidates = core.trusted_candidates()
    core.require(
        len(candidates) == 8,
        "Expected eight Module 8 candidates",
    )

    journal = core.LessonJournal(primary)

    for index, candidate in enumerate(candidates, 1):
        lesson = core.safe_lesson(candidate)

        result = journal.command(
            "SUBMIT",
            lesson,
            "trusted_importer",
            f"module8-submit-{index}",
        )
        core.require(
            result["accepted"],
            "Candidate import failed",
        )

    first = core.safe_lesson(candidates[0])

    replay = journal.command(
        "SUBMIT",
        first,
        "trusted_importer",
        "module8-submit-1",
    )
    conflict = journal.command(
        "SUBMIT",
        {
            **first,
            "lesson_text": "Modified lesson text",
        },
        "trusted_importer",
        "module8-submit-1",
    )

    rejected_approval = journal.command(
        "APPROVE",
        first,
        "trusted_reviewer_fixture",
        "unverified-approval",
        {
            "trusted_review": True,
            "independent_validation": True,
            "experiment_id": "no_independent_evidence",
        },
    )

    pending_summary = journal.summary()

    reopened = core.LessonJournal(primary)
    core.require(
        reopened.summary() == pending_summary,
        "Pending lesson restart verification failed",
    )

    # Explicitly isolated internal fixture test.
    test_lesson = core.make_fixture_lesson()
    fixture = core.LessonJournal(
        fixture_path,
        allow_fixture_activation=True,
    )

    evidence = {
        "trusted_review": True,
        "independent_validation": True,
        "experiment_id": "synthetic_review_fixture_only",
    }

    transitions = [
        fixture.command(
            "SUBMIT", test_lesson,
            "trusted_importer", "fixture-submit",
        ),
        fixture.command(
            "APPROVE", test_lesson,
            "trusted_reviewer_fixture",
            "fixture-approve",
            evidence,
        ),
        fixture.command(
            "ACTIVATE", test_lesson,
            "trusted_reviewer_fixture",
            "fixture-activate",
        ),
        fixture.command(
            "ROLLBACK", test_lesson,
            "trusted_reviewer_fixture",
            "fixture-rollback",
        ),
    ]

    core.require(
        all(item["accepted"] for item in transitions),
        "Synthetic lifecycle transitions failed",
    )

    core.require(
        fixture.state(test_lesson["lesson_id"])
        == "ROLLED_BACK",
        "Synthetic rollback failed",
    )

    reopened_fixture = core.LessonJournal(
        fixture_path,
        allow_fixture_activation=True,
    )
    core.require(
        reopened_fixture.verify()
        and reopened_fixture.state(
            test_lesson["lesson_id"]
        ) == "ROLLED_BACK",
        "Lifecycle restart verification failed",
    )

    # Attack the copied journal, not source history.
    shutil.copytree(primary, tamper_path)
    altered_file = (
        tamper_path / "lesson_journal.json"
    )

    altered = core.read_json(altered_file)
    altered[0]["lesson_id"] = "forged_lesson"

    altered_file.write_text(
        json.dumps(altered, indent=2),
        encoding="utf-8",
    )

    tamper_detected = False
    try:
        core.LessonJournal(tamper_path)
    except ValueError:
        tamper_detected = True

    core.require(
        tamper_detected,
        "Altered journal was accepted",
    )

    core.require(
        replay["reason"] == "IDEMPOTENT_REPLAY"
        and conflict["reason"] == "IDEMPOTENCY_CONFLICT"
        and rejected_approval["reason"]
        == "EVIDENCE_NOT_INDEPENDENTLY_VERIFIED",
        "Safety decisions did not match expectations",
    )

    core.require(
        pending_summary["pending_review"] == 8
        and pending_summary["approved"] == 0
        and pending_summary["active"] == 0,
        "Module 8 lesson was activated",
    )

    summary = {
        "module": 9,
        "chapter": "9.2",
        "mode": "durable_lesson_lifecycle",
        "imported_candidates": len(candidates),
        "pending_review": pending_summary["pending_review"],
        "approved_module_8_lessons": 0,
        "active_module_8_lessons": 0,
        "idempotent_replays_detected": 1,
        "idempotency_conflicts_rejected": 1,
        "unverified_approvals_rejected": 1,
        "synthetic_fixture_transitions": 4,
        "synthetic_fixture_final_state":
            reopened_fixture.state(
                test_lesson["lesson_id"]
            ),
        "fixture_rollback_verified": True,
        "journal_tampering_detected": tamper_detected,
        "restart_verifications": 2,
        "pending_journal_events":
            pending_summary["audit_events"],
        "fixture_journal_events":
            reopened_fixture.summary()["audit_events"],
        "guardian_permissions_changed": 0,
        "live_guardian_lessons_activated": 0,
        "network_requests": 0,
        "model_calls": 0,
    }

    core.atomic_json(
        output_dir / "m9_2_summary.json",
        summary,
    )

    core.atomic_json(
        output_dir / "manifest.json",
        {
            "module": 9,
            "chapter": "9.2",
            "summary_sha256": core.digest(summary),
            "pending_audit_head":
                pending_summary["audit_head"],
            "fixture_audit_head":
                reopened_fixture.summary()["audit_head"],
        },
    )

    findings_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (findings_dir / "M9_C2_FINDINGS.md").write_text(
        "\n".join([
            "# Guardian AI — Module 9.2 Findings",
            "",
            "Eight controller-authored Module 8 candidates",
            "were imported into a persistent review journal.",
            "All eight remain pending.",
            "",
            "The isolated synthetic fixture completed",
            "approval, activation and rollback transitions.",
            "No fixture was connected to live Guardian.",
            "",
            "Idempotent replays, command-key conflicts,",
            "unauthorized approval, and journal tampering",
            "were tested.",
            "",
            "## Limitations",
            "",
            "- Single-writer local journal",
            "- Review allowlist is not authentication",
            "- Synthetic review evidence is not real validation",
            "- Audit hashes have no independent anchor",
            "- ACTIVE is not a live deployment mechanism",
            "- No behavioral safety benefit tested",
            "",
        ]) + "\n",
        encoding="utf-8",
    )

    print("\nFINAL MODULE 9 C2 SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nSaved:", output_dir)

    return summary


if __name__ == "__main__":
    run_benchmark()
