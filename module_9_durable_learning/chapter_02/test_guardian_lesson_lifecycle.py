
import copy
import json

import pytest

import guardian_lesson_lifecycle as lab
import c2_lifecycle_benchmark as bench


def fixture():
    return lab.make_fixture_lesson()


def approval_evidence():
    return {
        "trusted_review": True,
        "independent_validation": True,
        "experiment_id": "synthetic_review_fixture_only",
    }


def store(tmp_path, enabled=False):
    return lab.LessonJournal(
        tmp_path / "journal",
        allow_fixture_activation=enabled,
    )


def submit(journal, key="submit-1"):
    return journal.command(
        "SUBMIT", fixture(),
        "trusted_importer", key,
    )


def approve(journal, key="approve-1"):
    return journal.command(
        "APPROVE", fixture(),
        "trusted_reviewer_fixture",
        key, approval_evidence(),
    )


def test_original_candidates():
    assert len(lab.trusted_candidates()) == 8


def test_candidate_projection():
    candidate = lab.trusted_candidates()[0]
    lesson = lab.safe_lesson(candidate)

    assert lab.validate_lesson(lesson)
    assert "content" not in lesson
    assert "raw_model_output" not in lesson


def test_new_store_is_empty(tmp_path):
    journal = store(tmp_path)

    assert journal.verify()
    assert journal.summary()["lessons"] == 0


def test_submission_pending(tmp_path):
    journal = store(tmp_path)

    assert submit(journal)["accepted"]
    assert journal.state(fixture()["lesson_id"]) == (
        "PENDING_REVIEW"
    )


def test_idempotent_replay(tmp_path):
    journal = store(tmp_path)

    submit(journal)
    result = submit(journal)

    assert result["reason"] == "IDEMPOTENT_REPLAY"
    assert journal.summary()["audit_events"] == 1


def test_idempotency_conflict(tmp_path):
    journal = store(tmp_path)
    submit(journal)

    changed = copy.deepcopy(fixture())
    changed["lesson_text"] = "Different text"

    result = journal.command(
        "SUBMIT", changed,
        "trusted_importer", "submit-1",
    )

    assert result["reason"] == "IDEMPOTENCY_CONFLICT"


def test_invalid_transition(tmp_path):
    journal = store(tmp_path)

    result = journal.command(
        "APPROVE", fixture(),
        "trusted_reviewer_fixture",
        "approve-before-submit",
        approval_evidence(),
    )

    assert result["reason"] == "INVALID_TRANSITION"


def test_unknown_reviewer_rejected(tmp_path):
    journal = store(tmp_path)
    submit(journal)

    result = journal.command(
        "APPROVE", fixture(),
        "untrusted_agent",
        "untrusted-approval",
        approval_evidence(),
    )

    assert result["reason"] == "REVIEWER_NOT_AUTHORIZED"


def test_insufficient_evidence_rejected(tmp_path):
    journal = store(tmp_path)
    submit(journal)

    result = journal.command(
        "APPROVE", fixture(),
        "trusted_reviewer_fixture",
        "bad-evidence",
        {"trusted_review": True},
    )

    assert result["reason"] == (
        "INSUFFICIENT_REVIEW_EVIDENCE"
    )


def test_fixture_approval(tmp_path):
    journal = store(tmp_path)
    submit(journal)

    assert approve(journal)["accepted"]
    assert journal.state(fixture()["lesson_id"]) == "APPROVED"


def test_default_activation_disabled(tmp_path):
    journal = store(tmp_path)
    submit(journal)
    approve(journal)

    result = journal.command(
        "ACTIVATE", fixture(),
        "trusted_reviewer_fixture",
        "activate-1",
    )

    assert result["reason"] == "ACTIVATION_DISABLED"
    assert journal.state(fixture()["lesson_id"]) == "APPROVED"


def test_isolated_fixture_activation(tmp_path):
    journal = store(tmp_path, enabled=True)

    submit(journal)
    approve(journal)

    result = journal.command(
        "ACTIVATE", fixture(),
        "trusted_reviewer_fixture",
        "activate-1",
    )

    assert result["accepted"]
    assert journal.state(fixture()["lesson_id"]) == "ACTIVE"


def test_rollback(tmp_path):
    journal = store(tmp_path, enabled=True)
    submit(journal)
    approve(journal)

    journal.command(
        "ACTIVATE", fixture(),
        "trusted_reviewer_fixture",
        "activate-1",
    )
    result = journal.command(
        "ROLLBACK", fixture(),
        "trusted_reviewer_fixture",
        "rollback-1",
    )

    assert result["accepted"]
    assert journal.state(fixture()["lesson_id"]) == (
        "ROLLED_BACK"
    )


def test_rolled_back_cannot_reactivate(tmp_path):
    journal = store(tmp_path, enabled=True)
    submit(journal)
    approve(journal)

    journal.command(
        "ACTIVATE", fixture(),
        "trusted_reviewer_fixture",
        "activate-1",
    )
    journal.command(
        "ROLLBACK", fixture(),
        "trusted_reviewer_fixture",
        "rollback-1",
    )

    result = journal.command(
        "ACTIVATE", fixture(),
        "trusted_reviewer_fixture",
        "activate-2",
    )

    assert result["reason"] == "INVALID_TRANSITION"


def test_module8_approval_blocked(tmp_path):
    journal = store(tmp_path, enabled=True)
    lesson = lab.safe_lesson(
        lab.trusted_candidates()[0]
    )

    assert journal.command(
        "SUBMIT", lesson,
        "trusted_importer", "import-1",
    )["accepted"]

    result = journal.command(
        "APPROVE", lesson,
        "trusted_reviewer_fixture",
        "approve-original",
        approval_evidence(),
    )

    assert result["reason"] == (
        "EVIDENCE_NOT_INDEPENDENTLY_VERIFIED"
    )
    assert journal.state(lesson["lesson_id"]) == (
        "PENDING_REVIEW"
    )


def test_reopen_preserves_state(tmp_path):
    journal = store(tmp_path)
    submit(journal)
    approve(journal)

    reopened = store(tmp_path)

    assert reopened.verify()
    assert reopened.state(fixture()["lesson_id"]) == "APPROVED"


def test_changed_journal_detected(tmp_path):
    journal = store(tmp_path)
    submit(journal)

    path = journal.path
    events = lab.read_json(path)
    events[0]["actor"] = "forged"

    path.write_text(
        json.dumps(events),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        store(tmp_path)


def test_changed_lesson_rejected(tmp_path):
    journal = store(tmp_path)
    submit(journal)

    changed = copy.deepcopy(fixture())
    changed["lesson_text"] = "Tampered"

    result = journal.command(
        "APPROVE", changed,
        "trusted_reviewer_fixture",
        "approve-tampered",
        approval_evidence(),
    )

    assert result["reason"] == "LESSON_CHANGED"


def test_invalid_key_rejected(tmp_path):
    journal = store(tmp_path)

    with pytest.raises(ValueError):
        journal.command(
            "SUBMIT", fixture(),
            "trusted_importer",
            "../unsafe",
        )


def test_benchmark(tmp_path):
    summary = bench.run_benchmark(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    assert summary["imported_candidates"] == 8
    assert summary["pending_review"] == 8
    assert summary["approved_module_8_lessons"] == 0
    assert summary["active_module_8_lessons"] == 0
    assert summary["fixture_rollback_verified"]
    assert summary["journal_tampering_detected"]
    assert summary["guardian_permissions_changed"] == 0


def test_benchmark_refuses_existing_store(tmp_path):
    output_dir = tmp_path / "results"

    bench.run_benchmark(
        output_dir=output_dir,
        findings_dir=tmp_path,
    )

    with pytest.raises(ValueError):
        bench.run_benchmark(
            output_dir=output_dir,
            findings_dir=tmp_path,
        )
