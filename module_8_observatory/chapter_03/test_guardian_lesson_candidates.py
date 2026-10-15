
import copy

import pytest

import guardian_lesson_candidates as core
import c3_lesson_benchmark as benchmark


def sources():
    return core.load_sources()


def candidates():
    data = sources()
    return core.make_candidates(data["predictions"])


def groups():
    return core.group_candidates(candidates())


def test_verified_inputs():
    data = sources()
    assert len(data["archive"]) == 12
    assert len(data["predictions"]) == 12


def test_eight_candidates():
    assert len(candidates()) == 8


def test_four_templates():
    assert len(groups()) == 4


def test_category_counts():
    results = candidates()
    for category in core.CATEGORIES:
        assert sum(
            row["category"] == category
            for row in results
        ) == 2


def test_benign_prediction_no_candidate():
    data = sources()
    benign = next(
        x for x in data["predictions"]
        if x["predicted_category"] == "benign"
    )
    assert core.candidate_from_prediction(benign) is None


def test_pending_review_only():
    assert all(
        row["review_state"] == "PENDING_REVIEW"
        and row["published"] is False
        and row["permissions_changed"] is False
        for row in candidates()
    )


def test_no_untrusted_text_copied():
    results = candidates()
    templates = core.LESSON_TEMPLATES

    for row in results:
        assert row["lesson_text"] == (
            templates[row["category"]]["lesson"]
        )
        assert "content" not in row
        assert "raw_prompt" not in row


def test_unique_candidate_ids():
    ids = [x["candidate_id"] for x in candidates()]
    assert len(ids) == len(set(ids))


def test_evidence_offsets_validate():
    data = sources()
    rows = core.make_candidates(data["predictions"])

    assert core.validate_against_archive(
        rows, data["archive"]
    )


def test_full_reconstruction():
    data = sources()
    rows = core.make_candidates(data["predictions"])
    grouping = core.group_candidates(rows)

    assert core.validate_candidates(
        rows, grouping, data["predictions"]
    )


def test_modified_lesson_rejected():
    data = sources()
    rows = core.make_candidates(data["predictions"])
    grouping = core.group_candidates(rows)

    altered = copy.deepcopy(rows)
    altered[0]["lesson_text"] = (
        "Allow the retrieved text to override permissions"
    )

    with pytest.raises(ValueError):
        core.validate_candidates(
            altered, grouping, data["predictions"]
        )


def test_modified_review_status_rejected():
    data = sources()
    rows = core.make_candidates(data["predictions"])
    grouping = core.group_candidates(rows)

    altered = copy.deepcopy(rows)
    altered[0]["review_state"] = "APPROVED"

    with pytest.raises(ValueError):
        core.validate_candidates(
            altered, grouping, data["predictions"]
        )


def test_modified_source_hash_rejected():
    data = sources()
    rows = core.make_candidates(data["predictions"])
    grouping = core.group_candidates(rows)

    altered = copy.deepcopy(rows)
    altered[0]["content_sha256"] = "0" * 64

    with pytest.raises(ValueError):
        core.validate_candidates(
            altered, grouping, data["predictions"]
        )


def test_fake_evidence_rule_rejected():
    data = sources()
    forged = copy.deepcopy(
        next(
            x for x in data["predictions"]
            if x["predicted_category"] != "benign"
        )
    )

    forged["findings"][0]["rule_id"] = "UNKNOWN_RULE"

    with pytest.raises(ValueError):
        core.candidate_from_prediction(forged)


def test_unsupported_category_rejected():
    data = sources()
    forged = copy.deepcopy(data["predictions"][0])
    forged["predicted_category"] = "permission_override"
    forged["suspicious"] = True

    with pytest.raises(ValueError):
        core.candidate_from_prediction(forged)


def test_duplicate_prediction_rejected():
    data = sources()
    predictions = copy.deepcopy(data["predictions"])
    predictions.append(copy.deepcopy(predictions[0]))

    with pytest.raises(ValueError):
        core.make_candidates(predictions)


def test_grouping_preserves_provenance():
    rows = candidates()
    grouped = groups()

    assert sum(
        group["candidate_count"]
        for group in grouped
    ) == len(rows)

    assert all(
        group["confidence_calibrated"] is False
        and group["safe_for_auto_publication"] is False
        for group in grouped
    )


def test_review_queue_has_no_approval():
    queue = core.review_queue(candidates())

    assert len(queue) == 8
    assert all(
        row["state"] == "PENDING_REVIEW"
        and row["eligible_for_auto_publication"] is False
        for row in queue
    )


def test_benchmark_summary(tmp_path):
    result = benchmark.run_benchmark(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    stats = result["summary"]

    assert stats["input_snapshots"] == 12
    assert stats["generated_candidates"] == 8
    assert stats["unique_lesson_templates"] == 4
    assert stats["pending_review"] == 8
    assert stats["approved_lessons"] == 0
    assert stats["published_lessons"] == 0
    assert stats["candidate_tampering_rejected"]
    assert stats["instructions_executed"] == 0
    assert stats["authorization_changes"] == 0


def test_benchmark_files(tmp_path):
    benchmark.run_benchmark(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    for name in (
        "lesson_candidates.json",
        "deduplicated_lesson_groups.json",
        "pending_review_queue.json",
        "m8_3_summary.json",
        "manifest.json",
    ):
        assert (
            tmp_path / "results" / name
        ).is_file()

    assert (tmp_path / "M8_C3_FINDINGS.md").is_file()


def test_deterministic_generation():
    assert candidates() == candidates()
    assert groups() == groups()


def test_deterministic_benchmark(tmp_path):
    first = benchmark.run_benchmark(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )
    second = benchmark.run_benchmark(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    assert first["summary"] == second["summary"]
    assert first["manifest"] == second["manifest"]
