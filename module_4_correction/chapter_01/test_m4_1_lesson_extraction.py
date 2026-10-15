
import copy
import json

import pytest

import m4_1_lesson_extraction as lab


def evidence():
    report, scenarios = lab.load_evidence()
    return report, scenarios


def extract_training_evidence(report, scenarios):
    rows = []

    for record in report["records"]:
        rows.extend(
            lab.classify_denials(
                record,
                scenarios[record["scenario_id"]],
            )
        )

    return rows


def test_full_live_source_loads():
    report, scenarios = evidence()

    assert report["mode"] == "live"
    assert len(report["records"]) == 120
    assert len(scenarios) == 60


def test_original_live_metrics():
    report, _ = evidence()

    llama = report["summary"]["by_model"][
        "llama3.2:3b"
    ]
    qwen = report["summary"]["by_model"][
        "qwen2.5:3b"
    ]

    assert llama["completions"] == 6
    assert llama["blocked_proposals"] == 207
    assert llama["wrong_target_proposals"] == 26

    assert qwen["completions"] == 25
    assert qwen["blocked_proposals"] == 103
    assert qwen["wrong_target_proposals"] == 21


def test_evidence_is_training_only():
    report, scenarios = evidence()
    rows = extract_training_evidence(
        report, scenarios
    )

    assert rows
    assert all(
        item["split"] == "train"
        for item in rows
    )


def test_no_raw_model_output_in_evidence():
    report, scenarios = evidence()
    rows = extract_training_evidence(
        report, scenarios
    )

    for row in rows:
        assert "raw" not in row
        assert "content" not in row
        assert "argument" not in row


def test_three_supported_failure_categories():
    assert set(lab.RULES) == {
        "schema_error",
        "target_mismatch",
        "workflow_scope_error",
    }


def test_all_lessons_use_trusted_rules():
    report, scenarios = evidence()
    rows = extract_training_evidence(
        report, scenarios
    )
    lessons = lab.make_lessons(rows)

    assert lessons

    for lesson in lessons:
        rule = lab.RULES[
            lesson["failure_category"]
        ]

        assert lesson["violated_rule"] == (
            rule["rule"]
        )
        assert lesson["safe_correction"] == (
            rule["correction"]
        )
        assert lesson["authorization_change"] is False


def test_no_test_or_validation_leakage():
    report, scenarios = evidence()
    rows = extract_training_evidence(
        report, scenarios
    )
    lessons = lab.make_lessons(rows)

    train_ids = {
        scenario_id
        for scenario_id, scenario
        in scenarios.items()
        if scenario["split"] == "train"
    }

    for lesson in lessons:
        assert set(
            lesson["supporting_scenario_ids"]
        ) <= train_ids


def test_validation_evidence_not_extracted():
    report, scenarios = evidence()

    record = next(
        row for row in report["records"]
        if row["split"] == "validation"
    )

    assert lab.classify_denials(
        record,
        scenarios[record["scenario_id"]],
    ) == []


def test_lesson_merging_is_deterministic():
    report, scenarios = evidence()
    rows = extract_training_evidence(
        report, scenarios
    )

    first = lab.make_lessons(rows)
    second = lab.make_lessons(
        list(reversed(rows))
    )

    assert first == second
    assert lab.digest(first) == lab.digest(second)


def test_unknown_category_rejected():
    rows = [{
        "split": "train",
        "category": "fabricated_rule",
        "stage": "need_search",
        "proposed_tool": "read_document",
        "expected_tool": "search_documents",
        "model": "test",
        "scenario_id": "public_lookup_01",
        "attack_present": True,
    }]

    with pytest.raises(KeyError):
        lab.make_lessons(rows)


def test_permission_expansion_rejected():
    report, scenarios = evidence()
    rows = extract_training_evidence(
        report, scenarios
    )
    lessons = lab.make_lessons(rows)

    corrupted = copy.deepcopy(lessons)
    corrupted[0]["authorization_change"] = True

    with pytest.raises(ValueError):
        lab.validate_lessons(corrupted)


def test_evaluation_scenario_in_lesson_rejected():
    report, scenarios = evidence()
    rows = extract_training_evidence(
        report, scenarios
    )
    lessons = lab.make_lessons(rows)

    corrupted = copy.deepcopy(lessons)
    test_id = next(
        scenario_id
        for scenario_id, scenario
        in scenarios.items()
        if scenario["split"] == "test"
    )

    corrupted[0][
        "supporting_scenario_ids"
    ].append(test_id)

    with pytest.raises(ValueError):
        lab.validate_lessons(corrupted)


def test_corrupted_episode_counter_rejected():
    report, scenarios = evidence()

    original = next(
        row for row in report["records"]
        if row["model_calls"] > 0
    )

    corrupted = copy.deepcopy(original)
    corrupted["blocked_proposals"] += 1

    with pytest.raises(ValueError):
        lab.validate_episode(
            corrupted,
            scenarios[corrupted["scenario_id"]],
            report["max_steps"],
        )


def test_tampered_dataset_fingerprint_rejected(
    tmp_path,
):
    report, _ = evidence()

    corrupted = copy.deepcopy(report)
    corrupted["dataset_fingerprint"] = (
        "0" * 64
    )

    path = tmp_path / "bad_source.json"
    path.write_text(
        json.dumps(corrupted),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        lab.load_evidence(path)


def test_offline_evidence_rejected(tmp_path):
    report, _ = evidence()

    corrupted = copy.deepcopy(report)
    corrupted["mode"] = "offline"

    path = tmp_path / "offline.json"
    path.write_text(
        json.dumps(corrupted),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        lab.load_evidence(path)


def test_final_extraction(tmp_path, monkeypatch):
    monkeypatch.setattr(
        lab, "HERE", tmp_path
    )

    result = lab.run_extraction(
        output_dir=tmp_path / "results"
    )

    summary = result["summary"]

    assert summary["source_episodes"] == 120
    assert summary["train_episodes"] == 60
    assert summary["held_out_episodes"] == 60
    assert summary["candidate_lessons"] > 0
    assert summary["authorization_changes"] == 0
    assert summary["unauthorized_executions"] == 0

    assert (
        tmp_path / "results"
        / "candidate_lessons.json"
    ).exists()

    assert (
        tmp_path / "M4_C1_FINDINGS.md"
    ).exists()
