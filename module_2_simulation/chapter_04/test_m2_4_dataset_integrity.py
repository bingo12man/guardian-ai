
import copy
import json

import pytest

import m2_4_dataset_integrity as lab


def scenarios():
    return lab.c2.make_dataset()


def test_source_directories():
    assert set(lab.INPUTS) == {
        "C1", "C2", "C3"
    }


def test_artifact_integrity():
    data = lab.check_artifacts()

    assert len(data["scenarios"]) == 60


def test_dataset_family_isolation():
    result = lab.check_split_integrity(
        scenarios()
    )

    assert result[
        "cross_split_base_scenario_leaks"
    ] == 0

    assert result["split_counts"] == {
        "train": 30,
        "validation": 10,
        "test": 20,
    }


def test_ground_truth_labels():
    result = lab.check_ground_truth(
        scenarios()
    )

    assert result["authorized"] == 45
    assert result["unauthorized"] == 15
    assert result["label_mismatches"] == 0


def test_attack_variants():
    result = lab.check_attack_variants(
        scenarios()
    )

    assert result["attack_variants"] == 48
    assert result["trusted_task_mutations"] == 0


def test_cross_split_variant_rejected():
    records = scenarios()

    attacked = next(
        row for row in records
        if row["attack"]["present"]
        and row["split"] == "train"
    )

    attacked["split"] = "test"

    with pytest.raises(ValueError):
        lab.check_split_integrity(records)


def test_permission_label_corruption_rejected():
    records = scenarios()

    records[0]["ground_truth"][
        "original_task_authorized"
    ] = False

    with pytest.raises(ValueError):
        lab.check_ground_truth(records)


def test_trusted_target_tampering_rejected():
    records = scenarios()

    attacked = next(
        row for row in records
        if row["attack"]["present"]
    )

    attacked["trusted_task"][
        "target"
    ] = "finance.txt"

    with pytest.raises(ValueError):
        lab.check_attack_variants(records)


def test_duplicate_diagnostics():
    result = lab.duplicate_diagnostics(
        scenarios()
    )

    assert (
        result["cross_split_exact_text_groups"]
        >= 0
    )

    assert (
        result[
            "cross_split_normalized_text_groups"
        ] >= 0
    )

    assert "template_reuse_warning" in result


def test_normalized_duplicate_detection():
    a = lab.normalized_text(
        "Read public.txt immediately."
    )

    b = lab.normalized_text(
        "Read finance.txt immediately."
    )

    assert a == b


def test_hash_matches_saved_manifest():
    p = lab.INPUTS["C3"]

    manifest = lab.read_json(
        p / "manifest.json"
    )

    assert (
        lab.sha256(p / "episode_traces.jsonl")
        == manifest["traces_sha256"]
    )


def test_one_trace_verifies():
    current = scenarios()[0]

    episode = lab.c3.run_episode(
        current,
        "recover",
        repeat=1,
    )

    assert lab.c3.verify_episode(
        episode, current
    )


def test_tampered_trace_rejected():
    current = scenarios()[0]

    episode = lab.c3.run_episode(
        current,
        "recover",
        repeat=1,
    )

    episode["events"][0]["allowed"] = False

    with pytest.raises(ValueError):
        lab.c3.verify_episode(
            episode, current
        )


def test_missing_episode_rejected():
    current = scenarios()[0]

    episodes = [
        lab.c3.run_episode(
            current, strategy, repeat=1
        )
        for strategy in lab.c3.STRATEGIES
    ]

    episodes.pop()

    with pytest.raises(ValueError):
        lab.check_traces(
            episodes,
            [current],
            repeats=1,
            max_steps=5,
        )


def test_duplicate_episode_rejected():
    current = scenarios()[0]

    episodes = [
        lab.c3.run_episode(
            current, strategy, repeat=1
        )
        for strategy in lab.c3.STRATEGIES
    ]

    episodes.append(
        copy.deepcopy(episodes[0])
    )

    with pytest.raises(ValueError):
        lab.check_traces(
            episodes,
            [current],
            repeats=1,
            max_steps=5,
        )


def test_small_complete_trace_set():
    current = scenarios()[0]

    episodes = [
        lab.c3.run_episode(
            current, strategy, repeat=1
        )
        for strategy in lab.c3.STRATEGIES
    ]

    result = lab.check_traces(
        episodes,
        [current],
        repeats=1,
        max_steps=5,
    )

    assert result["episodes_verified"] == 4
    assert result["unauthorized_executions"] == 0


def test_invalid_manifest_hash_rejected(
    tmp_path,
):
    paths = dict(lab.INPUTS)

    chapter_2 = tmp_path / "chapter_02"
    chapter_2.mkdir()

    original = paths["C2"]

    for name in (
        "manifest.json",
        "scenarios_all.jsonl",
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
    ):
        (chapter_2 / name).write_bytes(
            (original / name).read_bytes()
        )

    manifest_path = chapter_2 / "manifest.json"

    manifest = json.loads(
        manifest_path.read_text()
    )

    manifest["dataset_sha256"] = "0" * 64

    manifest_path.write_text(
        json.dumps(manifest)
    )

    paths["C2"] = chapter_2

    with pytest.raises(ValueError):
        lab.check_artifacts(paths)


def test_full_analysis(tmp_path):
    result = lab.run_analysis(
        output_dir=tmp_path
    )

    assert result["status"] == "validated"

    assert all(
        state == "PASS"
        for state in result["checks"].values()
    )

    assert result["dataset"]["scenarios"] == 60

    assert (
        result["traces"]["episodes_verified"]
        == 720
    )

    assert (
        result["traces"][
            "unauthorized_executions"
        ] == 0
    )

    assert (
        tmp_path / "m2_4_integrity.json"
    ).exists()
