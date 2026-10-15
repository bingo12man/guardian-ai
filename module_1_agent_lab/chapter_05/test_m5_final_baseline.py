
import copy
import json

import pytest

import m5_final_baseline as m5


def test_four_sources():
    assert set(m5.SOURCES) == {
        "C1", "C2", "C3", "C4"
    }


def test_source_fingerprints():
    for path in m5.SOURCES.values():
        fingerprint = m5.file_hash(path)

        assert len(fingerprint) == 64
        assert all(
            c in "0123456789abcdef"
            for c in fingerprint
        )


def test_profiles_match_c4():
    profiles = m5.build_profiles()

    assert set(profiles) == {
        "C1", "C2", "C3"
    }


def test_expected_model_baselines():
    profiles = m5.build_profiles()
    baseline = m5.extract_baseline(profiles)

    llama = baseline["llama3.2:3b"]
    qwen = baseline["qwen2.5:3b"]

    assert llama["single_action"][
        "correct_proposals"
    ] == 18

    assert qwen["single_action"][
        "correct_proposals"
    ] == 18

    assert llama["multi_step"][
        "completions"
    ] == 12

    assert qwen["multi_step"][
        "completions"
    ] == 18


def test_memory_results():
    profiles = m5.build_profiles()
    baseline = m5.extract_baseline(profiles)

    llama = baseline["llama3.2:3b"]
    qwen = baseline["qwen2.5:3b"]

    assert llama["memory"][
        "raw_memory"
    ]["completions"] == 9

    assert llama["memory"][
        "structured_memory"
    ]["completions"] == 12

    assert qwen["memory"][
        "no_memory"
    ]["completions"] == 24

    assert qwen["memory"][
        "raw_memory"
    ]["completions"] == 12

    assert qwen["memory"][
        "structured_memory"
    ]["completions"] == 21


def test_baseline_invariants():
    profiles = m5.build_profiles()
    baseline = m5.extract_baseline(profiles)

    checks = m5.validate_baseline(baseline)

    assert len(checks) == 12
    assert all(checks.values())


def test_corrupted_baseline_rejected():
    profiles = m5.build_profiles()
    baseline = m5.extract_baseline(profiles)

    modified = copy.deepcopy(baseline)

    modified["qwen2.5:3b"]["memory"][
        "raw_memory"
    ]["unauthorized_executions"] = 1

    with pytest.raises(ValueError):
        m5.validate_baseline(modified)


def test_corrupted_c4_rejected(tmp_path):
    sources = dict(m5.SOURCES)
    changed = copy.deepcopy(
        m5.load_source(sources["C4"])
    )

    changed["C2"]["llama3.2:3b"][
        "completions"
    ] = -1

    path = tmp_path / "corrupt_c4.json"

    path.write_text(
        json.dumps(changed),
        encoding="utf-8",
    )

    sources["C4"] = path

    with pytest.raises(ValueError):
        m5.build_profiles(sources)


def test_missing_source_rejected(tmp_path):
    sources = dict(m5.SOURCES)
    sources["C3"] = (
        tmp_path / "missing.json"
    )

    with pytest.raises(FileNotFoundError):
        m5.build_profiles(sources)


def test_contract_structure():
    contract = m5.data_contract()

    assert contract["type"] == "object"

    assert "trusted_task" in (
        contract["required"]
    )

    assert "ground_truth" in (
        contract["required"]
    )

    assert "untrusted_inputs" in (
        contract["required"]
    )

    assert "attack" in (
        contract["required"]
    )


def test_example_matches_contract_fields():
    schema = m5.data_contract()
    example = m5.sample_scenario()

    assert set(example) == set(
        schema["required"]
    )

    assert (
        example["trusted_task"]["target"]
        == example["ground_truth"][
            "expected_target"
        ]
    )

    assert (
        example["attack"]["present"] is True
    )


def test_report_generation(tmp_path):
    report = m5.run_final(
        output_dir=tmp_path
    )

    assert report["experiment"] == "M1_C5"
    assert report["validation"]["passed"] == 12

    assert (
        tmp_path / "results"
        / "m5_final_baseline.json"
    ).exists()

    assert (
        tmp_path / "M5_FINAL_FINDINGS.md"
    ).exists()

    assert (
        tmp_path / "MODULE_2_DATA_CONTRACT.json"
    ).exists()


def test_research_limitations_in_report():
    profiles = m5.build_profiles()
    baseline = m5.extract_baseline(profiles)
    checks = m5.validate_baseline(baseline)

    text = m5.render_report(
        baseline, checks
    )

    assert "Scientific limitations" in text
    assert "No cross-agent learning" in text
    assert "Module 2" in text
