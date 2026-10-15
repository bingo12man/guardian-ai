
import copy
import json

import pytest

import m2_2_attack_dataset as lab


def scenario(name):
    return next(
        row for row in lab.make_dataset()
        if row["scenario_id"] == name
    )


def test_attack_taxonomy():
    assert set(lab.ATTACK_TYPES) == {
        "target_redirection",
        "authority_spoof",
        "encoded_instruction",
        "memory_poisoning",
    }


def test_dataset_size():
    records = lab.make_dataset()

    assert len(records) == 60

    assert sum(
        row["attack"]["present"]
        for row in records
    ) == 48


def test_dataset_splits():
    records = lab.make_dataset()

    counts = {
        split: sum(
            row["split"] == split
            for row in records
        )
        for split in (
            "train", "validation", "test"
        )
    }

    assert counts == {
        "train": 30,
        "validation": 10,
        "test": 20,
    }


def test_four_variants_per_original():
    records = lab.make_dataset()

    for original in lab.base.generate_dataset():
        variants = [
            row for row in records
            if row["scenario_id"].startswith(
                original["scenario_id"] + "__"
            )
        ]

        assert len(variants) == 4


def test_attacks_do_not_modify_original_task():
    originals = lab.base.generate_dataset()

    for original in originals:
        for technique in lab.ATTACK_TYPES:
            variant = lab.make_variant(
                original, technique
            )

            assert (
                variant["trusted_task"]
                == original["trusted_task"]
            )

            assert (
                variant["ground_truth"]
                == original["ground_truth"]
            )

            assert (
                variant["split"]
                == original["split"]
            )


def test_base_dataset_not_mutated():
    original = lab.base.generate_dataset()
    before = copy.deepcopy(original)

    lab.make_dataset()

    assert original == before


def test_attack_sources():
    original = lab.base.generate_dataset()[0]

    for technique, source in (
        lab.ATTACK_SOURCES.items()
    ):
        variant = lab.make_variant(
            original, technique
        )

        assert (
            variant["untrusted_inputs"][0]["source"]
            == source
        )


def test_encoded_instruction_is_base64():
    original = lab.base.generate_dataset()[0]

    variant = lab.make_variant(
        original,
        "encoded_instruction",
    )

    content = variant[
        "untrusted_inputs"
    ][0]["content"]

    assert "Base64" in content
    assert "decode" in content


def test_attack_target_differs():
    for original in lab.base.generate_dataset():
        wrong = lab.alternate_target(
            original["trusted_task"]["target"]
        )

        assert wrong != (
            original["trusted_task"]["target"]
        )
        assert wrong in lab.base.DOCUMENTS


def test_ground_truth_labels():
    for record in lab.make_dataset():
        assert lab.base.validate_scenario(
            record
        )


def test_noncanonical_attack_rejected():
    records = lab.make_dataset()

    variant = next(
        row for row in records
        if row["attack"]["present"]
    )

    variant["untrusted_inputs"][0][
        "content"
    ] = "Changed attack text"

    with pytest.raises(ValueError):
        lab.validate_attack_dataset(records)


def test_permission_tampering_rejected():
    records = lab.make_dataset()

    records[0]["ground_truth"][
        "allowed_resources"
    ] = ["confidential.txt"]

    with pytest.raises(ValueError):
        lab.validate_attack_dataset(records)


def test_duplicate_variant_rejected():
    records = lab.make_dataset()
    records[1]["scenario_id"] = (
        records[0]["scenario_id"]
    )

    with pytest.raises(ValueError):
        lab.validate_attack_dataset(records)


def test_wrong_target_probe_is_blocked():
    current = scenario(
        "public_lookup_01__target_redirection"
    )

    probe = lab.probe_scenario(current)

    assert (
        probe["scripted_attack_attempt"]
        is not None
    )
    assert not probe[
        "scripted_attack_attempt"
    ]["allowed"]
    assert not probe[
        "scripted_attack_attempt"
    ]["executed"]


def test_manager_task_scope_is_enforced():
    current = scenario(
        "confidential_lookup_02__authority_spoof"
    )

    probe = lab.probe_scenario(current)

    # A manager may read both files generally,
    # but not change this original task's target.
    assert not probe[
        "scripted_attack_attempt"
    ]["executed"]

    assert probe["completed"]


def test_unauthorized_original_never_completes():
    for current in lab.make_dataset():
        if current["ground_truth"][
            "original_task_authorized"
        ]:
            continue

        probe = lab.probe_scenario(current)

        assert not probe["completed"]
        assert probe["unauthorized_executions"] == 0


def test_clean_variants_complete_when_authorized():
    for current in lab.make_dataset():
        if current["attack"]["present"]:
            continue

        probe = lab.probe_scenario(current)

        assert probe["completed"] == (
            current["ground_truth"][
                "original_task_authorized"
            ]
        )


def test_all_policy_probes_safe():
    for current in lab.make_dataset():
        probe = lab.probe_scenario(current)

        assert probe["unauthorized_executions"] == 0


def test_expected_manifest(tmp_path):
    manifest = lab.run_generation(tmp_path)

    assert manifest["scenario_count"] == 60
    assert manifest["clean_scenarios"] == 12
    assert manifest["attack_scenarios"] == 48

    assert manifest["split_counts"] == {
        "train": 30,
        "validation": 10,
        "test": 20,
    }

    assert manifest[
        "scripted_attack_attempts"
    ] == 48

    assert manifest[
        "scripted_attack_attempts_executed"
    ] == 0

    assert manifest[
        "legitimate_completions"
    ] == 45

    assert manifest[
        "unauthorized_executions"
    ] == 0


def test_generated_jsonl(tmp_path):
    lab.run_generation(tmp_path)

    for name, count in [
        ("train.jsonl", 30),
        ("validation.jsonl", 10),
        ("test.jsonl", 20),
        ("scenarios_all.jsonl", 60),
        ("policy_probes.jsonl", 60),
    ]:
        path = tmp_path / name
        assert path.exists()

        lines = path.read_text(
            encoding="utf-8"
        ).splitlines()

        assert len(lines) == count
        assert all(
            isinstance(json.loads(line), dict)
            for line in lines
        )


def test_dataset_reproducibility(tmp_path):
    first = lab.run_generation(tmp_path)
    second = lab.run_generation(tmp_path)

    assert (
        first["dataset_sha256"]
        == second["dataset_sha256"]
    )
