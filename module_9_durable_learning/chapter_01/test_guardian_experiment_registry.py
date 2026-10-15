
import copy
import json
import shutil

import pytest

import guardian_experiment_registry as core
import c1_experiment_benchmark as bench


def records():
    return core.make_baseline_records(
        core.validate_module_8_handoff()
    )


def test_module_8_handoff():
    handoff = core.validate_module_8_handoff()
    assert len(handoff["final_sha256"]) == 64
    assert len(handoff["rule_sha256"]) == 64


def test_two_baseline_records():
    rows = records()
    assert len(rows) == 2
    assert all(
        row["status"] == "FROZEN_BASELINE"
        for row in rows
    )


def test_dataset_separation():
    rows = records()
    assert {
        row["configuration"]["dataset"]
        for row in rows
    } == {
        "original_fixtures",
        "robustness_decoys",
    }


def test_unique_experiment_ids():
    rows = records()
    ids = [row["experiment_id"] for row in rows]
    assert len(ids) == len(set(ids))


def test_record_validation():
    assert all(
        core.validate_record(row)
        for row in records()
    )


def test_result_does_not_change_identity():
    row = copy.deepcopy(records()[0])
    original_id = row["experiment_id"]
    row["result"]["accuracy"] = 0.0

    assert core.experiment_id(row) == original_id


def test_invalid_experiment_id_rejected():
    row = copy.deepcopy(records()[0])
    row["experiment_id"] = "forged"

    with pytest.raises(ValueError):
        core.validate_record(row)


def test_permission_override_field_rejected():
    row = copy.deepcopy(records()[0])
    row["permission_override"] = True

    with pytest.raises(ValueError):
        core.validate_record(row)


def test_register_two_records(tmp_path):
    store = core.ExperimentRegistry(tmp_path / "store")

    for row in records():
        assert store.register(row)["accepted"]

    assert len(store.list_records()) == 2
    assert store.verify()


def test_duplicate_registration_rejected(tmp_path):
    store = core.ExperimentRegistry(tmp_path / "store")
    row = records()[0]

    assert store.register(row)["accepted"]

    repeated = store.register(row)
    assert not repeated["accepted"]
    assert repeated["reason"] == "DUPLICATE_EXPERIMENT"


def test_conflicting_result_rejected(tmp_path):
    store = core.ExperimentRegistry(tmp_path / "store")
    original = records()[0]
    store.register(original)

    conflict = copy.deepcopy(original)
    conflict["result"]["accuracy"] = 0.0

    result = store.register(conflict)
    assert not result["accepted"]
    assert result["reason"] == "EXPERIMENT_ID_CONFLICT"


def test_restart_preserves_records(tmp_path):
    path = tmp_path / "store"
    store = core.ExperimentRegistry(path)

    for row in records():
        store.register(row)

    reopened = core.ExperimentRegistry(path)

    assert reopened.verify()
    assert reopened.list_records() == records()


def test_record_tampering_detected(tmp_path):
    path = tmp_path / "store"
    store = core.ExperimentRegistry(path)
    row = records()[0]

    store.register(row)

    record_path = (
        path / "records"
        / f"{row['experiment_id']}.json"
    )

    altered = core.read_json(record_path)
    altered["result"]["accuracy"] = 0.0

    record_path.write_text(
        json.dumps(altered), encoding="utf-8"
    )

    with pytest.raises(ValueError):
        core.ExperimentRegistry(path)


def test_audit_tampering_detected(tmp_path):
    path = tmp_path / "store"
    store = core.ExperimentRegistry(path)

    store.register(records()[0])

    audit_path = path / "audit.json"
    audit = core.read_json(audit_path)
    audit[0]["record_sha256"] = "0" * 64

    audit_path.write_text(
        json.dumps(audit), encoding="utf-8"
    )

    with pytest.raises(ValueError):
        core.ExperimentRegistry(path)


def test_unregistered_record_rejected(tmp_path):
    path = tmp_path / "store"
    store = core.ExperimentRegistry(path)
    store.register(records()[0])

    extra = path / "records/unregistered.json"
    extra.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError):
        core.ExperimentRegistry(path)


def test_missing_record_rejected(tmp_path):
    path = tmp_path / "store"
    store = core.ExperimentRegistry(path)

    row = records()[0]
    store.register(row)

    (
        path / "records"
        / f"{row['experiment_id']}.json"
    ).unlink()

    with pytest.raises(ValueError):
        core.ExperimentRegistry(path)


def test_baseline_cannot_be_candidate(tmp_path):
    row = copy.deepcopy(records()[0])
    row["status"] = "CANDIDATE_PENDING"

    with pytest.raises(ValueError):
        core.validate_record(row)


def test_experiment_deterministic():
    assert records() == records()


def test_benchmark_end_to_end(tmp_path):
    result = bench.run_benchmark(
        output_dir=tmp_path / "output",
        findings_dir=tmp_path,
    )

    stats = result["summary"]

    assert stats["baseline_experiments"] == 2
    assert stats["registrations_accepted"] == 2
    assert stats["duplicate_rejections"] == 2
    assert stats["conflicting_result_rejections"] == 1
    assert stats["record_tampering_detected"]
    assert stats["restart_verification"]
    assert stats["audit_entries"] == 2
    assert stats["candidate_models_activated"] == 0
    assert stats["permission_changes"] == 0


def test_benchmark_refuses_overwrite(tmp_path):
    output = tmp_path / "output"

    bench.run_benchmark(
        output_dir=output,
        findings_dir=tmp_path,
    )

    with pytest.raises(ValueError):
        bench.run_benchmark(
            output_dir=output,
            findings_dir=tmp_path,
        )


def test_no_network_or_model_calls(tmp_path):
    result = bench.run_benchmark(
        output_dir=tmp_path / "output",
        findings_dir=tmp_path,
    )

    stats = result["summary"]
    assert stats["network_requests"] == 0
    assert stats["model_calls"] == 0
    assert stats["permission_changes"] == 0
