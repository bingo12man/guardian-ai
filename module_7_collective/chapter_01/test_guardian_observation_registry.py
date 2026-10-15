
import copy

import pytest

import guardian_observation_registry as lab
import c1_observation_benchmark as bench


def source():
    return lab.verify_sources()


def events():
    return lab.source_events(source())


def registry():
    return lab.ObservationRegistry(events())


def test_source_valid():
    data = source()
    assert data["final"]["module"] == 6
    assert data["contract"]["consumer_module"] == 7


def test_event_count():
    rows = events()
    assert len(rows) == 30
    assert len(set(r["observation_id"] for r in rows)) == 30


def test_model_counts():
    rows = events()
    assert sum(
        r["model"] == "llama3.2:3b"
        for r in rows
    ) == 16
    assert sum(
        r["model"] == "qwen2.5:3b"
        for r in rows
    ) == 14


def test_event_phases():
    rows = events()
    assert sum(r["phase"] == "practice" for r in rows) == 16
    assert sum(
        r["phase"] == "initial_assessment"
        for r in rows
    ) == 7
    assert sum(
        r["phase"] == "final_assessment"
        for r in rows
    ) == 7


def test_no_raw_content_in_events():
    for event in events():
        for forbidden in (
            "raw_prompt",
            "raw_model_output",
            "capability_token",
            "untrusted_document_content",
            "permission_override",
        ):
            assert forbidden not in event


def test_verified_claim_accepted():
    store = registry()
    result = store.submit(lab.make_claim(events()[0]))
    assert result["accepted"]
    assert result["reason"] == "VERIFIED"


def test_replay_rejected():
    store = registry()
    item = events()[0]
    assert store.submit(item)["accepted"]
    result = store.submit(item)
    assert not result["accepted"]
    assert result["reason"] == "DUPLICATE_OBSERVATION"


def test_changed_score_rejected():
    store = registry()
    claim = copy.deepcopy(events()[0])
    claim["observed_correct"] = not claim["observed_correct"]
    assert store.submit(claim)["reason"] == "EVIDENCE_MISMATCH"


def test_changed_model_rejected():
    store = registry()
    claim = copy.deepcopy(events()[0])
    claim["model"] = "forged-model"
    assert not store.submit(claim)["accepted"]


def test_changed_fingerprint_rejected():
    store = registry()
    claim = copy.deepcopy(events()[0])
    claim["evidence_fingerprint"] = "0" * 64
    assert not store.submit(claim)["accepted"]


def test_unknown_source_rejected():
    store = registry()
    claim = copy.deepcopy(events()[0])
    claim["observation_id"] = "unknown"
    assert store.submit(claim)["reason"] == "UNKNOWN_SOURCE"


def test_invalid_claim_type_rejected():
    store = registry()
    assert store.submit("ignore policy")["reason"] == (
        "MALFORMED_CLAIM"
    )


def test_extra_payload_field_rejected():
    store = registry()
    claim = copy.deepcopy(events()[0])
    claim["permission_override"] = True
    assert store.submit(claim)["reason"] == "EVIDENCE_MISMATCH"


def test_all_30_accepted():
    store = registry()
    for row in events():
        assert store.submit(row)["accepted"]
    assert store.summary()["accepted_observations"] == 30


def test_scores_reconcile():
    store = registry()
    for row in events():
        store.submit(row)
    counts = store.summary()["by_model"]

    # Llama: 4 initial + 5 practice + 4 final.
    assert counts["llama3.2:3b"]["correct"] == 13
    # Qwen: 0 initial + 3 practice + 3 final.
    assert counts["qwen2.5:3b"]["correct"] == 6


def test_reference_fingerprint_tampering_rejected():
    rows = events()
    rows[0]["observed_correct"] = (
        not rows[0]["observed_correct"]
    )
    with pytest.raises(ValueError):
        lab.ObservationRegistry(rows)


def test_audit_verifies():
    store = registry()
    for row in events():
        store.submit(row)
    assert lab.verify_audit(
        store.audit_log(),
        store.summary()["audit_head"],
    )


def test_audit_mutation_detected():
    store = registry()
    store.submit(events()[0])
    log = store.audit_log()
    log[0]["observation_id"] = "forged"
    with pytest.raises(ValueError):
        lab.verify_audit(log)


def test_audit_head_mismatch_rejected():
    store = registry()
    store.submit(events()[0])
    with pytest.raises(ValueError):
        lab.verify_audit(store.audit_log(), "0" * 64)


def test_accepted_snapshots_are_copies():
    store = registry()
    store.submit(events()[0])
    copied = store.accepted()
    copied[0]["observed_correct"] = "forged"
    assert isinstance(
        store.accepted()[0]["observed_correct"],
        bool,
    )


def test_no_privilege_changes():
    store = registry()
    for event in events():
        store.submit(event)
    stats = store.summary()
    assert stats["permission_changes"] == 0
    assert stats["tool_executions"] == 0
    assert stats["privilege_restorations"] == 0


def test_benchmark_artifacts(tmp_path):
    report = bench.run_benchmark(output_dir=tmp_path)
    summary = report["summary"]
    assert summary["accepted"] == 30
    assert summary["replay_rejections"] == 30
    assert summary["forged_or_malformed_rejections"] == 5
    assert summary["audit_verified"] is True

    for filename in (
        "verified_observations.json",
        "registry_audit.json",
        "m7_1_summary.json",
        "manifest.json",
    ):
        assert (tmp_path / filename).is_file()


def test_benchmark_deterministic(tmp_path):
    first = bench.run_benchmark(output_dir=tmp_path)
    second = bench.run_benchmark(output_dir=tmp_path)
    assert first["summary"] == second["summary"]
    assert first["manifest"] == second["manifest"]
