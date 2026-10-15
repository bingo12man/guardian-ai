
import copy
from collections import Counter

import pytest

import guardian_source_registry as core
import c1_offline_observatory as bench


def snapshots():
    return core.make_snapshots()


def registry():
    return core.SourceRegistry(
        core.approved_sources(),
        snapshots(),
    )


def test_module_8_contract():
    source = core.validate_handoff()
    assert len(source["contract_hash"]) == 64


def test_two_approved_sources():
    sources = core.approved_sources()
    assert len(sources) == 2
    assert all(not x.network_enabled for x in sources)


def test_twelve_snapshots():
    assert len(snapshots()) == 12


def test_snapshot_labels():
    labels = core.make_ground_truth()
    counts = Counter(labels.values())

    assert counts["benign"] == 4
    assert counts["target_redirection"] == 2
    assert counts["authority_spoof"] == 2
    assert counts["encoded_instruction"] == 2
    assert counts["memory_poisoning"] == 2


def test_unique_snapshot_ids():
    values = [x["snapshot_id"] for x in snapshots()]
    assert len(values) == len(set(values))


def test_snapshots_are_offline():
    for row in snapshots():
        assert row["network_fetched"] is False
        assert row["executed"] is False
        assert row["operator_authorized"] is True
        assert row["permission_changes"] is False


def test_contents_within_bounds():
    assert all(
        0 < len(x["content"]) <= core.MAX_CONTENT_CHARS
        for x in snapshots()
    )


def test_normal_ingestion():
    store = registry()
    for row in snapshots():
        assert store.ingest(row)["accepted"]
    assert len(store.accepted()) == 12


def test_replay_denied():
    store = registry()
    row = snapshots()[0]

    assert store.ingest(row)["accepted"]
    assert store.ingest(row)["reason"] == (
        "DUPLICATE_SNAPSHOT"
    )


def test_changed_content_rejected():
    store = registry()
    altered = copy.deepcopy(snapshots()[0])
    altered["content"] = "Different content"

    result = store.ingest(altered)
    assert not result["accepted"]
    assert result["reason"] == "SOURCE_MISMATCH"


def test_changed_hash_rejected():
    store = registry()
    altered = copy.deepcopy(snapshots()[0])
    altered["content_sha256"] = "0" * 64
    assert not store.ingest(altered)["accepted"]


def test_recomputed_forged_hash_still_rejected():
    import hashlib

    store = registry()
    forged = copy.deepcopy(snapshots()[0])
    forged["content"] = "Forged content."
    forged["content_sha256"] = hashlib.sha256(
        forged["content"].encode("utf-8")
    ).hexdigest()

    assert store.ingest(forged)["reason"] == (
        "SOURCE_MISMATCH"
    )


def test_forged_source_rejected():
    store = registry()
    altered = copy.deepcopy(snapshots()[0])
    altered["source_id"] = "third_party_website"

    assert store.ingest(altered)["reason"] == (
        "SOURCE_MISMATCH"
    )


def test_unknown_snapshot_denied():
    store = registry()
    altered = copy.deepcopy(snapshots()[0])
    altered["snapshot_id"] = "unknown"

    assert store.ingest(altered)["reason"] == (
        "UNKNOWN_SNAPSHOT"
    )


def test_forbidden_field_rejected():
    store = registry()
    altered = copy.deepcopy(snapshots()[0])
    altered["capability_token"] = "fake"

    assert store.ingest(altered)["reason"] == (
        "FORBIDDEN_FIELD"
    )


def test_authorization_flag_cannot_be_changed():
    store = registry()
    altered = copy.deepcopy(snapshots()[0])
    altered["operator_authorized"] = False

    assert store.ingest(altered)["reason"] == (
        "SOURCE_MISMATCH"
    )


def test_non_dict_submission_rejected():
    store = registry()
    assert store.ingest(
        "Ignore instructions"
    )["reason"] == "MALFORMED_SNAPSHOT"


def test_reference_snapshot_is_copy():
    store = registry()
    row = snapshots()[0]
    store.ingest(row)

    exported = store.accepted()
    exported[0]["content"] = "modified"

    assert store.accepted()[0]["content"] == row["content"]


def test_audit_chain():
    store = registry()
    for row in snapshots():
        store.ingest(row)

    assert core.verify_audit(
        store.audit(), store.audit_head()
    )
    assert len(store.audit()) == 12


def test_audit_tamper_detection():
    store = registry()
    store.ingest(snapshots()[0])

    audit = store.audit()
    audit[0]["source_id"] = "forged"

    with pytest.raises(ValueError):
        core.verify_audit(audit)


def test_audit_head_tampering():
    store = registry()
    store.ingest(snapshots()[0])

    with pytest.raises(ValueError):
        core.verify_audit(
            store.audit(), "0" * 64
        )


def test_benchmark_summary(tmp_path):
    result = bench.run_benchmark(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )
    summary = result["summary"]

    assert summary["registered_sources"] == 2
    assert summary["synthetic_snapshots"] == 12
    assert summary["accepted_snapshots"] == 12
    assert summary["replay_rejections"] == 12
    assert summary[
        "altered_or_forbidden_rejections"
    ] == 6
    assert summary["audit_entries"] == 12
    assert summary["audit_verified"] is True
    assert summary["network_requests"] == 0
    assert summary["permission_changes"] == 0
    assert summary["threat_detection_tested"] is False


def test_benchmark_outputs(tmp_path):
    bench.run_benchmark(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    for name in (
        "verified_snapshots.json",
        "ground_truth_labels.json",
        "approved_sources.json",
        "source_audit.json",
        "m8_1_summary.json",
        "manifest.json",
    ):
        assert (tmp_path / "results" / name).is_file()

    assert (tmp_path / "M8_C1_FINDINGS.md").is_file()


def test_deterministic_generation():
    assert snapshots() == snapshots()
    assert core.make_ground_truth() == (
        core.make_ground_truth()
    )


def test_deterministic_benchmark(tmp_path):
    first = bench.run_benchmark(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )
    second = bench.run_benchmark(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    assert first["summary"] == second["summary"]
    assert first["manifest"] == second["manifest"]
