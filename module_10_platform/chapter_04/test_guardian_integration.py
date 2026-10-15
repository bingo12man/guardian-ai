
import copy

import pytest
from fastapi.testclient import TestClient

import guardian_integration_checks as lab
import c4_integration_benchmark as benchmark


def test_frontend_build():
    result = lab.frontend_preflight()
    assert result["build_present"]
    assert result["javascript_assets"] >= 1
    assert result["css_assets"] >= 1
    assert len(result["build_sha256"]) == 64


def test_fingerprints_available():
    hashes = lab.evidence_fingerprints()
    assert len(hashes) >= 14
    assert all(len(value) == 64 for value in hashes.values())


def test_read_api_contract():
    with TestClient(lab.api.app) as client:
        result = lab.verify_read_contract(client)

    assert result["experiments"] == 5
    assert result["pending_lessons"] == 8
    assert result["verified_audits"] == 4
    assert result["release_gates"] == 9
    assert result["paired_examples"] == 48


def test_detail_lookups():
    with TestClient(lab.api.app) as client:
        result = lab.verify_read_contract(client)

    assert result["detail_lookups"] == 17


def test_paired_regression():
    with TestClient(lab.api.app) as client:
        result = lab.verify_read_contract(client)

    assert result["paired_gains"]["original"] == 0
    assert result["paired_losses"]["original"] == 3
    assert result["paired_gains"]["fresh"] == 6
    assert result["paired_losses"]["fresh"] == 0


def test_write_methods_rejected():
    with TestClient(lab.api.app) as client:
        rejected = lab.verify_write_protection(client)

    assert rejected == 24


def test_no_source_mutation():
    before = lab.evidence_fingerprints()

    with TestClient(lab.api.app) as client:
        lab.verify_read_contract(client)
        lab.verify_write_protection(client)

    after = lab.evidence_fingerprints()
    assert before == after


def test_release_decision_unchanged():
    with TestClient(lab.api.app) as client:
        response = client.get("/api/v1/release-gates")

    assert response.status_code == 200
    assert response.json()["decision"] == "HOLD_FOR_REVIEW"
    assert response.json()["candidate_promoted"] is False


def test_unknown_records_rejected():
    paths = (
        "/api/v1/experiments/unknown-id",
        "/api/v1/lesson-review/unknown-id",
        "/api/v1/audits/unknown-component",
        "/api/v1/paired-outcomes/unknown-dataset",
    )

    with TestClient(lab.api.app) as client:
        for path in paths:
            assert client.get(path).status_code == 404


def test_fail_closed_on_corrupt_report(monkeypatch, tmp_path):
    api = lab.api.c1

    altered_path = tmp_path / "forged_final.json"
    altered = api.read_json(api.FINAL_PATH)
    altered["release_decision"] = "APPROVED"

    altered_path.write_text(
        __import__("json").dumps(altered),
        encoding="utf-8",
    )

    monkeypatch.setattr(api, "FINAL_PATH", altered_path)

    with TestClient(lab.api.app) as client:
        response = client.get("/api/v1/overview")

    assert response.status_code == 503
    assert response.json()["error"] == (
        "EVIDENCE_VERIFICATION_FAILED"
    )


def test_integration_result():
    result = lab.run_integration()

    assert result["contract"]["experiments"] == 5
    assert result["contract"]["pending_lessons"] == 8
    assert result["rejected_write_requests"] == 24
    assert result["source_files_unchanged"]
    assert result["permission_changes"] == 0
    assert result["candidate_promoted"] is False


def test_benchmark(tmp_path):
    result = benchmark.run_benchmark(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    assert result["frontend_build_verified"] is True
    assert result["experiments"] == 5
    assert result["pending_lessons"] == 8
    assert result["detail_lookups_verified"] == 17
    assert result["write_requests_rejected"] == 24
    assert result["source_files_unchanged"] is True
    assert result["release_decision"] == "HOLD_FOR_REVIEW"


def test_generated_artifacts(tmp_path):
    benchmark.run_benchmark(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    assert (tmp_path / "results/m10_4_summary.json").is_file()
    assert (tmp_path / "results/manifest.json").is_file()
    assert (tmp_path / "M10_C4_FINDINGS.md").is_file()


def test_reproducible_benchmark(tmp_path):
    first = benchmark.run_benchmark(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )
    second = benchmark.run_benchmark(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    assert first == second
