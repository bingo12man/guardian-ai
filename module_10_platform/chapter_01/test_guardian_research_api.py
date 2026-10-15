
import copy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import guardian_research_api as api
import c1_api_benchmark as bench


@pytest.fixture
def client():
    return TestClient(api.app)


def test_evidence_loads():
    data = api.load_verified_research()

    assert data["final"]["module"] == 9
    assert data["contract"]["consumer_module"] == 10


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "evidence_verified": True,
        "read_only": True,
    }


def test_overview(client):
    response = client.get("/api/v1/overview")

    assert response.status_code == 200

    data = response.json()
    assert data["project"] == "Guardian AI"
    assert data["frozen_baseline_experiments"] == 2
    assert data["candidate_detector_experiments"] == 3
    assert data["pending_lessons"] == 8
    assert data["active_lessons"] == 0
    assert data["permission_changes"] == 0


def test_detector_comparison(client):
    response = client.get("/api/v1/detectors")
    assert response.status_code == 200

    data = response.json()
    assert len(data["comparison_results"]) == 3
    assert data["candidate_promoted"] is False
    assert data["candidate_status"] == (
        "CANDIDATE_PENDING"
    )


def test_original_regression_display(client):
    data = client.get("/api/v1/detectors").json()

    original = next(
        row for row in data["comparison_results"]
        if row["dataset"] == "original"
    )

    assert original["v1_accuracy"] == 1.0
    assert original["v2_accuracy"] == 0.75
    assert original["delta_accuracy_pp"] == -25.0
    assert original["paired_losses"] == 3


def test_development_improvement_display(client):
    data = client.get("/api/v1/detectors").json()

    row = next(
        x for x in data["comparison_results"]
        if x["dataset"] == "development"
    )

    assert row["delta_accuracy_pp"] == 54.17


def test_fresh_improvement_display(client):
    data = client.get("/api/v1/detectors").json()

    row = next(
        x for x in data["comparison_results"]
        if x["dataset"] == "fresh"
    )

    assert row["v2_accuracy"] == 0.9167
    assert row["delta_accuracy_pp"] == 50.0


def test_lessons_are_pending(client):
    data = client.get("/api/v1/lessons").json()

    assert data["pending"] == 8
    assert data["approved"] == 0
    assert data["active"] == 0
    assert data["review_controls_exposed"] is False
    assert data["lesson_content_exposed"] is False


def test_release_decision(client):
    data = client.get("/api/v1/release").json()

    assert data["decision"] == "HOLD_FOR_REVIEW"
    assert data["gates_passed"] == 3
    assert data["gates_failed"] == 6
    assert data["candidate_promoted"] is False
    assert data["automatic_promotion_allowed"] is False


def test_no_post_approval(client):
    response = client.post(
        "/api/v1/approve",
        json={"lesson_id": "candidate", "approved": True},
    )
    assert response.status_code in (404, 405)


def test_no_detector_promotion(client):
    response = client.post(
        "/api/v1/promote",
        json={"detector": "v2"},
    )
    assert response.status_code in (404, 405)


def test_no_permission_change(client):
    response = client.post(
        "/api/v1/permissions",
        json={"role": "administrator"},
    )
    assert response.status_code in (404, 405)


def test_no_write_methods_on_data_routes(client):
    for path in (
        "/api/v1/overview",
        "/api/v1/detectors",
        "/api/v1/lessons",
        "/api/v1/release",
    ):
        assert client.post(
            path, json={"test": True}
        ).status_code == 405


def test_unknown_path(client):
    response = client.get("/api/v1/unknown")
    assert response.status_code == 404


def test_no_raw_attack_content(client):
    for path in bench.ENDPOINTS:
        data = client.get(path).json()

        for forbidden in (
            "content",
            "raw_model_output",
            "capability_token",
            "permission_override",
        ):
            # Check actual dictionary keys recursively
            # without treating ordinary prose as a secret.
            def check_keys(value):
                if isinstance(value, dict):
                    assert forbidden not in value
                    for child in value.values():
                        check_keys(child)
                elif isinstance(value, list):
                    for child in value:
                        check_keys(child)

            check_keys(data)


def test_contract_is_read_only():
    research = api.load_verified_research()
    contract = research["contract"]

    assert contract["integration_mode"] == (
        "read_only_research"
    )

    assert all(
        value is False
        for value in contract[
            "authorization_constraints"
        ].values()
    )


def test_verifier_rejects_invalid_final(
    monkeypatch,
    tmp_path,
):
    altered = api.read_json(api.FINAL_PATH)
    altered["release_decision"] = (
        "ELIGIBLE_FOR_SEPARATE_REVIEW"
    )

    forged_file = tmp_path / "final.json"
    forged_file.write_text(
        __import__("json").dumps(altered),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        api, "FINAL_PATH", forged_file
    )

    with pytest.raises(ValueError):
        api.load_verified_research()


def test_api_fails_closed_on_corrupt_final(
    monkeypatch,
    tmp_path,
    client,
):
    forged_file = tmp_path / "final.json"
    forged_file.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        api, "FINAL_PATH", forged_file
    )

    response = client.get("/api/v1/overview")

    assert response.status_code == 503
    assert response.json()["error"] == (
        "EVIDENCE_VERIFICATION_FAILED"
    )


def test_api_benchmark(tmp_path):
    report = bench.run_benchmark(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    summary = report["summary"]

    assert summary["get_endpoints_tested"] == 5
    assert summary["successful_get_endpoints"] == 5
    assert summary["prohibited_write_attempts"] == 4
    assert summary["write_attempts_rejected"] == 4
    assert summary["release_decision"] == (
        "HOLD_FOR_REVIEW"
    )
    assert summary["permission_changes"] == 0


def test_benchmark_files(tmp_path):
    bench.run_benchmark(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    for filename in (
        "api_responses.json",
        "m10_1_summary.json",
        "manifest.json",
    ):
        assert (
            tmp_path / "results" / filename
        ).is_file()

    assert (
        tmp_path / "M10_C1_FINDINGS.md"
    ).is_file()
