
import pytest
from fastapi.testclient import TestClient

import guardian_research_detail_api as api
import c2_detail_api_benchmark as benchmark


@pytest.fixture
def client():
    return TestClient(api.app)


def test_source_verification():
    research, evidence = api.verified_snapshot()

    assert research["final"]["module"] == 9
    assert len(evidence["baseline"]["records"]) == 2
    assert len(evidence["candidate"]["records"]) == 3


def test_chapter_10_1_routes_preserved(client):
    routes = (
        "/health",
        "/api/v1/overview",
        "/api/v1/detectors",
        "/api/v1/lessons",
        "/api/v1/release",
    )

    for route in routes:
        assert client.get(route).status_code == 200


def test_experiment_list(client):
    response = client.get("/api/v1/experiments")

    assert response.status_code == 200
    data = response.json()

    assert data["total"] == 5
    assert len(data["experiments"]) == 5
    assert data["read_only"] is True


def test_baseline_filter(client):
    response = client.get(
        "/api/v1/experiments",
        params={"kind": "frozen_detector_baseline"},
    )

    assert response.status_code == 200
    assert response.json()["total"] == 2


def test_candidate_filter(client):
    response = client.get(
        "/api/v1/experiments",
        params={"kind": "candidate_detector"},
    )

    assert response.status_code == 200
    assert response.json()["total"] == 3


def test_invalid_experiment_filter(client):
    response = client.get(
        "/api/v1/experiments",
        params={"kind": "permission_override"},
    )

    assert response.status_code == 422


def test_experiment_details(client):
    records = client.get(
        "/api/v1/experiments"
    ).json()["experiments"]

    for record in records:
        response = client.get(
            "/api/v1/experiments/"
            + record["experiment_id"]
        )

        assert response.status_code == 200
        assert response.json() == record
        assert record["editable"] is False
        assert record["promoted"] is False


def test_unknown_experiment(client):
    assert client.get(
        "/api/v1/experiments/unknown"
    ).status_code == 404


def test_all_candidate_experiments_pending(client):
    records = client.get(
        "/api/v1/experiments",
        params={"kind": "candidate_detector"},
    ).json()["experiments"]

    assert len(records) == 3

    assert all(
        record["status"] == "CANDIDATE_PENDING"
        for record in records
    )


def test_lesson_review_list(client):
    response = client.get("/api/v1/lesson-review")

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 8
    assert data["pending"] == 8
    assert data["approved"] == 0
    assert data["active"] == 0
    assert data["approval_endpoint_available"] is False


def test_lesson_details(client):
    rows = client.get(
        "/api/v1/lesson-review"
    ).json()["candidates"]

    for row in rows:
        response = client.get(
            "/api/v1/lesson-review/"
            + row["candidate_id"]
        )

        assert response.status_code == 200
        assert response.json() == row
        assert row["approved"] is False
        assert row["active"] is False
        assert row["published"] is False


def test_lesson_content_not_exposed(client):
    rows = client.get(
        "/api/v1/lesson-review"
    ).json()["candidates"]

    for row in rows:
        assert "content" not in row
        assert "lesson_text" not in row
        assert "raw_model_output" not in row
        assert "capability_token" not in row


def test_unknown_lesson(client):
    assert client.get(
        "/api/v1/lesson-review/unknown"
    ).status_code == 404


def test_audit_list(client):
    response = client.get("/api/v1/audits")

    assert response.status_code == 200

    data = response.json()

    assert data["components"] == 4
    assert data["all_verified"] is True
    assert data["externally_anchored"] is False


def test_audit_details(client):
    rows = client.get(
        "/api/v1/audits"
    ).json()["audits"]

    for row in rows:
        response = client.get(
            "/api/v1/audits/" + row["component"]
        )

        assert response.status_code == 200
        assert response.json() == row
        assert row["verification"] == "VERIFIED"
        assert len(row["audit_head"]) == 64


def test_audit_counts(client):
    data = client.get("/api/v1/audits").json()

    counts = {
        x["component"]: x["audit_events"]
        for x in data["audits"]
    }

    assert counts == {
        "baseline_experiments": 2,
        "candidate_experiments": 3,
        "pending_lessons": 8,
        "fixture_lifecycle": 4,
    }


def test_unknown_audit(client):
    assert client.get(
        "/api/v1/audits/unknown"
    ).status_code == 404


def test_paired_datasets(client):
    for name, count in (
        ("original", 12),
        ("development", 24),
        ("fresh", 12),
    ):
        response = client.get(
            f"/api/v1/paired-outcomes/{name}"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["examples"] == count
        assert len(data["outcomes"]) == count
        assert data["contains_raw_content"] is False


def test_original_pair_regression(client):
    data = client.get(
        "/api/v1/paired-outcomes/original"
    ).json()

    assert data["gains"] == 0
    assert data["losses"] == 3
    assert data["accuracy_delta_pp"] == -25.0


def test_fresh_pair_improvement(client):
    data = client.get(
        "/api/v1/paired-outcomes/fresh"
    ).json()

    assert data["gains"] == 6
    assert data["losses"] == 0
    assert data["accuracy_delta_pp"] == 50.0


def test_unknown_dataset(client):
    assert client.get(
        "/api/v1/paired-outcomes/unknown"
    ).status_code == 404


def test_release_gates(client):
    response = client.get("/api/v1/release-gates")

    assert response.status_code == 200

    data = response.json()

    assert data["decision"] == "HOLD_FOR_REVIEW"
    assert data["passed"] == 3
    assert data["failed"] == 6
    assert len(data["gates"]) == 9
    assert data["candidate_promoted"] is False
    assert data["activation_available"] is False


def test_original_regression_gate_failed(client):
    data = client.get(
        "/api/v1/release-gates"
    ).json()

    gates = {
        x["gate"]: x["passed"]
        for x in data["gates"]
    }

    assert gates[
        "original_accuracy_non_regression"
    ] is False
    assert gates[
        "original_attack_recall_non_regression"
    ] is False


def test_write_methods_rejected(client):
    for route in (
        "/api/v1/experiments",
        "/api/v1/lesson-review",
        "/api/v1/audits",
        "/api/v1/release-gates",
    ):
        for method in (
            "POST", "PUT", "PATCH", "DELETE"
        ):
            response = client.request(
                method,
                route,
                json={"approved": True},
            )

            assert response.status_code in (404, 405)


def test_no_approval_route(client):
    assert client.post(
        "/api/v1/lesson-review/approve",
        json={"approved": True},
    ).status_code in (404, 405)


def test_benchmark(tmp_path):
    report = benchmark.run_benchmark(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    result = report["summary"]

    assert result["experiment_records"] == 5
    assert result["pending_lesson_records"] == 8
    assert result["verified_audit_components"] == 4
    assert result["paired_datasets"] == 3
    assert result["paired_examples"] == 48
    assert result["release_gates"] == 9
    assert result["write_methods_rejected"] == 16
    assert result["permission_changes"] == 0


def test_artifacts(tmp_path):
    benchmark.run_benchmark(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    for name in (
        "m10_2_summary.json",
        "response_manifest.json",
        "manifest.json",
    ):
        assert (
            tmp_path / "results" / name
        ).is_file()

    assert (
        tmp_path / "M10_C2_FINDINGS.md"
    ).is_file()
