
"""
Guardian AI — Module 10.1 API benchmark.

Uses an in-process FastAPI TestClient.

No external HTTP requests.
No Ollama calls.
No lesson activation.
No Guardian tool execution.
"""

import json
from pathlib import Path

from fastapi.testclient import TestClient

import guardian_research_api as api

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

ENDPOINTS = (
    "/health",
    "/api/v1/overview",
    "/api/v1/detectors",
    "/api/v1/lessons",
    "/api/v1/release",
)


def run_benchmark(
    output_dir=RESULTS,
    findings_dir=HERE,
):
    client = TestClient(api.app)

    responses = {}
    for endpoint in ENDPOINTS:
        response = client.get(endpoint)

        api.require(
            response.status_code == 200,
            f"Endpoint failed: {endpoint}",
        )

        responses[endpoint] = response.json()

    overview = responses["/api/v1/overview"]
    detectors = responses["/api/v1/detectors"]
    lessons = responses["/api/v1/lessons"]
    release = responses["/api/v1/release"]

    api.require(
        overview["frozen_baseline_experiments"] == 2
        and overview["candidate_detector_experiments"] == 3
        and overview["pending_lessons"] == 8
        and overview["active_lessons"] == 0,
        "Overview values mismatch",
    )

    api.require(
        len(detectors["comparison_results"]) == 3
        and detectors["candidate_promoted"] is False,
        "Detector response mismatch",
    )

    api.require(
        lessons["approved"] == 0
        and lessons["active"] == 0
        and lessons["review_controls_exposed"] is False,
        "Unsafe lesson response",
    )

    api.require(
        release["decision"] == "HOLD_FOR_REVIEW"
        and release["gates_passed"] == 3
        and release["gates_failed"] == 6
        and release["automatic_promotion_allowed"] is False,
        "Release decision mismatch",
    )

    prohibited_paths = (
        "/api/v1/approve",
        "/api/v1/activate",
        "/api/v1/promote",
        "/api/v1/permissions",
    )

    write_attempts = []

    for path in prohibited_paths:
        response = client.post(path, json={
            "approved": True,
            "permission_override": True,
        })

        api.require(
            response.status_code in (404, 405),
            "Unexpected write endpoint",
        )

        write_attempts.append({
            "path": path,
            "status": response.status_code,
            "accepted": False,
        })

    summary = {
        "module": 10,
        "chapter": "10.1",
        "mode": "verified_read_only_fastapi",
        "get_endpoints_tested": len(ENDPOINTS),
        "successful_get_endpoints": len(responses),
        "prohibited_write_attempts": len(write_attempts),
        "write_attempts_rejected": len(write_attempts),
        "verified_evidence": True,
        "baseline_experiments": 2,
        "candidate_experiments": 3,
        "pending_lessons": 8,
        "active_lessons": 0,
        "release_decision": "HOLD_FOR_REVIEW",
        "candidate_promoted": False,
        "permission_changes": 0,
        "model_calls": 0,
        "external_network_requests": 0,
        "live_guardian_connected": False,
    }

    output_dir = Path(output_dir)
    findings_dir = Path(findings_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    findings_dir.mkdir(parents=True, exist_ok=True)

    api_json = {
        "endpoint_responses": responses,
        "write_attempts": write_attempts,
    }

    (output_dir / "api_responses.json").write_text(
        json.dumps(api_json, indent=2) + "\n",
        encoding="utf-8",
    )

    (output_dir / "m10_1_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "module": 10,
        "chapter": "10.1",
        "api_responses_sha256": api.digest(api_json),
        "summary_sha256": api.digest(summary),
        "module_9_final_sha256":
            api.digest(api.read_json(api.FINAL_PATH)),
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    findings = [
        "# Guardian AI — Module 10.1 Findings",
        "",
        "## Read-only API",
        "",
        "Five GET routes returned verified research",
        "summaries from the Module 9 evidence chain.",
        "",
        "Four attempted write operations were rejected.",
        "",
        "## Safety boundaries",
        "",
        "- No lesson approval endpoint",
        "- No candidate detector promotion",
        "- No permission modification endpoint",
        "- No raw attack text in summary responses",
        "- No live Guardian executor integration",
        "",
        "## Limitations",
        "",
        "- Local single-user research prototype",
        "- No production authentication",
        "- No production deployment security review",
        "- Revalidation on every request adds latency",
        "- The evidence is predominantly synthetic",
        "",
        "## Next",
        "",
        "Chapter 10.2 will add provenance, experiment",
        "and audit inspection endpoints.",
    ]

    findings_dir.joinpath(
        "M10_C1_FINDINGS.md"
    ).write_text(
        "\n".join(findings) + "\n",
        encoding="utf-8",
    )

    print("\nFINAL MODULE 10 C1 SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nSaved:", output_dir)

    return {
        "summary": summary,
        "responses": responses,
        "manifest": manifest,
    }


if __name__ == "__main__":
    run_benchmark()
