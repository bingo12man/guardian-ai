
"""
Guardian AI — Module 10.2 API benchmark.

Verifies:
- 5 experiment records
- 8 pending lesson records
- 4 audit summaries
- 3 paired evaluation datasets
- 9 release gates
- Unknown identities return 404
- Mutating HTTP methods are rejected

All requests run in-process via TestClient.
"""

import json
from pathlib import Path

from fastapi.testclient import TestClient

import guardian_research_detail_api as api

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def run_benchmark(
    output_dir=RESULTS,
    findings_dir=HERE,
):
    client = TestClient(api.app)

    endpoints = {
        "overview": "/api/v1/overview",
        "experiments": "/api/v1/experiments",
        "lessons": "/api/v1/lesson-review",
        "audits": "/api/v1/audits",
        "original": "/api/v1/paired-outcomes/original",
        "development": "/api/v1/paired-outcomes/development",
        "fresh": "/api/v1/paired-outcomes/fresh",
        "release": "/api/v1/release-gates",
    }

    responses = {}

    for name, route in endpoints.items():
        response = client.get(route)

        api.c1.require(
            response.status_code == 200,
            f"GET failed: {route}",
        )

        responses[name] = response.json()

    experiments = responses["experiments"]["experiments"]
    lessons = responses["lessons"]["candidates"]
    audits = responses["audits"]["audits"]
    release = responses["release"]

    api.c1.require(
        len(experiments) == 5
        and len({
            item["experiment_id"]
            for item in experiments
        }) == 5,
        "Experiment API mismatch",
    )

    api.c1.require(
        len(lessons) == 8
        and all(
            item["review_state"] == "PENDING_REVIEW"
            and item["active"] is False
            for item in lessons
        ),
        "Lesson review API mismatch",
    )

    api.c1.require(
        len(audits) == 4
        and all(
            item["verification"] == "VERIFIED"
            and len(item["audit_head"]) == 64
            for item in audits
        ),
        "Audit API mismatch",
    )

    api.c1.require(
        release["decision"] == "HOLD_FOR_REVIEW"
        and release["passed"] == 3
        and release["failed"] == 6
        and len(release["gates"]) == 9,
        "Release-gate API mismatch",
    )

    # Check individually addressable records.
    detail_checks = []

    for item in experiments:
        route = (
            "/api/v1/experiments/"
            + item["experiment_id"]
        )
        response = client.get(route)

        api.c1.require(
            response.status_code == 200
            and response.json() == item,
            "Experiment detail mismatch",
        )
        detail_checks.append(route)

    for item in lessons:
        route = (
            "/api/v1/lesson-review/"
            + item["candidate_id"]
        )
        response = client.get(route)

        api.c1.require(
            response.status_code == 200
            and response.json() == item,
            "Lesson detail mismatch",
        )
        detail_checks.append(route)

    for item in audits:
        route = (
            "/api/v1/audits/"
            + item["component"]
        )
        response = client.get(route)

        api.c1.require(
            response.status_code == 200
            and response.json() == item,
            "Audit detail mismatch",
        )
        detail_checks.append(route)

    # Unknown IDs must not expose arbitrary files.
    unknown = (
        "/api/v1/experiments/not-an-experiment",
        "/api/v1/lesson-review/not-a-candidate",
        "/api/v1/audits/not-an-audit",
        "/api/v1/paired-outcomes/not-a-dataset",
    )

    rejected_unknown = sum(
        client.get(route).status_code == 404
        for route in unknown
    )

    api.c1.require(
        rejected_unknown == len(unknown),
        "Unknown record route was accepted",
    )

    write_routes = (
        "/api/v1/experiments",
        "/api/v1/lesson-review",
        "/api/v1/audits",
        "/api/v1/release-gates",
    )

    rejected_writes = 0

    for route in write_routes:
        for method in (
            "POST", "PUT", "PATCH", "DELETE"
        ):
            response = client.request(
                method, route,
                json={"permission_override": True},
            )
            api.c1.require(
                response.status_code in (404, 405),
                f"Write method available: {method} {route}",
            )
            rejected_writes += 1

    summary = {
        "module": 10,
        "chapter": "10.2",
        "mode": "verified_detailed_research_api",
        "collection_endpoints_tested": len(endpoints),
        "collection_endpoints_successful":
            len(responses),
        "experiment_records": len(experiments),
        "pending_lesson_records": len(lessons),
        "verified_audit_components": len(audits),
        "paired_datasets": len(api.DATASETS),
        "paired_examples": sum(
            responses[name]["examples"]
            for name in api.DATASETS
        ),
        "release_gates": len(release["gates"]),
        "release_gates_passed": release["passed"],
        "release_gates_failed": release["failed"],
        "detail_lookups_verified": len(detail_checks),
        "unknown_lookups_rejected": rejected_unknown,
        "write_methods_rejected": rejected_writes,
        "release_decision": release["decision"],
        "candidate_promoted": False,
        "active_module_8_lessons": 0,
        "permission_changes": 0,
        "live_guardian_connected": False,
        "external_network_requests": 0,
        "model_calls": 0,
    }

    output_dir = Path(output_dir)
    findings_dir = Path(findings_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    findings_dir.mkdir(parents=True, exist_ok=True)

    public_responses = {
        name: {
            "status": 200,
            "response_sha256": api.c1.digest(value),
        }
        for name, value in responses.items()
    }

    api.c1.require(
        len(public_responses) == len(endpoints),
        "Response manifest mismatch",
    )

    paths = {
        "summary": output_dir / "m10_2_summary.json",
        "responses": output_dir / "response_manifest.json",
        "manifest": output_dir / "manifest.json",
    }

    paths["summary"].write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["responses"].write_text(
        json.dumps(public_responses, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "module": 10,
        "chapter": "10.2",
        "summary_sha256": api.c1.digest(summary),
        "response_manifest_sha256":
            api.c1.digest(public_responses),
        "module_9_final_sha256":
            api.c1.digest(
                api.c1.read_json(api.c1.FINAL_PATH)
            ),
        "route_count": len(endpoints),
    }

    paths["manifest"].write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    findings = [
        "# Guardian AI — Module 10.2 Findings",
        "",
        "## Detailed read-only APIs",
        "",
        "- Five versioned experiment records",
        "- Eight pending lesson candidates",
        "- Four verified audit components",
        "- Three paired detector evaluation datasets",
        "- Nine release-gate decisions",
        "",
        "## Security",
        "",
        "No approval, activation, promotion or",
        "permission-changing endpoints were added.",
        "The responses exclude raw attack text.",
        "",
        "## Limitations",
        "",
        "The API is intended for local research.",
        "The stored evidence is predominantly synthetic.",
        "The audit chain is not externally anchored.",
        "The API has no production authentication.",
        "",
        "## Next: Chapter 10.3",
        "",
        "Implement the React research dashboard",
        "using these read-only routes.",
    ]

    (findings_dir / "M10_C2_FINDINGS.md").write_text(
        "\n".join(findings) + "\n",
        encoding="utf-8",
    )

    print("\nFINAL MODULE 10 C2 SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nSaved:", paths["summary"])

    return {
        "summary": summary,
        "manifest": manifest,
        "response_manifest": public_responses,
    }


if __name__ == "__main__":
    run_benchmark()
