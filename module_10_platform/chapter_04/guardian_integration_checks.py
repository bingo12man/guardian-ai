
"""
Guardian AI — Module 10, Chapter 10.4.

Read-only cross-layer integration checks.

- FastAPI TestClient, without starting a server
- Frontend production-build preflight
- Response contracts and provenance
- Source-file before/after fingerprint comparison
- Forbidden HTTP method checks
- No intentional writes to research journals

Does not claim a full automated browser test.
"""

import hashlib
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
M10 = ROOT / "module_10_platform"
M9 = ROOT / "module_9_durable_learning"

sys.path.insert(0, str(M10 / "chapter_02"))

import guardian_research_detail_api as api

FRONTEND = M10 / "chapter_03"
RESULTS = HERE / "results"

DATASETS = ("original", "development", "fresh")
WRITES = ("POST", "PUT", "PATCH", "DELETE")

# Evidence and journal files that the dashboard
# must never modify.
PROTECTED_FILES = {
    "m9_final": M9 / "chapter_05/results/m9_5_final_results.json",
    "m9_manifest": M9 / "chapter_05/results/manifest.json",
    "m10_contract": M9 / "chapter_05/MODULE_10_INTERFACE.json",
    "frozen_audit": M9 / "chapter_01/results/experiment_store/audit.json",
    "candidate_audit": M9 / "chapter_03/results/candidate_experiment_store/audit.json",
    "pending_lessons": M9 / "chapter_02/results/pending_lesson_store/lesson_journal.json",
    "fixture_lifecycle": M9 / "chapter_02/results/fixture_lifecycle_store/lesson_journal.json",
    "detector_summary": M9 / "chapter_03/results/m9_3_summary.json",
    "statistical_summary": M9 / "chapter_04/results/m9_4_analysis.json",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(value):
    content = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence_fingerprints():
    """
    Include every JSON record in both experiment
    stores, not just their audit head files.
    """
    files = dict(PROTECTED_FILES)

    for name, directory in (
        ("baseline", M9 / "chapter_01/results/experiment_store/records"),
        ("candidate", M9 / "chapter_03/results/candidate_experiment_store/records"),
    ):
        require(directory.is_dir(), f"Missing {name} records")
        for path in sorted(directory.glob("*.json")):
            files[f"{name}:{path.stem}"] = path

    require(all(path.is_file() for path in files.values()),
            "Missing protected evidence file")

    return {
        name: sha256_file(path)
        for name, path in sorted(files.items())
    }


def frontend_preflight():
    """
    Inspect the Vite production artifact.

    This is a build preflight, not a simulated
    browser rendering test.
    """
    index_path = FRONTEND / "dist/index.html"
    assets_path = FRONTEND / "dist/assets"

    require(index_path.is_file(), "Run npm run build first")
    require(assets_path.is_dir(), "Missing frontend assets")

    html = index_path.read_text(encoding="utf-8")
    js_files = list(assets_path.glob("*.js"))
    css_files = list(assets_path.glob("*.css"))

    require('id="root"' in html, "Missing React mount")
    require(bool(js_files), "Missing compiled JavaScript")
    require(bool(css_files), "Missing compiled CSS")

    return {
        "build_present": True,
        "javascript_assets": len(js_files),
        "css_assets": len(css_files),
        "build_sha256": fingerprint({
            "index": sha256_file(index_path),
            "js": sorted(sha256_file(p) for p in js_files),
            "css": sorted(sha256_file(p) for p in css_files),
        }),
    }


def get_json(client, path):
    response = client.get(path)
    require(response.status_code == 200,
            f"GET {path} returned {response.status_code}")
    return response.json()


def verify_read_contract(client):
    health = get_json(client, "/health")
    overview = get_json(client, "/api/v1/overview")
    detectors = get_json(client, "/api/v1/detectors")
    old_lessons = get_json(client, "/api/v1/lessons")
    old_release = get_json(client, "/api/v1/release")

    experiments = get_json(client, "/api/v1/experiments")
    lessons = get_json(client, "/api/v1/lesson-review")
    audits = get_json(client, "/api/v1/audits")
    release = get_json(client, "/api/v1/release-gates")

    require(
        health["evidence_verified"] is True
        and health["read_only"] is True,
        "Health contract failed",
    )

    require(
        overview["frozen_baseline_experiments"] == 2
        and overview["candidate_detector_experiments"] == 3
        and experiments["total"] == 5,
        "Experiment count mismatch",
    )

    require(
        overview["pending_lessons"] == 8
        and lessons["total"] == 8
        and old_lessons["pending"] == 8,
        "Lesson count mismatch",
    )

    require(
        overview["active_lessons"] == 0
        and lessons["active"] == 0
        and lessons["approved"] == 0,
        "Unsafe lesson state",
    )

    require(
        audits["components"] == 4
        and audits["all_verified"] is True
        and all(a["verification"] == "VERIFIED"
                for a in audits["audits"]),
        "Audit status mismatch",
    )

    require(
        release["decision"]
        == old_release["decision"]
        == overview["release_decision"]
        == "HOLD_FOR_REVIEW",
        "Release decision mismatch",
    )

    require(
        release["passed"] == 3
        and release["failed"] == 6
        and len(release["gates"]) == 9
        and release["candidate_promoted"] is False
        and release["activation_available"] is False,
        "Release gates mismatch",
    )

    require(
        len(detectors["comparison_results"]) == 3,
        "Detector comparison mismatch",
    )

    experiment_ids = [
        record["experiment_id"]
        for record in experiments["experiments"]
    ]
    lesson_ids = [
        item["candidate_id"]
        for item in lessons["candidates"]
    ]

    require(
        len(set(experiment_ids)) == 5
        and len(set(lesson_ids)) == 8,
        "Duplicate identifiers",
    )

    # Exercise individual detail routes.
    details_checked = 0
    for record in experiments["experiments"]:
        detail = get_json(
            client,
            "/api/v1/experiments/" + record["experiment_id"],
        )
        require(detail == record, "Experiment detail mismatch")
        details_checked += 1

    for candidate in lessons["candidates"]:
        detail = get_json(
            client,
            "/api/v1/lesson-review/" + candidate["candidate_id"],
        )
        require(detail == candidate, "Lesson detail mismatch")
        details_checked += 1

    for audit in audits["audits"]:
        detail = get_json(
            client, "/api/v1/audits/" + audit["component"]
        )
        require(detail == audit, "Audit detail mismatch")
        details_checked += 1

    paired_counts = {}
    paired_gains = {}
    paired_losses = {}

    for name in DATASETS:
        dataset = get_json(
            client, f"/api/v1/paired-outcomes/{name}"
        )

        require(
            dataset["dataset"] == name
            and dataset["examples"] == len(dataset["outcomes"])
            and dataset["contains_raw_content"] is False,
            f"Invalid paired result: {name}",
        )

        paired_counts[name] = dataset["examples"]
        paired_gains[name] = dataset["gains"]
        paired_losses[name] = dataset["losses"]

    require(
        paired_counts == {
            "original": 12,
            "development": 24,
            "fresh": 12,
        },
        "Unexpected evaluation sizes",
    )

    require(
        paired_gains == {
            "original": 0,
            "development": 13,
            "fresh": 6,
        }
        and paired_losses == {
            "original": 3,
            "development": 0,
            "fresh": 0,
        },
        "Paired outcomes changed",
    )

    return {
        "experiments": len(experiment_ids),
        "pending_lessons": len(lesson_ids),
        "verified_audits": len(audits["audits"]),
        "release_gates": len(release["gates"]),
        "detail_lookups": details_checked,
        "paired_examples": sum(paired_counts.values()),
        "paired_gains": paired_gains,
        "paired_losses": paired_losses,
        "release_decision": release["decision"],
    }


def verify_write_protection(client):
    paths = (
        "/api/v1/experiments",
        "/api/v1/lesson-review",
        "/api/v1/audits",
        "/api/v1/release-gates",
        "/api/v1/experiments/not-an-id",
        "/api/v1/lesson-review/not-an-id",
    )

    rejected = 0
    for path in paths:
        for method in WRITES:
            response = client.request(
                method, path,
                json={"approved": True, "permission_override": True},
            )
            require(
                response.status_code in (404, 405),
                f"Unexpected writable route: {method} {path}",
            )
            rejected += 1

    return rejected


def run_integration():
    before = evidence_fingerprints()
    frontend = frontend_preflight()

    with TestClient(api.app) as client:
        contract = verify_read_contract(client)
        rejected = verify_write_protection(client)

    after = evidence_fingerprints()

    require(
        before == after,
        "Research source files changed during API testing",
    )

    return {
        "frontend": frontend,
        "contract": contract,
        "rejected_write_requests": rejected,
        "source_files_checked": len(before),
        "source_files_unchanged": True,
        "permission_changes": 0,
        "candidate_promoted": False,
        "live_lessons_activated": 0,
    }
