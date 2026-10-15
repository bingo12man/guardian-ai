
import copy
import json

import pytest

import guardian_final_evaluation as lab


def validated():
    return lab.validate_platform()


def test_chapter_10_4_source():
    source = lab.validate_chapter_10_4()
    assert source["summary"]["experiments"] == 5
    assert source["summary"]["write_requests_rejected"] == 24
    assert source["summary"]["source_files_unchanged"]
    assert len(source["frontend"]["build_sha256"]) == 64


def test_complete_evidence_chain():
    source = validated()

    assert source["research"]["final"]["module"] == 9
    assert len(source["evidence"]["baseline"]["records"]) == 2
    assert len(source["evidence"]["candidate"]["records"]) == 3


def test_platform_counts():
    report = lab.make_report(validated())
    evidence = report["research_evidence"]

    assert evidence["frozen_baseline_experiments"] == 2
    assert evidence["candidate_detector_experiments"] == 3
    assert evidence["original_lessons_pending"] == 8
    assert evidence["original_lessons_approved"] == 0
    assert evidence["original_lessons_active"] == 0
    assert evidence["audit_components_verified"] == 4
    assert evidence["paired_examples"] == 48


def test_release_held():
    report = lab.make_report(validated())
    release = report["release_gates"]

    assert release["decision"] == "HOLD_FOR_REVIEW"
    assert release["passed"] == 3
    assert release["failed"] == 6
    assert release["candidate_promoted"] is False
    assert release["baseline_replaced"] is False


def test_original_detector_regression():
    report = lab.make_report(validated())
    row = report["detector_comparisons"]["original"]

    assert row["v1_accuracy"] == 1.0
    assert row["v2_accuracy"] == 0.75
    assert row["accuracy_delta_pp"] == -25.0
    assert row["gains"] == 0
    assert row["losses"] == 3


def test_additional_assessment_results():
    report = lab.make_report(validated())
    row = report["detector_comparisons"]["fresh"]

    assert row["v1_accuracy"] == 0.4167
    assert row["v2_accuracy"] == 0.9167
    assert row["gains"] == 6
    assert row["losses"] == 0


def test_all_research_evidence_still_pending():
    report = lab.make_report(validated())

    assert report["candidate_promoted"] is False
    assert report["automatic_lesson_publication"] is False
    assert report["live_agent_permission_changes"] == 0
    assert report["backend"]["live_guardian_connected"] is False


def test_protected_sources_unchanged():
    before = lab.integration.evidence_fingerprints()

    lab.validate_platform()

    after = lab.integration.evidence_fingerprints()
    assert before == after
    assert len(before) >= 14


def test_no_unsupported_production_claims():
    report = lab.make_report(validated())

    assert report["platform_status"] == (
        "LOCAL_RESEARCH_PROTOTYPE"
    )
    assert report["frontend"]["automated_browser_tested"] is False

    limitations = " ".join(report["research_limitations"]).lower()
    assert "real-world" in limitations
    assert "production" in limitations


def test_tampered_chapter_10_4_summary(tmp_path):
    summary = lab.read_json(lab.C4_SUMMARY)
    altered = copy.deepcopy(summary)

    altered["release_decision"] = "APPROVED"

    destination = tmp_path / "altered_summary.json"
    destination.write_text(
        json.dumps(altered), encoding="utf-8"
    )

    with pytest.raises(ValueError):
        lab.validate_chapter_10_4(
            summary_path=destination,
            manifest_path=lab.C4_MANIFEST,
        )


def test_tampered_manifest(tmp_path):
    manifest = lab.read_json(lab.C4_MANIFEST)
    altered = copy.deepcopy(manifest)
    altered["summary_sha256"] = "0" * 64

    destination = tmp_path / "altered_manifest.json"
    destination.write_text(
        json.dumps(altered), encoding="utf-8"
    )

    with pytest.raises(ValueError):
        lab.validate_chapter_10_4(
            summary_path=lab.C4_SUMMARY,
            manifest_path=destination,
        )


def test_missing_summary_rejected(tmp_path):
    with pytest.raises(ValueError):
        lab.validate_chapter_10_4(
            summary_path=tmp_path / "missing.json",
        )


def test_final_report_written(tmp_path):
    result = lab.run_final(
        output_dir=tmp_path / "results",
        document_dir=tmp_path,
    )

    assert result["summary"]["experiments"] == 5
    assert result["summary"]["pending_lessons"] == 8
    assert result["summary"]["release_decision"] == (
        "HOLD_FOR_REVIEW"
    )
    assert result["summary"]["final_manifest_verified"]


def test_all_documents_written(tmp_path):
    lab.run_final(
        output_dir=tmp_path / "results",
        document_dir=tmp_path,
    )

    for name in (
        "GUARDIAN_AI_ARCHITECTURE.md",
        "GUARDIAN_AI_SETUP.md",
        "GUARDIAN_AI_FINAL_REPORT.md",
    ):
        assert (tmp_path / name).is_file()

    assert (
        tmp_path / "results/guardian_final_report.json"
    ).is_file()
    assert (
        tmp_path / "results/manifest.json"
    ).is_file()


def test_report_manifest_hashes(tmp_path):
    output = tmp_path / "results"

    lab.run_final(
        output_dir=output,
        document_dir=tmp_path,
    )

    manifest = lab.read_json(
        output / "manifest.json"
    )

    assert manifest["report_sha256"] == (
        lab.sha256_file(
            output / "guardian_final_report.json"
        )
    )

    assert manifest["architecture_sha256"] == (
        lab.sha256_file(
            tmp_path / "GUARDIAN_AI_ARCHITECTURE.md"
        )
    )

    assert manifest["setup_sha256"] == (
        lab.sha256_file(
            tmp_path / "GUARDIAN_AI_SETUP.md"
        )
    )


def test_deterministic_finalization(tmp_path):
    first = lab.run_final(
        output_dir=tmp_path / "results",
        document_dir=tmp_path,
    )
    second = lab.run_final(
        output_dir=tmp_path / "results",
        document_dir=tmp_path,
    )

    assert first == second


def test_scientific_findings_present():
    report = lab.make_report(validated())
    findings = lab.make_findings(report)

    assert "HOLD_FOR_REVIEW" in findings
    assert "real-world" in findings
    assert "synthetic" in findings.lower()


def test_architecture_has_trust_boundaries():
    architecture = lab.make_architecture()

    assert "Trust boundaries" in architecture
    assert "read-only" in architecture.lower()
    assert "HOLD_FOR_REVIEW" in architecture


def test_setup_has_backend_and_frontend():
    instructions = lab.make_setup()

    assert "uvicorn" in instructions
    assert "npm run dev" in instructions
    assert "127.0.0.1" in instructions
    assert "npm run build" in instructions
