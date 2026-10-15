
import copy
import json

import pytest

import finalize_module_0 as module0


def test_all_ten_evidence_sources_listed():
    assert set(module0.SOURCES) == {
        "B1", "B2", "B3", "B4", "B5",
        "R1", "R2", "R3", "R4", "R5",
    }


def test_existing_results_validate():
    evidence, results = module0.evaluate_evidence()

    assert len(evidence) == 10
    assert results["R1"]["completions"] == 18
    assert results["R2"]["completions"] == 18
    assert results["R3"]["completions"] == 15


def test_r4_evidence():
    _, results = module0.evaluate_evidence()

    assert results["R4"]["raw_completions"] == 3
    assert results["R4"]["sanitized_completions"] == 12
    assert results["R4"]["unauthorized_executions"] == 0


def test_b4_evidence():
    _, results = module0.evaluate_evidence()

    assert results["B4"]["trials"] == 18
    assert results["B4"]["invalid_outputs"] == 6
    assert results["B4"]["unsafe_proposals"] == 2


def test_missing_source_rejected(tmp_path):
    sources = copy.copy(module0.SOURCES)
    sources["B1"] = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        module0.evaluate_evidence(sources)


def test_report_contains_hypotheses():
    _, results = module0.evaluate_evidence()

    report = module0.render_report(results)

    assert "**H1:**" in report
    assert "**H0:**" in report
    assert "Candidate research gap" in report
    assert "Module 1" in report


def test_no_false_novelty_claim():
    _, results = module0.evaluate_evidence()
    report = module0.render_report(results)

    assert "NOT a verified" in report
    assert "has not yet been demonstrated" in report


def test_report_has_prior_work():
    _, results = module0.evaluate_evidence()
    report = module0.render_report(results)

    for title, _, _ in module0.PRIOR_WORK:
        assert title in report
