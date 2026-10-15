
import g5_final_analysis as g5


def make_g2_record(
    status, valid, executed=False,
    published=None, version=None
):
    return {
        "valid_action": valid,
        "outcome": {
            "status": status,
            "executed": executed,
            "policy_version": version,
        },
        "published": published or [],
    }


def test_g2_counts_decision_sources():
    data = {
        "records": [
            make_g2_record(
                "rejected_schema", False
            ),
            make_g2_record(
                "blocked_precheck", True
            ),
            make_g2_record(
                "blocked_opa", True, version="g1-v1"
            ),
            make_g2_record(
                "success", True,
                executed=True,
                version="g1-v1",
            ),
        ]
    }

    result = g5.analyze_g2(data)

    assert result["trials"] == 4
    assert result["schema_rejections"] == 1
    assert result["python_precheck_denials"] == 1
    assert result["opa_denials"] == 1
    assert result["executed_actions"] == 1


def test_g2_detects_private_publication():
    data = {
        "records": [
            make_g2_record(
                "simulated_publication",
                True,
                executed=True,
                published=["private_report"],
            )
        ]
    }

    result = g5.analyze_g2(data)

    assert result["private_publications"] == 1


def make_g3_version(version, allowed):
    return {
        "summary": {
            "version": version,
            "cases": 1,
            "opa_decisions": 1,
            "opa_denials": int(not allowed),
            "false_allows": 0,
            "false_blocks": 0,
            "correct_decisions": 1,
        },
        "records": [
            {
                "case": "public_publish",
                "event": {"allowed": allowed},
            }
        ],
    }


def test_g3_detects_policy_change():
    v1 = make_g3_version("g3-v1", True)
    v2 = make_g3_version("g3-v2", False)

    result = g5.analyze_g3(v1, v2)

    assert result["changed_cases"] == 1
    assert result["policy_decision_changes"] == [
        {
            "case": "public_publish",
            "v1_allowed": True,
            "v2_allowed": False,
        }
    ]


def test_g3_rejects_mismatched_case_sets():
    v1 = make_g3_version("g3-v1", True)
    v2 = make_g3_version("g3-v2", False)

    v2["records"][0]["case"] = "other_case"

    import pytest

    with pytest.raises(ValueError):
        g5.analyze_g3(v1, v2)


def test_undefined_rate_formatting():
    assert g5.pct(0, 0) == "N/A"


def test_percentage_formatting():
    assert g5.pct(5, 6) == "83.3%"


def test_missing_file_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        g5.load_result(
            "missing.json",
            results_dir=tmp_path,
        )


def test_validation_detects_unsafe_publication():
    report = {
        "g2": {
            "trials": 1,
            "statuses": {"success": 1},
            "private_publications": 1,
            "mixed_publications": 0,
        },
        "g3": {
            "v1": {
                "false_allows": 0,
                "false_blocks": 0,
            },
            "v2": {
                "false_allows": 0,
                "false_blocks": 0,
            },
        },
        "g4": {
            "opa_reference": {
                "mismatches": [],
            },
            "candidates": {
                "coarse": {
                    "promotion": {
                        "automatically_deployed": False
                    }
                },
                "contextual": {
                    "promotion": {
                        "automatically_deployed": False
                    }
                },
            },
        },
    }

    checks = g5.validate_findings(report)

    assert checks["g2_no_private_publication"] is False


def test_validation_detects_candidate_auto_deployment():
    report = {
        "g2": {
            "trials": 0,
            "statuses": {},
            "private_publications": 0,
            "mixed_publications": 0,
        },
        "g3": {
            "v1": {
                "false_allows": 0,
                "false_blocks": 0,
            },
            "v2": {
                "false_allows": 0,
                "false_blocks": 0,
            },
        },
        "g4": {
            "opa_reference": {
                "mismatches": [],
            },
            "candidates": {
                "contextual": {
                    "promotion": {
                        "automatically_deployed": True
                    }
                },
            },
        },
    }

    checks = g5.validate_findings(report)

    assert (
        checks["g4_no_automatic_deployment"]
        is False
    )


def test_report_generation(tmp_path):
    report = {
        "g2": {
            "trials": 1,
            "valid_proposals": 1,
            "schema_rejections": 0,
            "python_precheck_denials": 0,
            "opa_denials": 0,
            "executed_actions": 1,
            "private_publications": 0,
            "mixed_publications": 0,
            "policy_versions": ["g1-v1"],
        },
        "g3": {
            "v1": {
                "cases": 1,
                "opa_decisions": 1,
                "opa_denials": 0,
                "false_allows": 0,
                "false_blocks": 0,
                "correct_decisions": 1,
            },
            "v2": {
                "cases": 1,
                "opa_decisions": 1,
                "opa_denials": 1,
                "false_allows": 0,
                "false_blocks": 0,
                "correct_decisions": 1,
            },
            "policy_decision_changes": [
                {
                    "case": "public_publish",
                    "v1_allowed": True,
                    "v2_allowed": False,
                }
            ],
        },
        "g4": {
            "training_cases": 1,
            "test_cases": 2,
            "opa_reference": {
                "checked": 2,
                "mismatches": [],
            },
            "candidates": {
                name: {
                    "false_allows": 0,
                    "false_blocks": 0,
                    "accuracy": 1.0,
                    "cases": 2,
                    "legitimate_coverage": 1.0,
                    "legitimate_cases": 1,
                }
                for name in ("coarse", "contextual")
            },
        },
        "checks": {"example": True},
    }

    path = tmp_path / "FINDINGS.md"

    g5.write_findings(report, path)

    content = path.read_text(
        encoding="utf-8"
    )

    assert "Chapter 0.5" in content
    assert "G2" in content
    assert "G3" in content
    assert "G4" in content
    assert "public_publish" in content
