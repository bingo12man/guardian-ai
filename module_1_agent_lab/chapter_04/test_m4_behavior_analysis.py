
import copy
import json

import pytest

import m4_behavior_analysis as m4


def load(chapter):
    data = m4.load_json(
        m4.INPUTS[chapter], chapter
    )
    m4.validate_records(data, chapter)
    return data


def test_all_three_sources():
    assert set(m4.INPUTS) == {
        "C1", "C2", "C3"
    }


def test_c1_evidence():
    data = load("C1")

    assert len(data["records"]) == 36

    profile = m4.analyze_single_action(
        data["records"]
    )

    for model in m4.MODELS:
        assert profile[model]["trials"] == 18
        assert profile[model][
            "correct_proposals"
        ] == 18


def test_c2_evidence():
    data = load("C2")

    profile = m4.analyze_multistep(
        data["records"]
    )

    assert profile["llama3.2:3b"][
        "completions"
    ] == 12

    assert profile["qwen2.5:3b"][
        "completions"
    ] == 18

    assert profile["llama3.2:3b"][
        "unsafe_proposals"
    ] == 12

    assert profile["qwen2.5:3b"][
        "unsafe_proposals"
    ] == 6


def test_c3_memory_evidence():
    data = load("C3")

    profile = m4.analyze_memory(
        data["records"]
    )

    llama = profile["llama3.2:3b"]
    qwen = profile["qwen2.5:3b"]

    assert llama["structured_memory"][
        "completions"
    ] == 12

    assert qwen["no_memory"][
        "completions"
    ] == 24

    assert qwen["raw_memory"][
        "completions"
    ] == 12

    assert qwen["structured_memory"][
        "completions"
    ] == 21


def test_qwen_memory_deltas():
    data = load("C3")

    profile = m4.analyze_memory(
        data["records"]
    )

    comparison = profile["qwen2.5:3b"][
        "comparisons"
    ]

    assert comparison[
        "raw_minus_no_memory_completions"
    ] == -12

    assert comparison[
        "structured_minus_raw_completions"
    ] == 9

    assert comparison[
        "structured_minus_raw_unsafe_proposals"
    ] == -15


def test_llama_memory_deltas():
    data = load("C3")

    profile = m4.analyze_memory(
        data["records"]
    )

    comparison = profile["llama3.2:3b"][
        "comparisons"
    ]

    assert comparison[
        "raw_minus_no_memory_completions"
    ] == 0

    assert comparison[
        "structured_minus_raw_completions"
    ] == 3


@pytest.mark.parametrize(
    "reason,category",
    [
        ("INVALID_OUTPUT", "schema_error"),
        ("TARGET_MISMATCH",
         "target_or_search_mismatch"),
        ("INVALID_SEARCH",
         "target_or_search_mismatch"),
        ("OUT_OF_ORDER_READ",
         "workflow_order_error"),
        ("PREMATURE_FINISH",
         "workflow_order_error"),
        ("ACCESS_DENIED", "access_denial"),
        ("READ_OK", "authorized_read"),
        ("SEARCH_OK", "authorized_search"),
        ("TASK_COMPLETED", "authorized_finish"),
        ("NEW_REASON", "other"),
    ],
)
def test_error_taxonomy(reason, category):
    assert m4.categorize_event({
        "decision": reason
    }) == category


def test_duplicate_record_rejected():
    data = copy.deepcopy(load("C2"))

    data["records"][1] = copy.deepcopy(
        data["records"][0]
    )

    with pytest.raises(ValueError):
        m4.validate_records(data, "C2")


def test_corrupt_unsafe_count_rejected():
    data = copy.deepcopy(load("C3"))

    data["records"][0][
        "unsafe_proposals"
    ] += 1

    with pytest.raises(ValueError):
        m4.validate_records(data, "C3")


def test_corrupt_block_count_rejected():
    data = copy.deepcopy(load("C2"))

    data["records"][0][
        "guardian_blocks"
    ] += 1

    with pytest.raises(ValueError):
        m4.validate_records(data, "C2")


def test_unauthorized_execution_rejected():
    data = copy.deepcopy(load("C3"))

    data["records"][0][
        "unauthorized_executions"
    ] = 1

    with pytest.raises(ValueError):
        m4.validate_records(data, "C3")


def test_offline_report_rejected(
    tmp_path,
):
    data = copy.deepcopy(load("C1"))
    data["mode"] = "offline"

    path = tmp_path / "offline.json"
    path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        m4.load_json(path, "C1")


def test_report_generation(tmp_path):
    report = m4.run_analysis(
        results_dir=tmp_path / "results"
    )

    assert report["experiment"] == "M1_C4"

    assert (
        tmp_path / "results"
        / "m4_behavior_results.json"
    ).exists()

    markdown = (
        tmp_path / "M4_FINDINGS.md"
    )

    assert markdown.exists()

    text = markdown.read_text(
        encoding="utf-8"
    )

    assert "Limitations" in text
    assert "Memory conditions" in text
    assert "Qwen" in text
    assert "Llama" in text
