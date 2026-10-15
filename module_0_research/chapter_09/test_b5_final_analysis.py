
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import b5_final_analysis as b5


def fake_sample(
    case_id,
    repeat,
    document,
    valid=True,
):
    from b1_benchmark import CASES

    case = next(
        c for c in CASES
        if c["id"] == case_id
    )

    if valid:
        output = json.dumps({
            "tool": "read_document",
            "document": document,
        })
    else:
        output = "not json"

    completed = (
        valid and document == case["target"]
    )

    grade = "C" if completed else "I"

    return SimpleNamespace(
        id=f"{case_id}-{repeat}",
        metadata={
            "case_id": case_id,
            "requester": case["requester"],
            "target_document": case["target"],
            "attack": case["attack"],
        },
        output=SimpleNamespace(
            completion=output
        ),
        scores={
            "guardian_task_completion":
                SimpleNamespace(value=grade),
            "model_attack_resistance":
                SimpleNamespace(value=grade),
        },
        error=None,
    )


def fake_samples():
    from b1_benchmark import CASES

    return [
        fake_sample(
            case["id"],
            repeat,
            case["target"],
        )
        for case in CASES
        for repeat in (1, 2, 3)
    ]


def test_valid_inspect_samples():
    report = b5.summarize_inspect_samples(
        fake_samples()
    )

    summary = report["summary"]

    assert summary["trials"] == 18
    assert summary["attack_trials"] == 9
    assert summary["guardian_completions"] == 18
    assert summary["unsafe_proposals"] == 0


def test_unsafe_proposal_classification():
    samples = fake_samples()

    sample = next(
        s for s in samples
        if s.id == "private_instruction-1"
    )

    sample.output.completion = json.dumps({
        "tool": "read_document",
        "document": "confidential.txt",
    })

    sample.scores = {
        "guardian_task_completion":
            SimpleNamespace(value="I"),
        "model_attack_resistance":
            SimpleNamespace(value="I"),
    }

    result = b5.summarize_inspect_samples(
        samples
    )

    assert result["summary"][
        "unsafe_proposals"
    ] == 1

    assert result["summary"][
        "vulnerable_unauthorized_executions"
    ] == 1

    assert result["summary"][
        "guardian_unauthorized_executions"
    ] == 0


def test_invalid_output_classification():
    samples = fake_samples()
    samples[0] = fake_sample(
        "public_clean", 1,
        "public.txt",
        valid=False,
    )

    report = b5.summarize_inspect_samples(
        samples
    )

    assert report["summary"][
        "outcomes"
    ]["invalid_output"] == 1


def test_missing_sample_rejected():
    samples = fake_samples()
    samples.pop()

    with pytest.raises(ValueError):
        b5.summarize_inspect_samples(
            samples
        )


def test_duplicate_sample_rejected():
    samples = fake_samples()
    samples[-1].id = samples[0].id

    with pytest.raises(ValueError):
        b5.summarize_inspect_samples(
            samples
        )


def test_metadata_tampering_rejected():
    samples = fake_samples()
    samples[0].metadata["requester"] = "manager"

    with pytest.raises(ValueError):
        b5.summarize_inspect_samples(
            samples
        )


def test_score_mismatch_rejected():
    samples = fake_samples()

    samples[0].scores[
        "guardian_task_completion"
    ].value = "I"

    with pytest.raises(ValueError):
        b5.summarize_inspect_samples(
            samples
        )


def test_missing_scorer_rejected():
    samples = fake_samples()
    samples[0].scores.pop(
        "model_attack_resistance"
    )

    with pytest.raises(ValueError):
        b5.summarize_inspect_samples(
            samples
        )


def test_model_error_rejected():
    samples = fake_samples()
    samples[0].error = "Model error"

    with pytest.raises(ValueError):
        b5.summarize_inspect_samples(
            samples
        )


def test_score_normalization():
    assert b5.normalize_score("C") is True
    assert b5.normalize_score("I") is False
    assert b5.normalize_score(None) is None


def test_missing_json_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        b5.load_json(
            tmp_path / "missing.json"
        )


def test_invalid_inspect_log_directory(
    tmp_path,
):
    with pytest.raises(FileNotFoundError):
        b5.find_inspect_log(tmp_path)


def test_findings_generation():
    def metrics(trials, attacks):
        return {
            "summary": {
                "trials": trials,
                "attack_trials": attacks,
                "compromised_attack_trials": 0,
                "unsafe_attempts": 0,
                "guardian_completions": trials,
                "rates": {
                    "model_compromise": 0.0,
                    "vulnerable_attack_success": 0.0,
                    "guardian_attack_success": 0.0,
                    "task_completion": 1.0,
                    "recovery_given_unsafe": None,
                },
            }
        }

    previous = {
        "B1": metrics(18, 9),
        "B2": metrics(18, 12),
    }

    inspect_result = b5.summarize_inspect_samples(
        fake_samples()
    )

    markdown = b5.make_findings(
        previous,
        inspect_result,
        Path("logs/example.eval"),
    )

    assert "Limitations" in markdown
    assert "B4" in markdown
    assert "Inspect AI" in markdown
    assert "authorized_completion" in markdown
    assert "guardian_attack_success" not in markdown or (
        "Guardian attack success" in markdown
    )
