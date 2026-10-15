
import json

from e5_reflexion import (
    EVAL_TASKS,
    contains_fact,
    evaluate,
    load_memory,
    run_evaluation,
    save_memory,
    train,
)


def test_correct_strategy_and_cost_answer():
    task = EVAL_TASKS["strategy_cost"]

    outcome = {
        "files": ["strategy.txt", "costs.txt"],
        "answer": (
            "The company plans regional expansion "
            "and expenses increased by 8%."
        ),
    }

    result = evaluate(task, outcome)

    assert result["evidence_ok"] is True
    assert result["facts_ok"] is True
    assert result["success"] is True


def test_missing_document_is_detected():
    task = EVAL_TASKS["strategy_cost"]

    outcome = {
        "files": ["costs.txt"],
        "answer": (
            "The company plans regional expansion "
            "and expenses increased by 8%."
        ),
    }

    result = evaluate(task, outcome)

    assert result["evidence_ok"] is False
    assert result["success"] is False


def test_wrong_percentage_is_rejected():
    task = EVAL_TASKS["strategy_cost"]

    outcome = {
        "files": ["strategy.txt", "costs.txt"],
        "answer": (
            "The company plans regional expansion "
            "and expenses increased by 18%."
        ),
    }

    result = evaluate(task, outcome)

    assert result["facts_ok"] is False
    assert result["success"] is False


def test_percentage_points_require_unit():
    assert contains_fact(
        "The difference is 12 percentage points.",
        "12 percentage points",
    )

    assert not contains_fact(
        "The difference is 12%.",
        "12 percentage points",
    )


def test_persistent_memory(tmp_path):
    path = tmp_path / "lessons.json"

    lessons = [
        "Collect all relevant documents before answering."
    ]

    save_memory(lessons, path)

    assert load_memory(path) == lessons


def test_training_creates_lesson_on_failure(
    tmp_path, monkeypatch
):
    import e5_reflexion as lab

    monkeypatch.setattr(
        lab,
        "MEMORY_FILE",
        tmp_path / "training_memory.json",
    )

    def failed_actor(question, lessons):
        return {
            "files": ["sales.txt"],
            "observations": {
                "sales.txt": "Revenue increased by 20%."
            },
            "answer": "Revenue increased by 20%.",
        }

    def fake_reflector(question, outcome, evaluation):
        return (
            "Identify all documents needed "
            "for multi-part questions."
        )

    result = train(
        actor=failed_actor,
        reflector=fake_reflector,
    )

    assert len(result["lessons"]) == 1
    assert result["attempts"][0]["evaluation"]["success"] is False
    assert len(load_memory(lab.MEMORY_FILE)) == 1


def test_memory_comparison(tmp_path):
    """
    A fake actor that succeeds on strategy
    only when it receives a lesson.

    This tests experiment wiring,
    not actual Ollama learning.
    """

    def fake_actor(question, lessons):
        if "expansion" in question.lower():
            files = (
                ["strategy.txt", "costs.txt", "sales.txt"]
                if lessons
                else ["costs.txt", "sales.txt"]
            )
        else:
            files = ["sales.txt", "costs.txt"]

        return {
            "files": files,
            "observations": {},
            "answer": (
                "The company plans regional expansion. "
                "Revenue increased by 20%, "
                "expenses increased by 8%, "
                "a difference of 12 percentage points."
            ),
        }

    result = run_evaluation(
        lessons=["Check every required document."],
        repeats=1,
        actor=fake_actor,
    )

    summary = result["summary"]

    assert summary["no_memory"]["attempts"] == 4
    assert summary["with_memory"]["attempts"] == 4

    assert (
        summary["with_memory"]["successful"]
        >= summary["no_memory"]["successful"]
    )

def test_regional_expansion_paraphrase():
    assert contains_fact(
        "The company plans to expand regionally.",
        "regional expansion"
    )

    assert contains_fact(
        "The company plans regional expansion.",
        "regional expansion"
    )

    assert not contains_fact(
        "The company plans to reduce regional operations.",
        "regional expansion"
    )