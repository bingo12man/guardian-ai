
from e6_targeted import (
    TASKS,
    evaluate,
    number_present,
    run_comparison,
)


def test_percentage_points_not_percent():
    task = TASKS["marketing_difference"]

    correct = {
        "files": task["needed"],
        "answer": "11 percentage points",
    }

    incorrect = {
        "files": task["needed"],
        "answer": "11%",
    }

    assert evaluate(task, correct)["success"] is True
    assert evaluate(task, incorrect)["success"] is False


def test_wrong_number_rejected():
    task = TASKS["energy_difference"]

    outcome = {
        "files": task["needed"],
        "answer": "19 percentage points",
    }

    assert evaluate(task, outcome)["success"] is False
    assert not number_present(
        "19 percentage points",
        "9",
    )


def test_missing_evidence_fails_even_with_correct_answer():
    task = TASKS["marketing_difference"]

    outcome = {
        "files": ["reach.txt"],
        "answer": "11 percentage points",
    }

    grade = evaluate(task, outcome)

    assert grade["success"] is False
    assert grade["failure_type"] == "missing_evidence"


def test_strategy_paraphrase():
    task = TASKS["strategy"]

    outcome = {
        "files": ["plans.txt"],
        "answer": "We plan to expand regionally.",
    }

    assert evaluate(task, outcome)["success"] is True


def test_missing_profit():
    task = TASKS["missing_profit"]

    valid = {
        "files": [],
        "answer": "The exact profit is not provided.",
    }

    invalid = {
        "files": [],
        "answer": "Net profit is 1000.",
    }

    assert evaluate(task, valid)["success"] is True
    assert evaluate(task, invalid)["success"] is False


def test_summary():
    task = TASKS["marketing_summary"]

    valid = {
        "files": task["needed"],
        "answer": "Reach +18%, costs +7%.",
    }

    invalid = {
        "files": task["needed"],
        "answer": "Reach +18%, costs +70%.",
    }

    assert evaluate(task, valid)["success"] is True
    assert evaluate(task, invalid)["success"] is False


def test_conditions_and_negative_transfer():

    small_tasks = {
        "marketing_difference": TASKS["marketing_difference"],
        "strategy": TASKS["strategy"],
    }

    def fake_actor(task, lessons):

        targeted = any(
            "percentage points" in lesson
            for lesson in lessons
        )

        if task["kind"] == "difference":
            answer = (
                "11 percentage points"
                if targeted
                else "11"
            )
        else:
            # Simulate a negative-transfer case.
            answer = (
                "wrong"
                if targeted
                else "We plan to expand regionally."
            )

        return {
            "files": task["needed"],
            "answer": answer,
        }

    result = run_comparison(
        repeats=1,
        actor=fake_actor,
        tasks=small_tasks,
    )

    summary = result["summary"]

    for condition in summary:
        assert summary[condition]["attempts"] == 2

    changes = result["paired_changes"]["targeted"]

    assert changes["control_fail_treatment_pass"] == 1
    assert changes["control_pass_treatment_fail"] == 1
    assert changes["control_task_regressions"] == 1
