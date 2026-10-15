
import copy
import json

import pytest

import guardian_adaptive_scheduler as lab
import c2_adaptive_benchmark as bench


def dataset():
    return lab.load_verified_dataset()


def scheduler_for(model="llama3.2:3b", budget=8):
    data = dataset()
    return lab.AdaptiveScheduler(
        model=model,
        profile=data["profiles"][model],
        exercises=data["exercises"],
        max_attempts=budget,
    )


def trusted_answer(assignment):
    data = dataset()
    item = next(
        row for row in data["exercises"]
        if row["exercise_id"] == assignment.exercise_id
    )
    return json.dumps(item["expected_action"])


def test_dataset_loads():
    data = dataset()
    assert len(data["exercises"]) == 21
    assert len(data["profiles"]) == 2


def test_scheduler_has_budget():
    scheduler = scheduler_for(budget=3)
    assert scheduler.remaining() == 3


def test_assignment_contains_no_gold_answer():
    scheduler = scheduler_for()
    assignment = scheduler.next_exercise()

    assert assignment.model == "llama3.2:3b"
    assert not hasattr(assignment, "expected_action")
    assert assignment.exercise_id.startswith("exercise_")


def test_pending_exercise_must_be_completed():
    scheduler = scheduler_for()
    scheduler.next_exercise()

    with pytest.raises(RuntimeError):
        scheduler.next_exercise()


def test_correct_submission():
    scheduler = scheduler_for()
    assignment = scheduler.next_exercise()

    result = scheduler.submit(
        assignment,
        trusted_answer(assignment),
    )

    assert result["correct"]
    assert result["attempts_used"] == 1
    assert result["attempts_remaining"] == 7
    assert not result["tool_executed"]
    assert not result["privileges_changed"]


def test_invalid_submission():
    scheduler = scheduler_for()
    assignment = scheduler.next_exercise()

    result = scheduler.submit(
        assignment, "invalid JSON"
    )

    assert not result["correct"]
    assert result["error"] == "INVALID_OUTPUT"
    assert scheduler.summary()["incorrect"] == 1


def test_no_replay_of_submission():
    scheduler = scheduler_for()
    assignment = scheduler.next_exercise()

    scheduler.submit(
        assignment,
        trusted_answer(assignment),
    )

    with pytest.raises(RuntimeError):
        scheduler.submit(
            assignment,
            trusted_answer(assignment),
        )


def test_budget_enforced():
    scheduler = scheduler_for(budget=2)

    for _ in range(2):
        assignment = scheduler.next_exercise()
        scheduler.submit(
            assignment,
            trusted_answer(assignment),
        )

    assert scheduler.remaining() == 0
    assert scheduler.next_exercise() is None
    assert scheduler.summary()["attempts"] == 2


def test_no_assessment_selected():
    scheduler = scheduler_for(budget=20)

    for _ in range(20):
        assignment = scheduler.next_exercise()
        assert "guide.txt" not in assignment.prompt

        scheduler.submit(
            assignment, trusted_answer(assignment)
        )

    assert scheduler.summary()[
        "assessment_exercises_used"
    ] == 0


def test_incorrect_increases_skill_priority_relative_to_success():
    first = scheduler_for()
    second = scheduler_for()

    a = first.next_exercise()
    b = second.next_exercise()

    assert a.exercise_id == b.exercise_id

    # Same assigned skill, two different outcomes.
    first.submit(a, "invalid JSON")
    second.submit(
        b, trusted_answer(b)
    )

    # Retrieve the exact lesson ID associated
    # with the selected exercise.
    data = dataset()

    selected_exercise = next(
        row for row in data["exercises"]
        if row["exercise_id"] == a.exercise_id
    )

    lesson_id = selected_exercise["lesson_id"]

    first_skill = next(
        x for x in first.priorities()
        if x["lesson_id"] == lesson_id
        and x["category"] == a.category
        and x["stage"] == a.stage
    )

    second_skill = next(
        x for x in second.priorities()
        if x["lesson_id"] == lesson_id
        and x["category"] == b.category
        and x["stage"] == b.stage
    )

    assert first_skill["incorrect"] == 1
    assert second_skill["correct"] == 1

    assert (
        first_skill["priority"]
        > second_skill["priority"]
    )


def test_priorities_remain_nonnegative():
    scheduler = scheduler_for()

    for _ in range(8):
        assignment = scheduler.next_exercise()
        scheduler.submit(assignment, "invalid JSON")

    assert all(
        item["priority"] >= 0
        for item in scheduler.priorities()
    )


def test_two_models_have_separate_sessions():
    llama = scheduler_for("llama3.2:3b")
    qwen = scheduler_for("qwen2.5:3b")

    a = llama.next_exercise()
    llama.submit(a, "invalid JSON")

    assert llama.summary()["attempts"] == 1
    assert qwen.summary()["attempts"] == 0


def test_unknown_model_rejected():
    data = dataset()

    with pytest.raises(ValueError):
        lab.AdaptiveScheduler(
            "unknown",
            data["profiles"]["llama3.2:3b"],
            data["exercises"],
        )


def test_invalid_budget_rejected():
    data = dataset()

    for budget in (0, -1, True, 101):
        with pytest.raises(ValueError):
            lab.AdaptiveScheduler(
                "llama3.2:3b",
                data["profiles"]["llama3.2:3b"],
                data["exercises"],
                max_attempts=budget,
            )


def test_wrong_assignment_rejected():
    scheduler = scheduler_for()
    assignment = scheduler.next_exercise()

    other = lab.Assignment(
        sequence=assignment.sequence,
        exercise_id="fabricated",
        model=assignment.model,
        category=assignment.category,
        stage=assignment.stage,
        prompt=assignment.prompt,
        priority=assignment.priority,
    )

    with pytest.raises(ValueError):
        scheduler.submit(other, "{}")


def test_audit_verification():
    scheduler = scheduler_for()

    assignment = scheduler.next_exercise()
    scheduler.submit(assignment, "invalid JSON")

    assert lab.verify_events(
        scheduler.events(),
        scheduler.summary()["final_audit_hash"],
    )


def test_audit_tampering_rejected():
    scheduler = scheduler_for()

    assignment = scheduler.next_exercise()
    scheduler.submit(assignment, "invalid JSON")

    events = scheduler.events()
    events[0]["correct"] = True

    with pytest.raises(ValueError):
        lab.verify_events(events)


def test_audit_snapshot_is_copy():
    scheduler = scheduler_for()

    assignment = scheduler.next_exercise()
    scheduler.submit(assignment, "invalid JSON")

    events = scheduler.events()
    events[0]["correct"] = True

    assert scheduler.events()[0]["correct"] is False


def test_scheduler_is_deterministic():
    a = scheduler_for()
    b = scheduler_for()

    a_ids = []
    b_ids = []

    for _ in range(8):
        first = a.next_exercise()
        second = b.next_exercise()

        a_ids.append(first.exercise_id)
        b_ids.append(second.exercise_id)

        a.submit(first, "invalid JSON")
        b.submit(second, "invalid JSON")

    assert a_ids == b_ids
    assert a.events() == b.events()


def test_no_privilege_restoration():
    scheduler = scheduler_for()

    for _ in range(8):
        assignment = scheduler.next_exercise()
        scheduler.submit(
            assignment, trusted_answer(assignment)
        )

    assert scheduler.summary()[
        "privilege_restorations"
    ] == 0
    assert scheduler.summary()["tool_executions"] == 0


def test_invalid_source_dataset_rejected(tmp_path):
    data = dataset()

    corrupted = copy.deepcopy(data["exercises"])
    corrupted[0]["restores_privileges"] = True

    for name, value in (
        ("agent_risk_profiles.json", data["profiles"]),
        ("curriculum_exercises.json", corrupted),
        ("m6_1_summary.json", data["summary"]),
        ("manifest.json", {
            "profiles_sha256": lab.digest(data["profiles"]),
            "exercises_sha256": lab.digest(corrupted),
            "summary_sha256": lab.digest(data["summary"]),
        }),
    ):
        (tmp_path / name).write_text(
            json.dumps(value),
            encoding="utf-8",
        )

    with pytest.raises(ValueError):
        lab.load_verified_dataset(tmp_path)


def test_benchmark_results(tmp_path):
    result = bench.run_benchmark(
        output_dir=tmp_path
    )

    summary = result["summary"]

    assert summary["sessions"] == 6
    assert summary["total_practice_attempts"] == 48
    assert summary["total_correct"] == 30
    assert summary["total_incorrect"] == 18
    assert summary["audits_verified"] == 6
    assert summary["assessments_consumed"] == 0
    assert summary["privilege_restorations"] == 0


def test_benchmark_artifacts(tmp_path):
    bench.run_benchmark(output_dir=tmp_path)

    assert (
        tmp_path / "c2_adaptive_results.json"
    ).exists()

    assert (tmp_path / "manifest.json").exists()
