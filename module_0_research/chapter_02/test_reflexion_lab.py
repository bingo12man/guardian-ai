from reflexion_lab import TASKS, evaluate, load_lessons, run_experiment, run_trial, save_lessons


def test_ground_truth_requires_correct_source_and_value():
    task = TASKS["expense"]
    assert not evaluate(task, {"filename": "sales.txt", "answer": "8%"})["success"]
    assert not evaluate(task, {"filename": "costs.txt", "answer": "20%"})["success"]
    assert evaluate(task, {"filename": "costs.txt", "answer": "Expenses increased 8%."})["success"]


def test_memory_round_trip(tmp_path):
    path = tmp_path / "lessons.json"
    save_lessons(["Check the matching document first."], path)
    assert load_lessons(path) == ["Check the matching document first."]


def test_invalid_memory_rejected(tmp_path):
    path = tmp_path / "lessons.json"
    path.write_text('{"lesson": "oops"}')
    import pytest
    with pytest.raises(ValueError):
        load_lessons(path)


def test_failure_feedback_persistence(tmp_path):
    def actor(question, lessons):
        filename = "costs.txt" if lessons else "sales.txt"
        return {"filename": filename,
                "observation": {"status": "success", "content": "synthetic"},
                "answer": "8%" if lessons else "unknown"}
    def reflector(question, outcome, evaluation):
        return "Choose the document that directly matches the requested measure."
    result = run_experiment(actor=actor, reflector=reflector,
                            memory_path=tmp_path / "lessons.json")
    assert not result["baseline"]["evaluation"]["success"]
    assert result["feedback_retry"]["evaluation"]["success"]
    assert result["persistent_retry"]["evaluation"]["success"]
    assert result["persistent_retry"]["lessons_used"] == [result["reflection"]]


def test_success_does_not_generate_fake_failure(tmp_path):
    def actor(question, lessons):
        return {"filename": "costs.txt", "answer": "Expenses rose 8%.",
                "observation": {"status": "success", "content": "8%"}}
    def no_reflection(*args):
        raise AssertionError("Do not reflect on successful attempts")
    result = run_experiment(actor=actor, reflector=no_reflection,
                            memory_path=tmp_path / "none.json")
    assert result["baseline"]["evaluation"]["success"]
    assert result["feedback_retry"] is None
    assert result["persistent_retry"] is None


def test_other_task_is_graded_separately():
    def actor(question, lessons):
        return {"filename": "sales.txt", "answer": "Revenue grew 20%.",
                "observation": {"status": "success", "content": "20%"}}
    assert run_trial("revenue", actor=actor)["evaluation"]["success"]
