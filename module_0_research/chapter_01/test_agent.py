
from react_agent import execute_tool, run_agent, MAX_STEPS

def test_read_valid_document():
    result = execute_tool("read_file", "sales.txt")
    assert result["status"] == "success"
    assert "20%" in result["content"]


def test_unknown_document():
    result = execute_tool("read_file", "unknown.txt")
    assert result["status"] == "error"


def test_unsupported_tool():
    result = execute_tool("delete_file", "sales.txt")
    assert result["status"] == "error"


def test_agent_loop():
    decisions = iter([
        {
            "note": "Read sales",
            "action": "read_file",
            "filename": "sales.txt",
            "final_answer": ""
        },
        {
            "note": "Read costs",
            "action": "read_file",
            "filename": "costs.txt",
            "final_answer": ""
        },
        {
            "note": "Enough evidence",
            "action": "finish",
            "filename": "",
            "final_answer": "Revenue grew 20%, expenses grew 8%."
        }
    ])

    def fake_model(task, history):
        return next(decisions)

    result = run_agent(
        "Summarize financial performance",
        decision_fn=fake_model
    )

    assert result["completed"] is True
    assert len(result["steps"]) == 3
    assert "20%" in result["answer"]
    assert "8%" in result["answer"]


def test_repeated_action_stops():
    def repeating_model(task, history):
        return {
            "note": "Read sales again",
            "action": "read_file",
            "filename": "sales.txt",
            "final_answer": ""
        }

    result = run_agent(
        "Summarize sales",
        decision_fn=repeating_model
    )

    assert result["completed"] is False
    assert len(result["steps"]) <= 3


def test_max_steps_with_different_actions():
    actions = ["sales.txt", "costs.txt"]

    def alternating_model(task, history):
        filename = actions[len(history) % 2]
        return {
            "note": "Read another document",
            "action": "read_file",
            "filename": filename,
            "final_answer": ""
        }

    result = run_agent(
        "Summarize financial performance",
        decision_fn=alternating_model
    )

    assert result["completed"] is False
    assert len(result["steps"]) == MAX_STEPS
