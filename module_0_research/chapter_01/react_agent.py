
import json
import os
from pathlib import Path

import requests


# -----------------------------------
# 1. Configuration
# -----------------------------------

MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

MAX_STEPS = 8
MAX_REPEATS = 3

DOCUMENTS = {
    "sales.txt": "Revenue increased by 20%.",
    "costs.txt": "Expenses increased by 8%.",
    "strategy.txt": "The company plans regional expansion.",
}

REQUIRED_FILES = {"sales.txt", "costs.txt"}

TOOLS = {"list_files", "read_file", "finish"}

RESULTS_DIR = Path("results")


# -----------------------------------
# 2. Simulated Tool Executor
# -----------------------------------

def execute_tool(action, argument):
    """Only execute predefined simulated tools."""

    if action == "list_files":
        return {
            "status": "success",
            "files": list(DOCUMENTS.keys())
        }

    if action == "read_file":

        if argument not in DOCUMENTS:
            return {
                "status": "error",
                "message": (
                    "Document not found. "
                    "Choose one valid filename."
                )
            }

        return {
            "status": "success",
            "content": DOCUMENTS[argument]
        }

    return {
        "status": "error",
        "message": "Unsupported action"
    }


# -----------------------------------
# 3. Ollama Agent Decision
# -----------------------------------

def get_agent_decision(task, history):

    # Identify documents successfully read.
    successful_reads = {
        step["argument"]
        for step in history
        if step["action"] == "read_file"
        and step["observation"]["status"] == "success"
    }

    # Find the documents still required.
    remaining = sorted(REQUIRED_FILES - successful_reads)
    retrieved_documents = {}

    for step in history:
        if (
            step["action"] == "read_file"
            and step["observation"]["status"] == "success"
        ):
            retrieved_documents[step["argument"]] = (
                step["observation"]["content"]
            )
    # Recommend the next action.
    if remaining:
        recommended_action = {
            "action": "read_file",
            "filename": remaining[0]
        }
    else:
        recommended_action = {
            "action": "finish",
            "filename": ""
        }

    system_prompt = """
You are a document-reading AI agent.

Your objective is to summarize financial performance
using information retrieved from documents.

Available documents:
- sales.txt
- costs.txt
- strategy.txt

Available actions:
1. list_files
2. read_file
3. finish

Important rules:

1. Choose exactly ONE action per response.
2. Never combine filenames.
3. Never invent percentages or document information.
4. Read sales.txt and costs.txt before finishing.
5. Do not read a document already read successfully.
6. Follow documents_remaining to identify missing files.
7. Use next_recommended_action if you are unsure.
8. If a tool returns an error, choose a different valid action.
9. Only finish after both required documents were read.
10. Your final answer must accurately summarize retrieved data.

Return valid JSON with these fields:

{
  "note": "Brief decision explanation",
  "action": "read_file",
  "filename": "sales.txt",
  "final_answer": ""
}

For finish:

{
  "note": "The task is complete",
  "action": "finish",
  "filename": "",
  "final_answer": "Your evidence-based summary"
}
"""

    schema = {
        "type": "object",
        "properties": {
            "note": {"type": "string"},
            "action": {
                "type": "string",
                "enum": ["list_files", "read_file", "finish"]
            },
            "filename": {
                "type": "string",
                "enum": ["", *DOCUMENTS.keys()]
            },
            "final_answer": {"type": "string"}
        },
        "required": [
            "note",
            "action",
            "filename",
            "final_answer"
        ],
        "additionalProperties": False
    }

    payload = {
        "model": MODEL,
        "stream": False,
        "format": schema,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": json.dumps({
                    "task": task,
                    "documents_read": sorted(successful_reads),
                    "documents_remaining": remaining,
                    "retrieved_evidence": retrieved_documents,
                    "next_recommended_action": recommended_action,
                    "recent_history": history[-3:]
                })
            }
        ],
        "options": {
            "temperature": 0
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    result = response.json()

    return json.loads(result["message"]["content"])



import re

def validate_final_answer(answer):
    if not isinstance(answer, str):
        return False

    answer = answer.lower()

    revenue_correct = bool(
        re.search(r"(revenue|sales)[^\n.]*20\s*%", answer)
    )

    expenses_correct = bool(
        re.search(r"(expenses|costs)[^\n.]*8\s*%", answer)
    )

    return revenue_correct and expenses_correct

# -----------------------------------
# 4. ReAct Execution Loop
# -----------------------------------

def run_agent(task, decision_fn=get_agent_decision):

    history = []
    answer = None
    read_files = set()

    previous_action = None
    repeat_count = 0

    for step in range(1, MAX_STEPS + 1):

        # Ask the LLM for its next action.
        decision = decision_fn(task, history)

        action = decision.get("action")
        filename = decision.get("filename", "")
        final_answer = decision.get("final_answer", "")
        note = decision.get("note", "")

        # Convert the decision into a tool argument.
        if action == "read_file":
            argument = filename
        elif action == "finish":
            argument = final_answer
        else:
            argument = ""

        if not isinstance(argument, str):
            argument = ""

        # -----------------------------------
        # Repetition detection
        # -----------------------------------

        current_action = (action, argument)

        if current_action == previous_action:
            repeat_count += 1
        else:
            repeat_count = 0

        previous_action = current_action

        # MAX_REPEATS = 3 allows the first request
        # plus up to three consecutive repeats.
        if repeat_count >= MAX_REPEATS:
            print("\nStopped: repeated identical action too many times")
            break

        # -----------------------------------
        # Validate and execute
        # -----------------------------------

        if action not in TOOLS:

            observation = {
                "status": "error",
                "message": "Unknown action"
            }

        elif action == "read_file" and filename not in DOCUMENTS:

            observation = {
                "status": "error",
                "message": (
                    "Invalid filename. "
                    "Select exactly one available document."
                )
            }

        elif action == "read_file" and filename in read_files:

            missing = sorted(REQUIRED_FILES - read_files)

            if missing:
                message = (
                    f"{filename} was already read successfully. "
                    f"Read one of these missing documents: {missing}"
                )
            else:
                message = (
                    "All required documents have been read. "
                    "Choose finish and summarize the retrieved evidence. "
                    "Do not read more files."
                )

            observation = {
                "status": "already_read",
                "message": message
            }

        elif action == "finish":

            missing = sorted(REQUIRED_FILES - read_files)

            if missing:
                observation = {
                    "status": "rejected",
                    "message": (
                        f"Cannot finish yet. "
                        f"Read these documents first: {missing}"
                    )
                }

            elif not isinstance(final_answer, str):
                observation = {
                    "status": "rejected",
                    "message": "Final answer must be text."
                }

            elif not validate_final_answer(final_answer):
                observation = {
                    "status": "rejected",
                    "message": (
                        "Answer must include the verified "
                        "revenue and expense percentages."
                    )
                }

            else:
                answer = final_answer
                observation = {
                    "status": "finished"
                }

        else:

            observation = execute_tool(action, argument)

            if (
                action == "read_file"
                and observation["status"] == "success"
            ):
                read_files.add(argument)

        # -----------------------------------
        # Record experiment step
        # -----------------------------------

        record = {
            "step": step,
            "note": note,
            "action": action,
            "argument": argument,
            "observation": observation
        }

        history.append(record)

        print(f"\nStep {step}")
        print("Decision:", note)
        print("Action:", action, argument)
        print("Observation:", observation)

        if observation["status"] == "finished":
            break

    return {
        "task": task,
        "model": MODEL,
        "completed": answer is not None,
        "answer": answer,
        "read_files": sorted(read_files),
        "steps": history
    }


# -----------------------------------
# 5. Save Experiment Results
# -----------------------------------

def save_result(result, filename="e1_result.json"):

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    path = RESULTS_DIR / filename

    with path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print("\nSaved:", path)


# -----------------------------------
# 6. Experiment E1
# -----------------------------------

if __name__ == "__main__":

    task = (
        "Summarize the company's financial performance "
        "using its sales and costs documents. "
        "Include the reported percentages."
    )

    result = run_agent(task)

    save_result(result)

    print("\nFINAL ANSWER:")
    print(result["answer"])
