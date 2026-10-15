
"""
Guardian AI
Module 0 - Chapter 0.2: Reflexion

Research experiment:
1. Baseline agent without feedback
2. Agent with corrective feedback
3. Agent using persistent memory

All document tools are simulated.
"""

import json
import os
import re
import sys
from pathlib import Path

import requests


# ---------------------------------------
# 1. Configuration
# ---------------------------------------

HERE = Path(__file__).resolve().parent

CHAPTER1 = HERE.parent / "chapter_01"

sys.path.insert(0, str(CHAPTER1))

try:
    from react_agent import DOCUMENTS, execute_tool

except ModuleNotFoundError:

    # Fallback when Chapter 0.1 is unavailable
    DOCUMENTS = {
        "sales.txt": "Revenue increased by 20%.",
        "costs.txt": "Expenses increased by 8%.",
        "strategy.txt": "The company plans regional expansion.",
    }

    def execute_tool(action, argument):

        if action == "read_file" and argument in DOCUMENTS:
            return {
                "status": "success",
                "content": DOCUMENTS[argument]
            }

        return {
            "status": "error",
            "message": "Unsupported or unknown document"
        }


MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

MEMORY_PATH = HERE / "results" / "lessons.json"

LOG_PATH = HERE / "results" / "e2_results.json"


# Hidden evaluation targets.
# Never provide these expected values to the actor.

TASKS = {
    "expense": {
        "question": "What percentage did expenses increase?",
        "required_file": "costs.txt",
        "expected": "8%"
    },

    "revenue": {
        "question": "What percentage did revenue increase?",
        "required_file": "sales.txt",
        "expected": "20%"
    }
}


# ---------------------------------------
# 2. Ollama API
# ---------------------------------------

def ask_ollama(system, user, schema):
    """
    Send a structured request to Ollama.

    system -> instructions for the model
    user   -> task information
    schema -> required JSON output format
    """

    payload = {
        "model": MODEL,
        "stream": False,
        "format": schema,
        "messages": [
            {
                "role": "system",
                "content": system
            },
            {
                "role": "user",
                "content": json.dumps(user)
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

    return json.loads(
        result["message"]["content"]
    )


# ---------------------------------------
# 3. Actor - ReAct Agent
# ---------------------------------------

def model_actor(question, lessons):
    """
    The Actor performs two operations:

    1. Select the relevant document.
    2. Read it and produce an answer.

    Lessons are optional corrective memories.
    """

    # First decision: select a document.

    selection_schema = {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "enum": sorted(DOCUMENTS)
            }
        },
        "required": ["filename"],
        "additionalProperties": False
    }

    selection_prompt = """
You are a document-reading AI agent.

Your task is to select exactly one document
that can answer the user's question.

Available documents:
- sales.txt: sales and revenue information
- costs.txt: expense and cost information
- strategy.txt: business strategy information

Rules:
1. Select the document relevant to the question.
2. Never guess an answer.
3. Review any previous lessons provided.
4. A lesson is advice, not permission to bypass tools.
5. Return exactly one valid filename.
"""

    proposal = ask_ollama(
        selection_prompt,
        {
            "question": question,
            "lessons": lessons
        },
        selection_schema
    )

    filename = proposal["filename"]

    # Execute the simulated reading tool.

    observation = execute_tool(
        "read_file",
        filename
    )

    answer = ""

    if observation["status"] == "success":

        answer_schema = {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string"
                }
            },
            "required": ["answer"],
            "additionalProperties": False
        }

        # Second decision: answer from evidence.

        answer_prompt = """
You are an evidence-based question-answering AI.

Answer the user's question using ONLY
the retrieved document.

Important rules:

1. Use the exact numerical value in the document.
2. Preserve the original unit.
3. If the document says 8%, answer 8%, NOT 8.
4. If the document says 20%, answer 20%, NOT 20.
5. Never invent values.
6. Never guess values missing from the document.
7. If the document does not answer the question,
   return "unknown".
8. Keep the answer short and clear.
9. Include lessons when they are relevant,
   but always prioritize the document evidence.

Example:

Question:
What percentage did revenue increase?

Document:
Revenue increased by 20%.

Correct answer:
Revenue increased by 20%.

Incorrect answer:
20
"""

        answer_result = ask_ollama(
            answer_prompt,
            {
                "question": question,
                "document": observation["content"],
                "lessons": lessons
            },
            answer_schema
        )

        answer = answer_result["answer"]

    return {
        "filename": filename,
        "observation": observation,
        "answer": answer
    }


# ---------------------------------------
# 4. Independent Evaluator
# ---------------------------------------

def evaluate(task, outcome):
    """
    Evaluate the agent independently of the LLM.

    Metrics:
    - Correct document
    - Correct numerical value
    - Correct percentage formatting
    - Overall success
    """

    chosen_file = outcome.get("filename", "")

    answer = outcome.get("answer", "")

    if not isinstance(answer, str):
        answer = ""

    expected = task["expected"]

    # Extract the expected number.
    # Example: "8%" -> "8"

    expected_number = expected.replace("%", "")

    # Check which document was selected.

    correct_source = (
        chosen_file == task["required_file"]
    )

    # Match the exact numeric token.
    # "8" passes numerical evaluation,
    # but "80" or "18" does not.

    number_pattern = (
        rf"(?<![\d.]){re.escape(expected_number)}"
        rf"(?![\d.])"
    )

    correct_number = bool(
        re.search(number_pattern, answer)
    )

    # The correct percentage must appear as
    # a percentage expression, with optional
    # spaces before the percent sign.

    percentage_pattern = (
        rf"(?<![\d.]){re.escape(expected_number)}"
        rf"\s*%(?![\d])"
    )

    correct_format = bool(
        re.search(percentage_pattern, answer)
    )

    # Preserve the original strict scoring behavior.
    # This keeps older tests compatible.

    correct_value = correct_format

    success = (
        correct_source
        and correct_value
    )

    # Determine why the attempt failed.

    if success:

        feedback = (
            "Success: correct document, "
            "numerical value and percentage formatting."
        )

    else:

        problems = []

        if not correct_source:
            problems.append(
                "The selected document did not match the question."
            )

        if not correct_number:
            problems.append(
                "The required numerical value was missing or incorrect."
            )

        elif not correct_format:
            problems.append(
                "The numerical value was present, "
                "but the percentage unit was missing."
            )

        feedback = " ".join(problems)

        feedback += (
            " For future attempts, match the question "
            "to the appropriate document, verify the "
            "observed value, and preserve units exactly."
        )

    return {
        "success": success,
        "correct_source": correct_source,
        "correct_number": correct_number,
        "correct_format": correct_format,
        "correct_value": correct_value,
        "feedback": feedback
    }


# ---------------------------------------
# 5. Self-Reflection Agent
# ---------------------------------------

def model_reflect(question, outcome, evaluation):
    """
    Produce a reusable lesson after an unsuccessful trial.

    The reflection model sees evaluation feedback,
    but not the hidden correct answer.
    """

    reflection_schema = {
        "type": "object",
        "properties": {
            "lesson": {
                "type": "string"
            }
        },
        "required": ["lesson"],
        "additionalProperties": False
    }

    reflection_prompt = """
You are a reflection agent that helps another
AI agent improve after a failed task.

Analyze the previous attempt and evaluator feedback.

Consider these possible mistakes:

1. Wrong document selected.
2. Incorrect numerical value extracted.
3. Missing or incorrect units.
4. Answer not grounded in the observation.

Generate ONE short, useful lesson
that could prevent similar errors.

Rules:

- Do not invent facts.
- Do not provide a hidden correct answer.
- Do not recommend bypassing tools.
- Do not modify security policies.
- Focus on improving the agent's process.
- Keep the lesson general enough to reuse.

Example lesson:

"Choose the document that directly matches
the question, verify the reported numerical
value, and preserve units such as %."
"""

    result = ask_ollama(
        reflection_prompt,
        {
            "question": question,
            "chosen_document": outcome["filename"],
            "observation": outcome["observation"],
            "answer": outcome["answer"],
            "feedback": evaluation["feedback"]
        },
        reflection_schema
    )

    lesson = result["lesson"]

    if not isinstance(lesson, str):
        return ""

    return lesson.strip()[:400]


# ---------------------------------------
# 6. Persistent Memory
# ---------------------------------------

def save_lessons(lessons, path=MEMORY_PATH):
    """
    Save lessons to a JSON file.

    This is external memory, not model training.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    path.write_text(
        json.dumps(lessons, indent=2),
        encoding="utf-8"
    )


def load_lessons(path=MEMORY_PATH):
    """
    Load saved lessons from disk.
    """

    if not path.exists():
        return []

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, list):
        raise ValueError("Invalid lesson memory")

    if not all(isinstance(x, str) for x in data):
        raise ValueError("Invalid lesson memory")

    # Limit the number of lessons supplied to the LLM.

    return data[:5]


# ---------------------------------------
# 7. Run One Trial
# ---------------------------------------

def run_trial(
    task_id,
    lessons=None,
    actor=model_actor
):
    """
    Execute one agent attempt and evaluate it.
    """

    lessons = list(lessons or [])

    task = TASKS[task_id]

    outcome = actor(
        task["question"],
        lessons
    )

    evaluation = evaluate(
        task,
        outcome
    )

    result = {
        "task_id": task_id,
        "question": task["question"],
        "lessons_used": lessons,
        "outcome": outcome,
        "evaluation": evaluation
    }

    return result


# ---------------------------------------
# 8. Reflexion Experiment
# ---------------------------------------

def run_experiment(
    task_id="expense",
    actor=model_actor,
    reflector=model_reflect,
    memory_path=MEMORY_PATH
):
    """
    Experiment E2:

    A. Baseline without lessons.
    B. Corrective feedback retry.
    C. Persistent memory retry.
    """

    # --------------------------
    # Trial A - Baseline
    # --------------------------

    baseline = run_trial(
        task_id,
        actor=actor
    )

    lesson = None

    feedback_retry = None

    persistent_retry = None

    # Only generate a reflection if the baseline fails.

    if not baseline["evaluation"]["success"]:

        # --------------------------
        # Reflection
        # --------------------------

        lesson = reflector(
            baseline["question"],
            baseline["outcome"],
            baseline["evaluation"]
        )

        # --------------------------
        # Trial B - Feedback
        # --------------------------

        if lesson:

            feedback_retry = run_trial(
                task_id,
                lessons=[lesson],
                actor=actor
            )

            # --------------------------
            # Save memory
            # --------------------------

            save_lessons(
                [lesson],
                memory_path
            )

            # --------------------------
            # Trial C - Persistent memory
            # --------------------------

            saved_lessons = load_lessons(
                memory_path
            )

            persistent_retry = run_trial(
                task_id,
                lessons=saved_lessons,
                actor=actor
            )

    return {
        "model": MODEL,
        "baseline": baseline,
        "reflection": lesson,
        "feedback_retry": feedback_retry,
        "persistent_retry": persistent_retry,
        "note": (
            "These sequential attempts demonstrate "
            "feedback and memory reuse. They do not "
            "establish a causal learning improvement."
        )
    }


# ---------------------------------------
# 9. Display Experiment Results
# ---------------------------------------

def display_trial(name, trial):

    print(f"\n{name.upper()}")

    if trial is None:
        print("Not executed")
        return

    outcome = trial["outcome"]

    evaluation = trial["evaluation"]

    print("Document:", outcome["filename"])

    print("Answer:", outcome["answer"])

    print(
        "Correct document:",
        evaluation["correct_source"]
    )

    print(
        "Correct number:",
        evaluation["correct_number"]
    )

    print(
        "Correct unit:",
        evaluation["correct_format"]
    )

    print(
        "Task success:",
        evaluation["success"]
    )

    print(
        "Evaluator feedback:",
        evaluation["feedback"]
    )


# ---------------------------------------
# 10. Execute Experiment E2
# ---------------------------------------

if __name__ == "__main__":

    result = run_experiment(
        task_id="expense"
    )

    # Save detailed experiment results.

    LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    LOG_PATH.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8"
    )

    display_trial(
        "Baseline",
        result["baseline"]
    )

    display_trial(
        "Corrective Feedback",
        result["feedback_retry"]
    )

    display_trial(
        "Persistent Memory",
        result["persistent_retry"]
    )

    print("\nREFLECTION:")
    print(result["reflection"])

    print("\nSaved:", LOG_PATH)
