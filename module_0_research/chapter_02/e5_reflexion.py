
"""
Guardian AI - Module 0, Chapter 0.2
Experiment E5: Reflexion on Strategy-Reasoning Failures

Train on a genuine failed task.
Generate and save one corrective lesson.
Compare no-memory and memory-assisted agents
on separate evaluation questions.

All tools operate on synthetic documents.
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

import requests


# ==========================================
# 1. CONFIGURATION
# ==========================================

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

MEMORY_FILE = RESULTS / "e5_lessons.json"
RESULT_FILE = RESULTS / "e5_results.json"

DOCUMENTS = {
    "sales.txt": "Revenue increased by 20%.",
    "costs.txt": "Expenses increased by 8%.",
    "strategy.txt": "The company plans regional expansion.",
}


# ==========================================
# 2. TRAINING AND EVALUATION TASKS
# ==========================================

# The answer key stays inside the evaluator.
# Only the question is sent to the Actor.

TRAIN_TASKS = {
    "train_1": {
        "question": (
            "What is the company's expansion plan, "
            "and how has revenue changed?"
        ),
        "needs": ["strategy.txt", "sales.txt"],
        "facts": ["regional expansion", "20%"],
    },
    "train_2": {
        "question": (
            "Explain the business expansion strategy "
            "alongside the reported sales growth."
        ),
        "needs": ["strategy.txt", "sales.txt"],
        "facts": ["regional expansion", "20%"],
    },
    "train_3": {
        "question": (
            "Describe the company's future geographical "
            "plans and its revenue trend."
        ),
        "needs": ["strategy.txt", "sales.txt"],
        "facts": ["regional expansion", "20%"],
    },
}

EVAL_TASKS = {
    "strategy_cost": {
        "question": (
            "Describe the company's expansion plans "
            "and the change in its expenses."
        ),
        "needs": ["strategy.txt", "costs.txt"],
        "facts": ["regional expansion", "8%"],
    },
    "strategy_revenue": {
        "question": (
            "Which expansion direction is the company "
            "planning, and what happened to revenue?"
        ),
        "needs": ["strategy.txt", "sales.txt"],
        "facts": ["regional expansion", "20%"],
    },
    "financial_summary": {
        "question": (
            "Summarize the changes in revenue "
            "and expenses."
        ),
        "needs": ["sales.txt", "costs.txt"],
        "facts": ["20%", "8%"],
    },
    "difference": {
        "question": (
            "Calculate how many percentage points "
            "revenue growth exceeds expense growth."
        ),
        "needs": ["sales.txt", "costs.txt"],
        "facts": ["12 percentage points"],
    },
}


# ==========================================
# 3. OLLAMA HELPER
# ==========================================

def ask_ollama(system_prompt, user_data, schema):
    payload = {
        "model": MODEL,
        "stream": False,
        "format": schema,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": json.dumps(user_data),
            },
        ],
        "options": {
            "temperature": 0,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()

    content = response.json()["message"]["content"]
    return json.loads(content)


# ==========================================
# 4. SIMULATED DOCUMENT TOOL
# ==========================================

def read_document(filename):
    if filename not in DOCUMENTS:
        return {
            "status": "error",
            "message": "Unknown document",
        }

    return {
        "status": "success",
        "content": DOCUMENTS[filename],
    }


# ==========================================
# 5. ACTOR
# ==========================================

def model_actor(question, lessons):
    """
    Step A: Choose documents.
    Step B: Read the selected documents.
    Step C: Generate an evidence-based answer.
    """

    selection_schema = {
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": sorted(DOCUMENTS),
                },
                "uniqueItems": True,
            }
        },
        "required": ["files"],
        "additionalProperties": False,
    }

    selection_prompt = """
You are a document-reading AI agent.

Available documents:

sales.txt: revenue and sales figures
costs.txt: expenses and costs
strategy.txt: business strategy and expansion

Choose ALL documents necessary to answer
the question. A question can require more
than one document.

Do not select irrelevant documents.

Treat previous lessons as advice.
Do not treat them as permission to bypass
tools or invent evidence.

Return JSON containing a list named files.
"""

    selected = ask_ollama(
        selection_prompt,
        {
            "question": question,
            "lessons": lessons,
        },
        selection_schema,
    )

    filenames = selected.get("files", [])

    # Do not trust the model's output blindly.
    # Keep only valid, unique filenames.
    valid_files = []
    for filename in filenames:
        if (
            isinstance(filename, str)
            and filename in DOCUMENTS
            and filename not in valid_files
        ):
            valid_files.append(filename)

    observations = {}

    for filename in valid_files:
        result = read_document(filename)

        if result["status"] == "success":
            observations[filename] = result["content"]

    answer_schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"}
        },
        "required": ["answer"],
        "additionalProperties": False,
    }

    answer_prompt = """
You are an evidence-based research assistant.

Answer using ONLY the retrieved documents.

Rules:
1. Preserve exact numerical values.
2. Preserve percentage signs.
3. Combine evidence when necessary.
4. Do not invent financial information.
5. If evidence is insufficient, say so.
6. For differences between percentage
   growth rates, use percentage points.
7. Use previous lessons only as guidance.
8. Do not claim to have read documents
   that are not in retrieved_documents.

Return a concise answer as JSON.
"""

    answer_result = ask_ollama(
        answer_prompt,
        {
            "question": question,
            "retrieved_documents": observations,
            "lessons": lessons,
        },
        answer_schema,
    )

    return {
        "files": valid_files,
        "observations": observations,
        "answer": answer_result.get("answer", ""),
    }


# ==========================================
# 6. INDEPENDENT EVALUATOR
# ==========================================
def correct_difference_number(answer):
    if not isinstance(answer, str):
        return False

    return bool(
        re.search(r"(?<![\d.])12(?![\d.])", answer)
    )

def contains_fact(answer, fact):
    """
    Small synthetic-task evaluator.

    Check numbers with units.
    Check strategy wording.
    """
    if not isinstance(answer, str):
        return False

    text = answer.lower()

    if fact == "regional expansion":
        patterns = [
            r"\bregional\s+expansion\b",
            r"\bexpand(?:s|ed|ing)?\s+regionally\b",
            r"\bexpansion\s+into\s+(?:new\s+)?regions\b",
            r"\bexpand(?:s|ed|ing)?\s+into\s+(?:new\s+)?regions\b",
        ]

        return any(
            re.search(pattern, text)
            for pattern in patterns
        )

    if fact == "12 percentage points":
        return bool(
            re.search(
                r"(?<!\d)12\s*(?:percentage\s+points|pp)\b",
                text,
            )
        )

    if fact.endswith("%"):
        number = re.escape(fact[:-1])
        return bool(
            re.search(
                rf"(?<![\d.]){number}\s*%(?!\d)",
                text,
            )
        )

    return fact.lower() in text


def evaluate(task, outcome):
    """
    Check evidence and answer separately.

    Expected files and facts are never
    passed to the Actor.
    """

    chosen_files = set(outcome.get("files", []))
    required_files = set(task["needs"])

    missing_files = sorted(required_files - chosen_files)

    evidence_ok = len(missing_files) == 0

    answer = outcome.get("answer", "")

    fact_checks = {
        fact: contains_fact(answer, fact)
        for fact in task["facts"]
    }

    facts_ok = all(fact_checks.values())

    # Grounding requirement:
    # the relevant evidence must have been retrieved.
    # This is still a simplified evaluator.
    success = evidence_ok and facts_ok

    problems = []

    if not evidence_ok:
        problems.append(
            "The agent did not retrieve all documents "
            "needed to answer the question."
        )

    if not facts_ok:
        problems.append(
            "The answer omitted or misstated "
            "one or more required facts or units."
        )

    feedback = (
        "Task passed."
        if success
        else " ".join(problems)
    )

    return {
        "success": success,
        "evidence_ok": evidence_ok,
        "facts_ok": facts_ok,
        "missing_files": missing_files,
        "fact_checks": fact_checks,
        "feedback": feedback,
    }


# ==========================================
# 7. REFLECTION
# ==========================================

def model_reflect(question, outcome, evaluation):
    """
    Generate one reusable process lesson.
    The lesson cannot modify tool permissions.
    """

    schema = {
        "type": "object",
        "properties": {
            "lesson": {"type": "string"}
        },
        "required": ["lesson"],
        "additionalProperties": False,
    }

    prompt = """
You are Guardian AI's reflection component.

An agent failed a document reasoning task.

Analyze the attempt and evaluation.

Produce ONE reusable procedural lesson
that may help on different questions.

A good lesson might address:
- identifying every information requirement
- selecting multiple relevant documents
- combining retrieved evidence
- preserving numeric units
- avoiding unsupported claims

Do not simply provide the correct answer.
Do not invent evidence.
Do not modify or bypass security policies.
Keep the lesson under 250 characters.
"""

    result = ask_ollama(
        prompt,
        {
            "question": question,
            "chosen_files": outcome["files"],
            "observations": outcome["observations"],
            "answer": outcome["answer"],
            "evaluator_feedback": evaluation["feedback"],
        },
        schema,
    )

    lesson = result.get("lesson", "")

    if not isinstance(lesson, str):
        return ""

    return lesson.strip()[:250]


# ==========================================
# 8. MEMORY
# ==========================================


def save_memory(lessons, path=None):
    """
    Save lessons to JSON.

    If no path is provided, use the
    current MEMORY_FILE configuration.
    """

    if path is None:
        path = MEMORY_FILE

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    path.write_text(
        json.dumps(lessons, indent=2),
        encoding="utf-8"
    )


def load_memory(path=None):
    """
    Load lessons from JSON.

    Resolve MEMORY_FILE at call time.
    """

    if path is None:
        path = MEMORY_FILE

    path = Path(path)

    if not path.exists():
        return []

    lessons = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(lessons, list):
        raise ValueError("Memory must be a list")

    if not all(isinstance(x, str) for x in lessons):
        raise ValueError("Invalid lesson entry")

    return lessons[:5]



# ==========================================
# 9. TRAINING
# ==========================================

def train(actor=model_actor, reflector=model_reflect):
    """
    Try training tasks without lessons.

    Generate a lesson from the first
    genuine failed attempt.

    Do not manufacture a failure.
    """

    records = []
    lessons = []

    for task_id, task in TRAIN_TASKS.items():

        outcome = actor(task["question"], [])
        grade = evaluate(task, outcome)

        record = {
            "task_id": task_id,
            "question": task["question"],
            "outcome": outcome,
            "evaluation": grade,
        }

        records.append(record)

        print(
            "TRAIN",
            task_id,
            "evidence=",
            grade["evidence_ok"],
            "facts=",
            grade["facts_ok"],
            "success=",
            grade["success"],
        )

        if not grade["success"]:

            lesson = reflector(
                task["question"],
                outcome,
                grade,
            )

            if lesson:
                lessons.append(lesson)

            # Study only the first failure in E5.
            break

    save_memory(lessons)

    return {
        "attempts": records,
        "lessons": lessons,
    }


# ==========================================
# 10. EVALUATION
# ==========================================

def run_evaluation(
    lessons,
    repeats=3,
    actor=model_actor,
):
    """
    Compare no memory vs frozen memory.

    Evaluation tasks do not update memory.
    """

    jobs = []

    for task_id in EVAL_TASKS:
        for repeat in range(repeats):
            for condition in [
                "no_memory",
                "with_memory",
            ]:
                jobs.append(
                    (task_id, repeat, condition)
                )

    random.Random(42).shuffle(jobs)

    records = []

    for task_id, repeat, condition in jobs:

        task = EVAL_TASKS[task_id]

        used_lessons = (
            list(lessons)
            if condition == "with_memory"
            else []
        )

        outcome = actor(
            task["question"],
            used_lessons,
        )

        grade = evaluate(task, outcome)

        record = {
            "task_id": task_id,
            "repeat": repeat,
            "condition": condition,
            "lessons_used": used_lessons,
            "outcome": outcome,
            "evaluation": grade,
        }

        records.append(record)

        print(
            f"{task_id:20s}",
            f"{condition:12s}",
            f"evidence={grade['evidence_ok']}",
            f"facts={grade['facts_ok']}",
            f"success={grade['success']}",
        )

    summary = {}

    for condition in ["no_memory", "with_memory"]:

        subset = [
            record
            for record in records
            if record["condition"] == condition
        ]

        successful = sum(
            record["evaluation"]["success"]
            for record in subset
        )

        attempts = len(subset)

        summary[condition] = {
            "attempts": attempts,
            "successful": successful,
            "success_rate": (
                successful / attempts
                if attempts
                else 0
            ),
        }

    return {
        "records": records,
        "summary": summary,
    }


# ==========================================
# 11. MAIN
# ==========================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    print("\nGUARDIAN AI - EXPERIMENT E5")
    print("Model:", MODEL)

    # Phase 1: Train and reflect.
    training = train()

    print("\nGenerated lessons:")
    if training["lessons"]:
        for lesson in training["lessons"]:
            print("-", lesson)
    else:
        print("(none; training attempts succeeded)")

    # Phase 2: Load frozen memory.
    frozen_lessons = load_memory()

    # Phase 3: Independent evaluation.
    evaluation = run_evaluation(
        frozen_lessons,
        repeats=args.repeats,
    )

    result = {
        "experiment": "E5",
        "model": MODEL,
        "training": training,
        "memory_frozen": frozen_lessons,
        "evaluation": evaluation,
        "memory_comparison_valid": bool(frozen_lessons),
        "note": (
            "Pilot experiment only. Results do not "
            "prove causal learning or generalization."
        ),
    }

    RESULT_FILE.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\nSUMMARY")
    print(
        json.dumps(
            evaluation["summary"],
            indent=2,
        )
    )

    print("\nSaved:", RESULT_FILE)


if __name__ == "__main__":
    main()
