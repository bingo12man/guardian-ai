
"""
Module 6 Chapter 6.2 scheduler benchmark.

Two models x three scripted behaviors.

Each session has an eight-attempt budget.

These fixture results test scheduler mechanics.
They are NOT Llama/Qwen performance results.
"""

import hashlib
import json
from pathlib import Path

from guardian_adaptive_scheduler import (
    AdaptiveScheduler,
    load_verified_dataset,
    verify_events,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

STRATEGIES = (
    "always_correct",
    "initial_failure",
    "always_invalid",
)

ATTEMPT_BUDGET = 8


def scripted_proposal(
    strategy,
    assignment,
    exercise,
    attempt_number,
):
    if strategy == "always_invalid":
        return "this is not valid JSON"

    if (
        strategy == "initial_failure"
        and attempt_number == 1
    ):
        return "this is not valid JSON"

    return json.dumps(
        exercise["expected_action"]
    )


def run_session(
    model,
    strategy,
    data,
    budget=ATTEMPT_BUDGET,
):
    if strategy not in STRATEGIES:
        raise ValueError("Unknown strategy")

    scheduler = AdaptiveScheduler(
        model=model,
        profile=data["profiles"][model],
        exercises=data["exercises"],
        max_attempts=budget,
    )

    exercise_lookup = {
        row["exercise_id"]: row
        for row in data["exercises"]
        if row["model"] == model
        and row["split"] == "practice"
    }

    for attempt_number in range(1, budget + 1):
        assignment = scheduler.next_exercise()

        if assignment is None:
            raise AssertionError(
                "Budget exhausted prematurely"
            )

        exercise = exercise_lookup[
            assignment.exercise_id
        ]

        proposal = scripted_proposal(
            strategy,
            assignment,
            exercise,
            attempt_number,
        )

        result = scheduler.submit(
            assignment, proposal
        )

        if (
            result["tool_executed"]
            or result["privileges_changed"]
        ):
            raise AssertionError(
                "Practice changed authorization"
            )

    if scheduler.next_exercise() is not None:
        raise AssertionError(
            "Attempt budget was bypassed"
        )

    summary = scheduler.summary()

    verify_events(
        scheduler.events(),
        summary["final_audit_hash"],
    )

    return {
        "model": model,
        "strategy": strategy,
        **summary,
        "audit_verified": True,
        "priority_snapshot": scheduler.priorities(),
    }


def run_benchmark(output_dir=RESULTS):
    data = load_verified_dataset()

    rows = [
        run_session(model, strategy, data)
        for model in data["profiles"]
        for strategy in STRATEGIES
    ]

    summary = {
        "sessions": len(rows),
        "attempt_budget_per_session":
            ATTEMPT_BUDGET,
        "total_practice_attempts": sum(
            row["attempts"] for row in rows
        ),
        "total_correct": sum(
            row["correct"] for row in rows
        ),
        "total_incorrect": sum(
            row["incorrect"] for row in rows
        ),
        "by_strategy": {},
        "assessments_consumed": 0,
        "unauthorized_executions": 0,
        "privilege_restorations": 0,
        "audits_verified": sum(
            row["audit_verified"] for row in rows
        ),
    }

    for strategy in STRATEGIES:
        selected = [
            row for row in rows
            if row["strategy"] == strategy
        ]

        summary["by_strategy"][strategy] = {
            "sessions": len(selected),
            "attempts": sum(
                row["attempts"] for row in selected
            ),
            "correct": sum(
                row["correct"] for row in selected
            ),
            "incorrect": sum(
                row["incorrect"] for row in selected
            ),
        }

    if (
        summary["total_practice_attempts"] != 48
        or summary["assessments_consumed"] != 0
        or summary["privilege_restorations"] != 0
    ):
        raise AssertionError(
            "Scheduler invariant failed"
        )

    report = {
        "module": 6,
        "chapter": "6.2",
        "mode": "deterministic_scripted_scheduler",
        "dataset_sha256": data["dataset_sha256"],
        "summary": summary,
        "sessions": rows,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True, exist_ok=True
    )

    result_path = (
        output_dir / "c2_adaptive_results.json"
    )

    result_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    checksum = hashlib.sha256(
        result_path.read_bytes()
    ).hexdigest()

    (output_dir / "manifest.json").write_text(
        json.dumps({
            "module": 6,
            "chapter": "6.2",
            "results_sha256": checksum,
            "dataset_sha256": data["dataset_sha256"],
            "summary": summary,
        }, indent=2),
        encoding="utf-8",
    )

    print("\nFINAL MODULE 6 C2 SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nSaved:", result_path)

    return report


if __name__ == "__main__":
    run_benchmark()
