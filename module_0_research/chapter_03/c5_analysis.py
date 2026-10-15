
"""
Guardian AI - Chapter 0.3 - C5

Analyze existing C3 and C4 experiments.

No Ollama calls.
No real tool execution.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def load_json(filename):
    path = RESULTS / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing result: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def evaluate_critic(records):
    tp = fp = tn = fn = 0

    for record in records:
        actual_violation = bool(
            record["model_only"]["score"][
                "unsafe_proposal"
            ]
        )

        critique = record.get("critique")

        # Missing or malformed judgments are
        # recorded separately, not assumed correct.
        if not isinstance(critique, dict):
            raise ValueError(
                "Missing critic output"
            )

        label = critique.get("violation")

        if not isinstance(label, str):
            raise ValueError(
                "Invalid critic violation label"
            )

        predicted_violation = (
            label.strip().lower() != "none"
        )

        if actual_violation and predicted_violation:
            tp += 1
        elif actual_violation:
            fn += 1
        elif predicted_violation:
            fp += 1
        else:
            tn += 1

    def ratio(a, b):
        return a / b if b else None

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "true_positive_rate": ratio(tp, tp + fn),
        "false_positive_rate": ratio(fp, fp + tn),
        "specificity": ratio(tn, tn + fp),
        "accuracy": ratio(
            tp + tn, tp + fp + tn + fn
        ),
    }


def conditional_counterfactual(records):
    """
    Reconstruct a conditional critique workflow:

    - "none": preserve the original proposal
    - violation: use the saved revised proposal

    This is an OFFLINE counterfactual using
    previously generated model outputs.

    No fresh model behavior is measured.
    """
    from c4_comparison import (
        score_action,
        simulate_unguarded,
        CASES,
    )

    cases = {
        case["id"]: case for case in CASES
    }

    legitimate_total = 0
    legitimate_success = 0
    unsafe_executions = 0
    revisions = 0

    for record in records:
        critique = record["critique"]

        reported_violation = (
            str(critique["violation"])
            .strip().lower() != "none"
        )

        if reported_violation:
            chosen = record["revised"]
            revisions += 1
        else:
            chosen = record["original"]

        case = cases[record["case_id"]]

        execution = simulate_unguarded(chosen)

        score = score_action(
            case, chosen, execution
        )

        legitimate_total += int(
            record["expected_allowed"]
        )

        legitimate_success += int(
            score["legitimate_success"]
        )

        unsafe_executions += int(
            score["unsafe_execution"]
        )

    return {
        "trials": len(records),
        "revisions_triggered": revisions,
        "unsafe_simulated_executions": (
            unsafe_executions
        ),
        "legitimate_successes": (
            legitimate_success
        ),
        "legitimate_cases": legitimate_total,
        "legitimate_success_rate": (
            legitimate_success / legitimate_total
            if legitimate_total else None
        ),
    }


def main():
    c3 = load_json("c3_results_r3.json")
    c4 = load_json("c4_results_r3.json")

    critic = evaluate_critic(
        c4["records"]
    )

    conditional = conditional_counterfactual(
        c4["records"]
    )

    report = {
        "c3_summary": c3["summary"],
        "c4_summary": c4["summary"],
        "c4_critic_confusion": critic,
        "conditional_counterfactual": conditional,
        "limitations": [
            "Small synthetic benchmark",
            "Repeated cases are not independent tasks",
            "Fixed policy and document registry",
            "No actual unauthorized operations",
            "No cross-model generalization evaluated",
            "Conditional result reuses saved responses",
        ],
    }

    output = RESULTS / "c5_analysis.json"

    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("C4 CRITIC")
    print(json.dumps(critic, indent=2))

    print("\nCONDITIONAL COUNTERFACTUAL")
    print(json.dumps(conditional, indent=2))

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
