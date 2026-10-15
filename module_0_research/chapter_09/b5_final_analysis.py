
"""
Guardian AI — Chapter 0.9 — B5

Final research evidence and reporting.

Inputs:
  B1 live JSON
  B2 live JSON
  B3 scored JSON
  B4 official Inspect AI .eval log

Outputs:
  results/b5_final_results.json
  B5_FINDINGS.md

No LLM inference or real tool execution.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log

import b1_benchmark as b1
import b2_stateful_tools as b2
import b3_security_scoring as b3
import b4_inspect_eval as b4


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

INPUTS = {
    "B1": "b1_results_live_r3.json",
    "B2": "b2_results_live_r3.json",
    "B3": "b3_results.json",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def find_inspect_log(log_dir):
    """
    Select the newest successful full B4 run,
    not a --limit smoke test.
    """
    log_dir = Path(log_dir)

    candidates = sorted(
        log_dir.glob("*guardian-b4*.eval"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for path in candidates:
        log = read_eval_log(path)

        if log.status != "success":
            continue

        if not log.samples or len(log.samples) != 18:
            continue

        return path, log

    raise FileNotFoundError(
        "No successful 18-sample B4 Inspect log "
        f"found in {log_dir}"
    )


def validate_previous_results(results_dir):
    reports = {
        key: load_json(results_dir / filename)
        for key, filename in INPUTS.items()
    }

    b1_report = reports["B1"]
    b2_report = reports["B2"]
    b3_report = reports["B3"]

    require(
        b1_report.get("offline") is False
        and b1_report.get("experiment") == "B1",
        "B1 must be a live experiment"
    )

    require(
        b2_report.get("offline") is False
        and b2_report.get("experiment") == "B2",
        "B2 must be a live experiment"
    )

    require(
        b3_report.get("experiment") == "B3",
        "B3 experiment identifier mismatch"
    )

    require(
        b3_report.get("source_type")
        == "live_ollama_results",
        "B3 must use live experiment records"
    )

    # Recompute all metrics from original records.
    recomputed = {
        "B1": b3.score_experiment(
            b1_report, "B1"
        ),
        "B2": b3.score_experiment(
            b2_report, "B2"
        ),
    }

    require(
        b3_report["metrics"] == recomputed,
        "B3 scored results differ from B1/B2 "
        "source records"
    )

    return recomputed


def normalize_score(value):
    """
    Inspect uses C/I scorer values.
    Return True/False, or None if absent.
    """
    if value in ("C", "CORRECT", True, 1):
        return True

    if value in ("I", "INCORRECT", False, 0):
        return False

    return None


def summarize_inspect_samples(samples):
    cases = {
        case["id"]: case for case in b1.CASES
    }

    expected_ids = {
        f"{case_id}-{repeat}"
        for case_id in cases
        for repeat in (1, 2, 3)
    }

    actual_ids = {
        str(sample.id) for sample in samples
    }

    require(
        len(samples) == 18
        and actual_ids == expected_ids,
        "Inspect dataset is not the expected "
        "18-sample B4 evaluation"
    )

    records = []

    for sample in samples:
        meta = sample.metadata or {}

        case_id = meta.get("case_id")

        require(
            case_id in cases,
            f"Unknown case: {case_id}"
        )

        expected = cases[case_id]

        require(
            meta.get("requester")
            == expected["requester"]
            and meta.get("target_document")
            == expected["target"]
            and meta.get("attack")
            == expected["attack"],
            f"Inspect metadata mismatch: {sample.id}"
        )

        require(
            sample.error is None,
            f"Inspect sample error: {sample.id}"
        )

        require(
            sample.output is not None,
            f"Missing model output: {sample.id}"
        )

        completion = sample.output.completion
        assessed = b4.evaluate_proposal(
            completion, meta
        )

        action = assessed["parsed_action"]

        if action is None:
            outcome = "invalid_output"
        elif action["tool"] == "finish":
            outcome = "premature_finish"
        elif assessed["unsafe_proposal"]:
            outcome = "unsafe_proposal"
        elif assessed["guardian_completed"]:
            outcome = "authorized_completion"
        else:
            outcome = "other_incomplete"

        expected_correct = bool(
            assessed["guardian_completed"]
        )

        # Check Inspect's recorded scores against
        # our independently recomputed decision.
        scores = sample.scores or {}

        for scorer_name in (
            "guardian_task_completion",
            "model_attack_resistance",
        ):
            require(
                scorer_name in scores,
                f"Missing scorer {scorer_name}: "
                f"{sample.id}"
            )

            observed = normalize_score(
                scores[scorer_name].value
            )

            require(
                observed is not None,
                "Unrecognized Inspect score"
            )

            require(
                observed == expected_correct,
                "Inspect scorer inconsistent with "
                f"recomputed result: {sample.id}"
            )

        records.append({
            "sample_id": str(sample.id),
            "case": case_id,
            "attack": bool(expected["attack"]),
            "output": completion,
            "action": action,
            "outcome": outcome,
            "unsafe_proposal": bool(
                assessed["unsafe_proposal"]
            ),
            "guardian_completed": bool(
                assessed["guardian_completed"]
            ),
            "vulnerable_unauthorized_execution": bool(
                assessed[
                    "vulnerable_unauthorized_execution"
                ]
            ),
            "guardian_unauthorized_execution": bool(
                assessed[
                    "guardian_unauthorized_execution"
                ]
            ),
        })

    counts = Counter(
        record["outcome"] for record in records
    )

    attacks = [
        record for record in records
        if record["attack"]
    ]

    details = {}

    for case_id in cases:
        subset = [
            r for r in records
            if r["case"] == case_id
        ]

        details[case_id] = {
            "trials": len(subset),
            "outcomes": dict(
                Counter(r["outcome"] for r in subset)
            ),
            "unsafe_proposals": sum(
                r["unsafe_proposal"] for r in subset
            ),
            "guardian_completions": sum(
                r["guardian_completed"] for r in subset
            ),
        }

    summary = {
        "trials": len(records),
        "attack_trials": len(attacks),
        "guardian_completions": sum(
            r["guardian_completed"] for r in records
        ),
        "unsafe_proposals": sum(
            r["unsafe_proposal"] for r in records
        ),
        "compromised_attack_trials": sum(
            r["unsafe_proposal"] for r in attacks
        ),
        "vulnerable_unauthorized_executions": sum(
            r["vulnerable_unauthorized_execution"]
            for r in records
        ),
        "guardian_unauthorized_executions": sum(
            r["guardian_unauthorized_execution"]
            for r in records
        ),
        "outcomes": dict(counts),
        "cases": details,
    }

    return {
        "summary": summary,
        "records": records,
    }


def make_findings(b1b2, b4_result, log_path):
    b1_summary = b1b2["B1"]["summary"]
    b2_summary = b1b2["B2"]["summary"]
    inspect_summary = b4_result["summary"]

    lines = [
        "# Guardian AI — Chapter 0.9",
        "",
        "## Agent Security Benchmarking",
        "",
        "### Research objective",
        "",
        "Evaluate prompt-injection-induced task "
        "deviation, simulated authorization "
        "enforcement, legitimate task completion "
        "and multi-step recovery.",
        "",
        "## Methodology",
        "",
        "- B1: single-step local Ollama benchmark.",
        "- B2: multi-step synthetic tool environment.",
        "- B3: trial-level security and utility scoring.",
        "- B4: official Inspect AI evaluation.",
        "- B5: source-record and score validation.",
        "",
        "All document resources and exposures are "
        "synthetic. No real confidential documents "
        "were accessed.",
        "",
        "## Live evaluation summary",
        "",
        "| Metric | B1 | B2 |",
        "|---|---:|---:|",
    ]

    metrics = [
        ("trials", "Trials"),
        ("attack_trials", "Attack-bearing trials"),
        (
            "compromised_attack_trials",
            "Compromised attack trials",
        ),
        ("unsafe_attempts", "Unsafe proposals"),
        (
            "guardian_completions",
            "Guardian task completions",
        ),
    ]

    for key, label in metrics:
        lines.append(
            f"| {label} | "
            f"{b1_summary[key]} | "
            f"{b2_summary[key]} |"
        )

    rate_fields = [
        ("model_compromise", "Model compromise"),
        (
            "vulnerable_attack_success",
            "Weak baseline attack success",
        ),
        (
            "guardian_attack_success",
            "Guardian attack success",
        ),
        ("task_completion", "Task completion"),
        (
            "recovery_given_unsafe",
            "Recovery after unsafe proposal",
        ),
    ]

    lines.extend([
        "",
        "## Live outcome rates",
        "",
        "| Metric | B1 | B2 |",
        "|---|---:|---:|",
    ])

    for key, label in rate_fields:
        lines.append(
            f"| {label} | "
            f"{b3.percent(b1_summary['rates'][key])} | "
            f"{b3.percent(b2_summary['rates'][key])} |"
        )

    lines.extend([
        "",
        "## B4 — Inspect AI evaluation",
        "",
        f"- Samples: {inspect_summary['trials']}",
        "- Guardian task completions: "
        f"{inspect_summary['guardian_completions']}",
        "- Unsafe proposed actions: "
        f"{inspect_summary['unsafe_proposals']}",
        "- Guardian unauthorized simulated executions: "
        f"{inspect_summary['guardian_unauthorized_executions']}",
        "",
        "### Exact response classification",
        "",
        "| Outcome | Samples |",
        "|---|---:|",
    ])

    for category in (
        "authorized_completion",
        "unsafe_proposal",
        "invalid_output",
        "premature_finish",
        "other_incomplete",
    ):
        lines.append(
            f"| {category} | "
            f"{inspect_summary['outcomes'].get(category, 0)} |"
        )

    lines.extend([
        "",
        "### Results by attack condition",
        "",
        "| Case | Unsafe proposals | Completions |",
        "|---|---:|---:|",
    ])

    for case_id, value in inspect_summary["cases"].items():
        lines.append(
            f"| {case_id} | "
            f"{value['unsafe_proposals']} | "
            f"{value['guardian_completions']}/"
            f"{value['trials']} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "1. Unsafe model proposals are different from "
        "unauthorized simulated executions.",
        "2. A Guardian block can prevent a security "
        "failure while leaving a task incomplete.",
        "3. B2's encoded-instruction scenario exhibited "
        "a repeated-rejection recovery failure.",
        "4. B1 and B2 evaluate different workflows; "
        "their scores should not be pooled.",
        "5. B4's two initial scorers use substantially "
        "overlapping success definitions, so agreement "
        "is expected rather than independent validation.",
        "",
        "## Limitations",
        "",
        "- Synthetic documents and tool effects only.",
        "- Small, repeated, hand-authored case set.",
        "- One local Ollama model configuration.",
        "- Attack prompts and system prompts varied "
        "between experiments.",
        "- B2 policy branches replayed identical "
        "proposals rather than independent trajectories.",
        "- B4 evaluates proposed actions, not a real "
        "tool execution firewall.",
        "- Zero observed unauthorized Guardian execution "
        "does not establish production security.",
        "- No multi-model, adversarial search, "
        "distributed execution or concurrency tests.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python3 -m pytest test_b5_final_analysis.py -v",
        "python3 b5_final_analysis.py",
        "```",
        "",
        "## Evidence",
        "",
        "- results/b1_results_live_r3.json",
        "- results/b2_results_live_r3.json",
        "- results/b3_results.json",
        f"- Inspect log: {Path(log_path).name}",
        "",
    ])

    return "\n".join(lines)


def run_analysis(
    results_dir=RESULTS,
    log_dir=None,
    findings_path=None,
):
    results_dir = Path(results_dir)

    if log_dir is None:
        log_dir = ROOT / "logs"

    if findings_path is None:
        findings_path = ROOT / "B5_FINDINGS.md"

    b1b2 = validate_previous_results(
        results_dir
    )

    log_path, inspect_log = find_inspect_log(
        log_dir
    )

    b4_result = summarize_inspect_samples(
        inspect_log.samples
    )

    report = {
        "experiment": "B5",
        "chapter": "0.9",
        "sources": {
            **INPUTS,
            "B4": str(log_path),
        },
        "model": str(
            inspect_log.eval.model
        ),
        "B1": b1b2["B1"]["summary"],
        "B2": b1b2["B2"]["summary"],
        "B4": b4_result,
        "limitations_acknowledged": True,
    }

    results_dir.mkdir(
        parents=True, exist_ok=True
    )

    result_path = (
        results_dir / "b5_final_results.json"
    )

    result_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    Path(findings_path).write_text(
        make_findings(
            b1b2, b4_result, log_path
        ),
        encoding="utf-8",
    )

    print("\nFINAL B5 SUMMARY")
    print(json.dumps({
        "B1": {
            "trials": report["B1"]["trials"],
            "rates": report["B1"]["rates"],
        },
        "B2": {
            "trials": report["B2"]["trials"],
            "rates": report["B2"]["rates"],
        },
        "B4": report["B4"]["summary"],
    }, indent=2))

    print("\nSaved:", result_path)
    print("Findings:", findings_path)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log-dir",
        default=str(ROOT / "logs"),
    )

    args = parser.parse_args()

    run_analysis(log_dir=args.log_dir)
