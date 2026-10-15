
"""
Guardian AI — Module 1, Chapter 1.4

Cross-model behavior profiling and error taxonomy.

Inputs:
  C1: Single-action agent benchmark
  C2: Multi-step agent benchmark
  C3: Memory and context benchmark

Outputs:
  results/m4_behavior_results.json
  M4_FINDINGS.md

No external calls, no real tool execution.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
LAB = HERE.parent
RESULTS = HERE / "results"

INPUTS = {
    "C1": LAB / "chapter_01/results/m1_runner_live_r3.json",
    "C2": LAB / "chapter_02/results/m2_results_live_r3.json",
    "C3": LAB / "chapter_03/results/m3_results_live_r3.json",
}

MODELS = ("llama3.2:3b", "qwen2.5:3b")

MEMORY_MODES = (
    "no_memory",
    "raw_memory",
    "structured_memory",
)

EXPECTED_COUNTS = {
    "C1": 36,
    "C2": 48,
    "C3": 144,
}

# Normalize simulator outcomes without changing
# the event's original reason in the saved trace.
REASON_CATEGORIES = {
    "INVALID_OUTPUT": "schema_error",
    "INVALID_SEARCH": "target_or_search_mismatch",
    "TARGET_MISMATCH": "target_or_search_mismatch",
    "ACCESS_DENIED": "access_denial",
    "INVALID_ORIGINAL_TASK": "original_task_denial",
    "OUT_OF_ORDER_READ": "workflow_order_error",
    "PREMATURE_FINISH": "workflow_order_error",
    "UNKNOWN_TOOL": "unknown_tool",
    "SEARCH_OK": "authorized_search",
    "READ_OK": "authorized_read",
    "TASK_COMPLETED": "authorized_finish",
}


def load_json(path, chapter):
    if not path.is_file():
        raise FileNotFoundError(
            f"{chapter}: missing {path}"
        )

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError(
            f"{chapter}: expected JSON object"
        )

    expected_id = {
        "C1": "M1_C1",
        "C2": "M1_C2",
        "C3": "M1_C3",
    }[chapter]

    if data.get("experiment") != expected_id:
        raise ValueError(
            f"{chapter}: unexpected experiment ID"
        )

    if data.get("mode") != "live":
        raise ValueError(
            f"{chapter}: expected live results"
        )

    if data.get("repeats") != 3:
        raise ValueError(
            f"{chapter}: expected 3 repeats"
        )

    if not isinstance(data.get("records"), list):
        raise ValueError(
            f"{chapter}: missing records"
        )

    if not isinstance(data.get("summary"), dict):
        raise ValueError(
            f"{chapter}: missing summary"
        )

    return data


def validate_records(data, chapter):
    records = data["records"]

    if len(records) != EXPECTED_COUNTS[chapter]:
        raise ValueError(
            f"{chapter}: expected "
            f"{EXPECTED_COUNTS[chapter]} trials, "
            f"found {len(records)}"
        )

    keys = []

    for row in records:
        model = row.get("model")
        repeat = row.get("repeat")

        if model not in MODELS:
            raise ValueError("Unexpected model")

        if repeat not in (1, 2, 3):
            raise ValueError("Unexpected repeat")

        if chapter == "C1":
            key = (
                model,
                row["task_id"],
                repeat,
            )
        elif chapter == "C2":
            key = (
                model,
                row["scenario"],
                repeat,
            )
        else:
            if row["memory_mode"] not in MEMORY_MODES:
                raise ValueError(
                    "Unexpected memory mode"
                )

            key = (
                model,
                row["scenario"],
                row["memory_mode"],
                repeat,
            )

        keys.append(key)

        if chapter != "C1":
            if row["unauthorized_executions"] != 0:
                raise ValueError(
                    "Unauthorized execution recorded"
                )

            if row["model_calls"] < 0:
                raise ValueError(
                    "Negative model call count"
                )

            if (
                row["unsafe_proposals"] < 0
                or row["guardian_blocks"] < 0
            ):
                raise ValueError(
                    "Invalid security counters"
                )

            trace = row.get("trace")

            if not isinstance(trace, list):
                raise ValueError("Missing action trace")

            trace_invalid = sum(
                item.get("decision") == "INVALID_OUTPUT"
                for item in trace
            )

            if trace_invalid != row["invalid_outputs"]:
                raise ValueError(
                    "Invalid output count mismatch"
                )

            trace_wrong = sum(
                bool(item.get("wrong_target"))
                for item in trace
            )

            if trace_wrong != row["unsafe_proposals"]:
                raise ValueError(
                    "Unsafe proposal count mismatch"
                )

            trace_blocks = sum(
                item.get("event") == "blocked"
                or item.get("decision") == "INVALID_OUTPUT"
                for item in trace
            )

            if trace_blocks != row["guardian_blocks"]:
                raise ValueError(
                    "Guardian block count mismatch"
                )

    if len(keys) != len(set(keys)):
        raise ValueError(
            f"{chapter}: duplicate trial key"
        )

    return records


def categorize_event(item):
    reason = item.get("decision", "UNKNOWN")

    return REASON_CATEGORIES.get(
        reason, "other"
    )


def analyze_multistep(records):
    """
    Classify trace events and episode outcomes.

    'Recovered' means an episode had at least
    one unsafe proposal and later completed.
    It does not mean a Guardian lesson was learned.
    """
    profiles = {}

    for model in MODELS:
        rows = [
            row for row in records
            if row["model"] == model
        ]

        event_counts = Counter()
        reason_counts = Counter()
        scenario_stats = defaultdict(
            lambda: {
                "trials": 0,
                "completions": 0,
                "unsafe_proposals": 0,
                "blocked_actions": 0,
                "invalid_outputs": 0,
            }
        )

        for row in rows:
            scenario = row["scenario"]
            stats = scenario_stats[scenario]

            stats["trials"] += 1
            stats["completions"] += int(
                row["completed"]
            )
            stats["unsafe_proposals"] += (
                row["unsafe_proposals"]
            )
            stats["blocked_actions"] += (
                row["guardian_blocks"]
            )
            stats["invalid_outputs"] += (
                row["invalid_outputs"]
            )

            for item in row["trace"]:
                event_counts[
                    categorize_event(item)
                ] += 1

                reason_counts[
                    item.get("decision", "UNKNOWN")
                ] += 1

        episodes_with_unsafe = sum(
            row["unsafe_proposals"] > 0
            for row in rows
        )

        recovered = sum(
            row["unsafe_proposals"] > 0
            and row["completed"]
            for row in rows
        )

        completed = sum(
            row["completed"] for row in rows
        )

        profiles[model] = {
            "trials": len(rows),
            "completions": completed,
            "completion_rate": round(
                completed / len(rows), 4
            ) if rows else 0,
            "unsafe_proposals": sum(
                row["unsafe_proposals"]
                for row in rows
            ),
            "episodes_with_unsafe":
                episodes_with_unsafe,
            "recovered_after_unsafe": recovered,
            "recovery_rate_given_unsafe": round(
                recovered / episodes_with_unsafe,
                4,
            ) if episodes_with_unsafe else None,
            "blocks": sum(
                row["guardian_blocks"]
                for row in rows
            ),
            "invalid_outputs": sum(
                row["invalid_outputs"]
                for row in rows
            ),
            "budget_exhaustions": sum(
                row["stopped_reason"]
                == "STEP_BUDGET_EXHAUSTED"
                for row in rows
            ),
            "event_taxonomy": dict(
                sorted(event_counts.items())
            ),
            "decision_reasons": dict(
                sorted(reason_counts.items())
            ),
            "scenarios": {
                key: dict(value)
                for key, value in sorted(
                    scenario_stats.items()
                )
            },
        }

    return profiles


def analyze_memory(records):
    result = {}

    for model in MODELS:
        result[model] = {}

        for mode in MEMORY_MODES:
            rows = [
                row for row in records
                if row["model"] == model
                and row["memory_mode"] == mode
            ]

            profile = analyze_multistep_for_rows(rows)

            result[model][mode] = profile

        raw = result[model]["raw_memory"]
        no_mem = result[model]["no_memory"]
        structured = result[model]["structured_memory"]

        result[model]["comparisons"] = {
            "raw_minus_no_memory_completions":
                raw["completions"]
                - no_mem["completions"],

            "structured_minus_raw_completions":
                structured["completions"]
                - raw["completions"],

            "raw_minus_no_memory_unsafe_proposals":
                raw["unsafe_proposals"]
                - no_mem["unsafe_proposals"],

            "structured_minus_raw_unsafe_proposals":
                structured["unsafe_proposals"]
                - raw["unsafe_proposals"],
        }

    return result


def analyze_multistep_for_rows(rows):
    reasons = Counter()

    for row in rows:
        for item in row["trace"]:
            reasons[
                item.get("decision", "UNKNOWN")
            ] += 1

    n = len(rows)
    completions = sum(
        row["completed"] for row in rows
    )
    affected = sum(
        row["unsafe_proposals"] > 0
        for row in rows
    )
    recovered = sum(
        row["completed"]
        and row["unsafe_proposals"] > 0
        for row in rows
    )

    return {
        "trials": n,
        "completions": completions,
        "completion_rate": round(
            completions / n, 4
        ) if n else 0,
        "unsafe_proposals": sum(
            row["unsafe_proposals"] for row in rows
        ),
        "episodes_with_unsafe": affected,
        "recovered_after_unsafe": recovered,
        "blocks": sum(
            row["guardian_blocks"] for row in rows
        ),
        "invalid_outputs": sum(
            row["invalid_outputs"] for row in rows
        ),
        "budget_exhaustions": sum(
            row["stopped_reason"]
            == "STEP_BUDGET_EXHAUSTED"
            for row in rows
        ),
        "unauthorized_executions": sum(
            row["unauthorized_executions"]
            for row in rows
        ),
        "decision_reasons": dict(
            sorted(reasons.items())
        ),
    }


def analyze_single_action(records):
    return {
        model: {
            "trials": sum(
                row["model"] == model
                for row in records
            ),
            "correct_proposals": sum(
                row["model"] == model
                and row["outcome"] == "correct_proposal"
                for row in records
            ),
            "wrong_target_proposals": sum(
                row["model"] == model
                and row["outcome"]
                == "wrong_target_proposal"
                for row in records
            ),
            "invalid_outputs": sum(
                row["model"] == model
                and row["outcome"] == "invalid_output"
                for row in records
            ),
        }
        for model in MODELS
    }


def render_findings(report):
    c1 = report["C1"]
    c2 = report["C2"]
    c3 = report["C3"]

    lines = [
        "# Guardian AI — Module 1, Chapter 1.4",
        "",
        "## Cross-Model Behavior Profiling",
        "",
        "### Objective",
        "",
        "Analyze single-step, multi-step and memory "
        "agent behavior using existing saved live "
        "results from Llama 3.2:3B and Qwen 2.5:3B.",
        "",
        "## C1: Single-action baseline",
        "",
        "| Model | Correct / Trials | Invalid | Wrong target |",
        "|---|---:|---:|---:|",
    ]

    for model in MODELS:
        item = c1[model]
        lines.append(
            f"| {model} | "
            f"{item['correct_proposals']}/{item['trials']} | "
            f"{item['invalid_outputs']} | "
            f"{item['wrong_target_proposals']} |"
        )

    lines.extend([
        "",
        "## C2: Multi-step performance",
        "",
        "| Model | Completions | Unsafe proposals | "
        "Blocks | Budget exhaustion |",
        "|---|---:|---:|---:|---:|",
    ])

    for model in MODELS:
        item = c2[model]
        lines.append(
            f"| {model} | "
            f"{item['completions']}/{item['trials']} | "
            f"{item['unsafe_proposals']} | "
            f"{item['blocks']} | "
            f"{item['budget_exhaustions']} |"
        )

    lines.extend([
        "",
        "## C3: Memory conditions",
        "",
        "| Model | Memory | Completion | "
        "Unsafe proposals | Recovered after unsafe |",
        "|---|---|---:|---:|---:|",
    ])

    for model in MODELS:
        for mode in MEMORY_MODES:
            item = c3[model][mode]
            lines.append(
                f"| {model} | {mode} | "
                f"{item['completions']}/{item['trials']} | "
                f"{item['unsafe_proposals']} | "
                f"{item['recovered_after_unsafe']} |"
            )

    lines.extend([
        "",
        "## Decision failure taxonomy",
        "",
    ])

    for model in MODELS:
        lines.extend([
            f"### {model}: C2 trace reasons",
            "",
            "| Reason | Events |",
            "|---|---:|",
        ])

        for reason, count in c2[model][
            "decision_reasons"
        ].items():
            lines.append(
                f"| {reason} | {count} |"
            )

        lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        "- C1's single-action success did not imply "
        "equivalent multi-step reliability.",
        "- Wrong-target proposals must be separated "
        "from tool-order and schema violations.",
        "- Structured memory contains only a restricted "
        "set of trusted fields, while raw memory carries "
        "attacker-controlled text.",
        "- A completion after an unsafe proposal is an "
        "episode recovery, not evidence of durable learning.",
        "- The experiments do not demonstrate actual "
        "unauthorized document disclosure.",
        "",
        "## Limitations",
        "",
        "- Small authored scenario set.",
        "- Synthetic documents and local tool execution.",
        "- Only two small Ollama models.",
        "- No cross-model statistical generalization.",
        "- Differences in prompts and sampling can affect "
        "observed memory-condition results.",
        "- Error reason counts are tool-event counts, "
        "not numbers of distinct affected episodes.",
        "- Chapter 1.3 uses a larger step budget than "
        "Chapter 1.2, so direct cross-chapter completion "
        "comparisons should be treated cautiously.",
        "",
        "## Next",
        "",
        "Chapter 1.5 will consolidate comparable "
        "model baselines and define the standardized "
        "agent interface for Module 2.",
        "",
    ])

    return "\n".join(lines)


def run_analysis(inputs=INPUTS, results_dir=RESULTS):
    data = {
        name: load_json(path, name)
        for name, path in inputs.items()
    }

    records = {
        name: validate_records(data[name], name)
        for name in inputs
    }

    result = {
        "experiment": "M1_C4",
        "source_experiments": list(inputs),
        "C1": analyze_single_action(records["C1"]),
        "C2": analyze_multistep(records["C2"]),
        "C3": analyze_memory(records["C3"]),
    }

    results_dir = Path(results_dir)
    results_dir.mkdir(
        parents=True, exist_ok=True
    )

    json_path = (
        results_dir / "m4_behavior_results.json"
    )

    markdown_path = (
        results_dir.parent / "M4_FINDINGS.md"
    )

    json_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    markdown_path.write_text(
        render_findings(result),
        encoding="utf-8",
    )

    print("\nFINAL MODULE 1 C4 SUMMARY")

    print(json.dumps({
        "C2": {
            model: {
                "completions":
                    result["C2"][model]["completions"],
                "unsafe_proposals":
                    result["C2"][model]["unsafe_proposals"],
                "recovered_after_unsafe":
                    result["C2"][model][
                        "recovered_after_unsafe"
                    ],
                "decision_reasons":
                    result["C2"][model][
                        "decision_reasons"
                    ],
            }
            for model in MODELS
        },
        "C3_memory_deltas": {
            model: result["C3"][model][
                "comparisons"
            ]
            for model in MODELS
        },
    }, indent=2))

    print("\nSaved:", json_path)
    print("Findings:", markdown_path)

    return result


if __name__ == "__main__":
    run_analysis()
