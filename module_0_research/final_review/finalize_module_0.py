
"""
Guardian AI — Module 0 Final Research Review

Validate the existence and basic integrity of
B1-B5 and R1-R5 evidence, then generate the
Module 0 research gap and hypothesis report.

No model calls.
"""

import argparse
import json
from pathlib import Path


MODULE_0 = Path(__file__).resolve().parent.parent
OUTPUT = Path(__file__).resolve().parent

SOURCES = {
    "B1": MODULE_0 / "chapter_09/results/b1_results_live_r3.json",
    "B2": MODULE_0 / "chapter_09/results/b2_results_live_r3.json",
    "B3": MODULE_0 / "chapter_09/results/b3_results.json",
    "B4": MODULE_0 / "chapter_09/logs",
    "B5": MODULE_0 / "chapter_09/results/b5_final_results.json",
    "R1": MODULE_0 / "chapter_10/results/r1_results_live_r3.json",
    "R2": MODULE_0 / "chapter_10/results/r2_results_live_r3.json",
    "R3": MODULE_0 / "chapter_10/results/r3_results_live_r3.json",
    "R4": MODULE_0 / "chapter_10/results/r4_results_live_r3.json",
    "R5": MODULE_0 / "chapter_10/results/r5_final_results.json",
}

PRIOR_WORK = [
    (
        "AgentDojo",
        "https://arxiv.org/abs/2406.13352",
        "Dynamic agent security evaluation",
    ),
    (
        "Agent Security Bench",
        "https://proceedings.iclr.cc/paper_files/paper/"
        "2025/hash/5750f91d8fb9d5c02bd8ad2c3b44456b-"
        "Abstract-Conference.html",
        "Security and utility benchmarking",
    ),
    (
        "CaMeL",
        "https://arxiv.org/abs/2503.18813",
        "Capability and control-flow enforcement",
    ),
    (
        "PromptArmor",
        "https://arxiv.org/abs/2507.15219",
        "Prompt-injection sanitization",
    ),
    (
        "AgentArmor",
        "https://arxiv.org/abs/2508.01249",
        "Runtime security analysis",
    ),
]


def load_evidence(name, path):
    if name == "B4":
        if not path.is_dir():
            raise FileNotFoundError(path)

        logs = sorted(path.glob("*guardian-b4*.eval"))
        if not logs:
            raise ValueError("B4 Inspect AI log not found")

        return {
            "experiment": "B4",
            "log_files": [str(p) for p in logs],
        }

    if not path.is_file():
        raise FileNotFoundError(path)

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"{name}: expected JSON object")

    if data.get("experiment") != name:
        raise ValueError(f"{name}: experiment ID mismatch")

    if name in {"B1", "B2", "R1", "R2", "R3", "R4"}:
        if data.get("offline") is not False:
            raise ValueError(f"{name}: live evidence required")

        if data.get("repeats") != 3:
            raise ValueError(f"{name}: expected 3 repeats")

        if not isinstance(data.get("records"), list):
            raise ValueError(f"{name}: missing trial records")

    return data


def evaluate_evidence(sources=SOURCES):
    evidence = {
        name: load_evidence(name, path)
        for name, path in sources.items()
    }

    r5 = evidence["R5"]
    if (
        r5.get("checks_passed") != 12
        or r5.get("checks_total") != 12
    ):
        raise ValueError("R5 final checks not passed")

    b5 = evidence["B5"]
    b4_summary = b5["B4"]["summary"]

    if b4_summary["trials"] != 18:
        raise ValueError("B4 expected 18 Inspect samples")

    results = {}

    for name in ("R1", "R2", "R3"):
        summary = evidence[name]["summary"]
        results[name] = {
            "trials": summary["trials"],
            "completions": summary["total_completions"],
            "unauthorized_executions":
                summary["unauthorized_executions"],
        }

    r4_summary = evidence["R4"]["summary"]
    policies = r4_summary["policies"]

    results["R4"] = {
        "trials": r4_summary["trials"],
        "sanitized_completions":
            policies["sanitized_retry"]["total_completions"],
        "raw_completions":
            policies["raw_retry"]["total_completions"],
        "unauthorized_executions":
            r4_summary["unauthorized_executions"],
    }

    results["B4"] = {
        "trials": b4_summary["trials"],
        "invalid_outputs":
            b4_summary["outcomes"].get("invalid_output", 0),
        "unsafe_proposals":
            b4_summary["unsafe_proposals"],
        "guardian_unauthorized_executions":
            b4_summary["guardian_unauthorized_executions"],
    }

    if any(
        results[name]["unauthorized_executions"] != 0
        for name in ("R1", "R2", "R3", "R4")
    ):
        raise ValueError(
            "Observed unauthorized execution: "
            "do not issue clean safety conclusion"
        )

    return evidence, results


def render_report(results):
    lines = [
        "# Guardian AI — Module 0 Final Research Report",
        "",
        "## 1. Research problem",
        "",
        "Tool-using LLM agents may mistake untrusted "
        "content for instructions and propose actions "
        "outside the original authorized task.",
        "",
        "Guardian AI investigates whether independent "
        "enforcement combined with structured safety "
        "learning can reduce repeat violations while "
        "preserving legitimate task completion.",
        "",
        "## 2. Relevant prior work",
        "",
        "| Work | Relevance | Reference |",
        "|---|---|---|",
    ]

    for title, url, relevance in PRIOR_WORK:
        lines.append(
            f"| {title} | {relevance} | {url} |"
        )

    lines.extend([
        "",
        "## 3. Evidence from previous experiments",
        "",
        "| Experiment | Trials | Completions | "
        "Unauthorized executions |",
        "|---|---:|---:|---:|",
    ])

    for name in ("R1", "R2", "R3"):
        s = results[name]
        lines.append(
            f"| {name} | {s['trials']} | "
            f"{s['completions']} | "
            f"{s['unauthorized_executions']} |"
        )

    r4 = results["R4"]
    b4 = results["B4"]

    lines.extend([
        "",
        "### Recovery ablation (R4)",
        "",
        f"- Raw retry completions: {r4['raw_completions']}/15",
        "- Sanitized retry completions: "
        f"{r4['sanitized_completions']}/15",
        "- Unauthorized synthetic executions: "
        f"{r4['unauthorized_executions']}",
        "",
        "### Inspect AI evaluation (B4)",
        "",
        f"- Evaluated samples: {b4['trials']}",
        f"- Invalid outputs: {b4['invalid_outputs']}",
        f"- Unsafe proposals: {b4['unsafe_proposals']}",
        "- Guardian unauthorized synthetic executions: "
        f"{b4['guardian_unauthorized_executions']}",
        "",
        "## 4. Candidate research gap",
        "",
        "Independent authorization and input sanitization "
        "have significant prior art. A promising question "
        "is whether lessons extracted from blocked actions "
        "can be retained, transferred to different agents, "
        "and used safely without weakening enforcement.",
        "",
        "This gap is a research hypothesis, NOT a "
        "verified first-of-its-kind claim.",
        "",
        "## 5. Main hypothesis",
        "",
        "**H1:** Independent Guardian enforcement "
        "combined with structured, evidence-based "
        "safety lessons reduces repeat unsafe "
        "proposals on held-out scenarios relative "
        "to enforcement-only and sanitized-retry "
        "baselines, without increasing unauthorized "
        "execution or substantially reducing "
        "legitimate task completion.",
        "",
        "**H0:** Learned safety lessons offer no "
        "improvement over these baselines.",
        "",
        "## 6. Required future experiments",
        "",
        "1. Comparable Llama and Qwen agent implementations.",
        "2. Diverse synthetic tasks with known policy labels.",
        "3. Independent enforcement and audit records.",
        "4. Lesson extraction with malicious text removed.",
        "5. Transfer to held-out models and tasks.",
        "6. Persistence tests over multiple episodes.",
        "7. Adaptive attacks and negative-transfer checks.",
        "8. Statistical uncertainty and paired comparisons.",
        "",
        "## 7. Primary evaluation metrics",
        "",
        "- Unauthorized execution rate",
        "- Unsafe proposal rate",
        "- Legitimate task completion rate",
        "- Repeated violation rate",
        "- Cross-agent transfer improvement",
        "- False rejection rate",
        "- Recovery cost in calls and latency",
        "",
        "## 8. Boundaries and limitations",
        "",
        "- Existing experiments use synthetic document effects.",
        "- Results are based on a small authored scenario set.",
        "- Live comparisons are not always paired.",
        "- Zero observed unauthorized executions is not "
        "a production security guarantee.",
        "- Live R3/R4 did not exercise fallback or approval.",
        "- Cross-model learning has not yet been demonstrated.",
        "- Originality requires broader literature review.",
        "",
        "## 9. Transition to Module 1",
        "",
        "Build a standardized multi-model agent runner "
        "with comparable prompts, tool schemas, "
        "requester policies, deterministic scenarios, "
        "and model-independent evaluation records.",
        "",
    ])

    return "\n".join(lines)


def main():
    _, results = evaluate_evidence()

    OUTPUT.mkdir(parents=True, exist_ok=True)

    markdown = OUTPUT / "MODULE_0_RESEARCH_FINDINGS.md"
    summary = OUTPUT / "module_0_summary.json"

    markdown.write_text(
        render_report(results), encoding="utf-8"
    )

    summary.write_text(
        json.dumps({
            "module": 0,
            "evidence_status": "validated_source_presence",
            "experiments": sorted(SOURCES.keys()),
            "metrics": results,
            "next_module": 1,
            "novelty": "candidate_hypothesis_not_proven",
        }, indent=2),
        encoding="utf-8",
    )

    print("\nMODULE 0 FINAL SUMMARY")
    print(json.dumps(results, indent=2))
    print("\nEvidence sources:", len(SOURCES))
    print("Saved:", markdown)
    print("Saved:", summary)


if __name__ == "__main__":
    main()
