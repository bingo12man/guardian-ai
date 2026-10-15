
"""
Module 7.2 deterministic collective memory benchmark.

No Ollama calls or new model observations.

Expected:
  30 observations
   7 model-session-lesson units
   0 permission changes

Exact number of distinct collective lesson IDs
is determined from trusted source records.
"""

import copy
import json
from pathlib import Path

import guardian_collective_memory as memory

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def run_benchmark(output_dir=RESULTS,
                  findings_dir=HERE):
    source = memory.load_verified_sources()

    annotated = memory.annotate_observations(
        source["observations"],
        source["exercises"],
    )
    units = memory.build_evidence_units(annotated)
    collective = memory.build_collective_memory(units)

    memory.validate_memory(
        annotated, units, collective
    )

    summary = memory.make_summary(
        annotated, units, collective
    )

    memory.require(
        summary["verified_input_observations"] == 30,
        "Expected 30 source observations",
    )
    memory.require(
        summary["correlated_evidence_units"] == 7,
        "Expected seven model/lesson evidence units",
    )
    memory.require(
        summary["units_by_model"] == {
            "llama3.2:3b": 4,
            "qwen2.5:3b": 3,
        },
        "Unexpected model-specific evidence units",
    )

    # Tampering the aggregate, without changing
    # original observations, must be detectable.
    tampered = copy.deepcopy(collective)
    tampered[0]["correct"] += 1

    tamper_detected = False
    try:
        memory.validate_memory(
            annotated, units, tampered
        )
    except ValueError:
        tamper_detected = True

    memory.require(
        tamper_detected,
        "Altered aggregate was accepted",
    )

    summary["aggregate_tampering_detected"] = True

    output_dir = Path(output_dir)
    findings_dir = Path(findings_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    findings_dir.mkdir(parents=True, exist_ok=True)

    units_path = (
        output_dir / "correlated_evidence_units.json"
    )
    collective_path = (
        output_dir / "collective_safety_memory.json"
    )
    summary_path = (
        output_dir / "m7_2_summary.json"
    )
    manifest_path = output_dir / "manifest.json"

    memory.write_json(units_path, units)
    memory.write_json(collective_path, collective)
    memory.write_json(summary_path, summary)

    memory.write_json(manifest_path, {
        "module": 7,
        "chapter": "7.2",
        "source_registry_audit_head":
            source["registry_audit_head"],
        "dataset_sha256": source["dataset_sha256"],
        "evidence_units_sha256": memory.digest(units),
        "collective_memory_sha256":
            memory.digest(collective),
        "summary_sha256": memory.digest(summary),
    })

    lines = [
        "# Guardian AI — Module 7.2 Findings",
        "",
        "## Collective evidence aggregation",
        "",
        f"- Input observations: {len(annotated)}",
        f"- Model-session-skill units: {len(units)}",
        f"- Unique lesson signatures: {len(collective)}",
        f"- Units with mixed correctness: "
        f"{summary['mixed_outcome_units']}",
        f"- Lessons supported by both models: "
        f"{summary['cross_model_supported_lessons']}",
        "",
        "## Interpretation",
        "",
        "Multiple attempts on the same lesson in one",
        "model session are treated as correlated.",
        "Each model's contribution remains attributable.",
        "Conflicting correct/incorrect observations",
        "remain visible instead of being averaged away.",
        "",
        "## Limitations",
        "",
        "- One recorded live session per model.",
        "- Practice and assessment share templates.",
        "- Repeated reserved assessment items.",
        "- No independent cross-session replication.",
        "- No calibrated confidence or causal finding.",
        "- No model response authenticity beyond",
        "  the original stored-controller consistency.",
        "- No collective behavior experiment yet.",
        "",
        "## Chapter 7.3",
        "",
        "Compare isolated-agent and collective",
        "evidence feedback under identical trusted",
        "task authorization and bounded execution.",
    ]

    findings = findings_dir / "M7_C2_FINDINGS.md"
    findings.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\nFINAL MODULE 7 C2 SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nUnits:", units_path)
    print("Collective memory:", collective_path)
    print("Manifest:", manifest_path)
    print("Findings:", findings)

    return {
        "summary": summary,
        "units": units,
        "collective": collective,
    }


if __name__ == "__main__":
    run_benchmark()
