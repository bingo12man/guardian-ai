# Guardian AI — Module 0.2 (Reflexion)

This chapter is a *simplified Reflexion-style* learning lab, not an official reproduction.

## Requirements

- Chapter 0.1 `module_0_research/chapter_01/react_agent.py`
- Python 3.10+
- `pip install requests pytest`
- Ollama running with `llama3.2:3b` (or set `OLLAMA_MODEL`)

## Commands

From `module_0_research/chapter_02`:

```bash
python -m pytest -v
python reflexion_lab.py
```

It writes `results/e2_results.json` and, if the initial agent failed, `results/lessons.json`.

## Experimental arms

- Baseline: first trial with no lesson
- Corrective feedback: new trial with verbal lesson created after baseline failure
- Persistent memory: lesson saved as JSON, loaded from disk and used in a new trial

The mock tests deliberately create a failure so tests are deterministic. Live Ollama is **not forced to fail**. If the baseline passes, no reflection is generated and the two retries are `None`.

## Critical interpretation

This compares successive attempts, not randomized independent treatment groups. It demonstrates memory persistence, not neural weight updates, causality, or transfer to unseen agents. For a research-quality experiment, run many equivalent tasks under randomized controlled conditions, with and without memory, and report failure and task-success rates.

The evaluator is intentionally simple: for these synthetic tasks it checks that the correct source was read and that the answer contains the expected percentage. It cannot detect all semantic inaccuracies. No external targets are contacted except local Ollama.
