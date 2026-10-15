# Guardian AI — Experiment E4: Harder tasks & frozen-memory comparison

Standalone experiment: do **not** change the existing E2/E3 scripts. Uses mock documents in Python, not real file access. Ollama must be running with `llama3.2:3b` installed.

```bash
cd ~/Downloads/guardian-ai/module_0_research/chapter_02
python -m pytest -v
OLLAMA_MODEL=llama3.2:3b python e4_generalization.py --repeats 3
```

E4 trains on two synthetic tasks and evaluates four different task variants. Training creates lessons **only after failure**, freezes them, writes to JSON, and reloads them for treatment runs. The actor chooses zero to three documents and answers from their content. Evaluation checks selected evidence and task-specific answers. It reports both source selection and output correctness.

**Research caution:** If `Lessons: (none)`, with-memory is an empty treatment and no learning comparison has happened. Training versus evaluation questions may share an underlying skill and source documents. No statistical significance is claimed from a few repeated tasks. The rubrics are simplified string/regex checks and not suitable for security enforcement. The controller has exclusive authority over available document content.

Results: `results/e4_lessons.json` and `results/e4_results.json`.
