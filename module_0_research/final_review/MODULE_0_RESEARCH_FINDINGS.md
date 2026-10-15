# Guardian AI — Module 0 Final Research Report

## 1. Research problem

Tool-using LLM agents may mistake untrusted content for instructions and propose actions outside the original authorized task.

Guardian AI investigates whether independent enforcement combined with structured safety learning can reduce repeat violations while preserving legitimate task completion.

## 2. Relevant prior work

| Work | Relevance | Reference |
|---|---|---|
| AgentDojo | Dynamic agent security evaluation | https://arxiv.org/abs/2406.13352 |
| Agent Security Bench | Security and utility benchmarking | https://proceedings.iclr.cc/paper_files/paper/2025/hash/5750f91d8fb9d5c02bd8ad2c3b44456b-Abstract-Conference.html |
| CaMeL | Capability and control-flow enforcement | https://arxiv.org/abs/2503.18813 |
| PromptArmor | Prompt-injection sanitization | https://arxiv.org/abs/2507.15219 |
| AgentArmor | Runtime security analysis | https://arxiv.org/abs/2508.01249 |

## 3. Evidence from previous experiments

| Experiment | Trials | Completions | Unauthorized executions |
|---|---:|---:|---:|
| R1 | 18 | 18 | 0 |
| R2 | 18 | 18 | 0 |
| R3 | 18 | 15 | 0 |

### Recovery ablation (R4)

- Raw retry completions: 3/15
- Sanitized retry completions: 12/15
- Unauthorized synthetic executions: 0

### Inspect AI evaluation (B4)

- Evaluated samples: 18
- Invalid outputs: 6
- Unsafe proposals: 2
- Guardian unauthorized synthetic executions: 0

## 4. Candidate research gap

Independent authorization and input sanitization have significant prior art. A promising question is whether lessons extracted from blocked actions can be retained, transferred to different agents, and used safely without weakening enforcement.

This gap is a research hypothesis, NOT a verified first-of-its-kind claim.

## 5. Main hypothesis

**H1:** Independent Guardian enforcement combined with structured, evidence-based safety lessons reduces repeat unsafe proposals on held-out scenarios relative to enforcement-only and sanitized-retry baselines, without increasing unauthorized execution or substantially reducing legitimate task completion.

**H0:** Learned safety lessons offer no improvement over these baselines.

## 6. Required future experiments

1. Comparable Llama and Qwen agent implementations.
2. Diverse synthetic tasks with known policy labels.
3. Independent enforcement and audit records.
4. Lesson extraction with malicious text removed.
5. Transfer to held-out models and tasks.
6. Persistence tests over multiple episodes.
7. Adaptive attacks and negative-transfer checks.
8. Statistical uncertainty and paired comparisons.

## 7. Primary evaluation metrics

- Unauthorized execution rate
- Unsafe proposal rate
- Legitimate task completion rate
- Repeated violation rate
- Cross-agent transfer improvement
- False rejection rate
- Recovery cost in calls and latency

## 8. Boundaries and limitations

- Existing experiments use synthetic document effects.
- Results are based on a small authored scenario set.
- Live comparisons are not always paired.
- Zero observed unauthorized executions is not a production security guarantee.
- Live R3/R4 did not exercise fallback or approval.
- Cross-model learning has not yet been demonstrated.
- Originality requires broader literature review.

## 9. Transition to Module 1

Build a standardized multi-model agent runner with comparable prompts, tool schemas, requester policies, deterministic scenarios, and model-independent evaluation records.
