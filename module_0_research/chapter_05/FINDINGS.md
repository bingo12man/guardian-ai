# Guardian AI — Chapter 0.5

## Independent Policy Enforcement and Trace Learning

### Research objective

Evaluate whether an independently running policy engine
can govern synthetic AI-agent tool operations, support
policy changes and auditing, and provide a reliable
reference for candidate policy learning.

### Research foundations

- AgentGuardian: https://arxiv.org/abs/2601.10440
- ToolGuardian: https://arxiv.org/abs/2607.21835
- Open Policy Agent: https://www.openpolicyagent.org/docs/

This chapter is inspired by the research literature.
It does not reproduce AgentGuardian or ToolGuardian.

### Experimental environment

- Local Ollama model: Llama 3.2 3B
- Python execution gateway
- Open Policy Agent using Rego
- Synthetic documents and identities
- In-memory tool execution
- No real publication or external data transfer

### G1 — Initial OPA integration

A Python gateway was integrated with a separate OPA service.

The G1 design retained Python-side authorization prechecks,
so it did not move the full authorization decision into OPA.

G1 results are not included in this consolidated report
unless separately verified.

### G2 — Ollama-to-OPA gateway

Total live proposals: 30

- Strictly valid proposals: 15
- Schema rejections: 15
- Python precheck denials: 9
- OPA denials: 0
- Executed actions: 6
- Confidential summary publications: 0
- Mixed restricted summary publications: 0
- Observed policy versions: g1-v1

The G2 experiment demonstrated integration and fail-closed
behavior in the observed synthetic cases.

It did not exercise independent OPA denials:
Python prechecks rejected the prohibited model proposals.

### G3 — OPA-owned authorization decisions

| Metric | G3-v1 | G3-v2 |
|---|---:|---:|
| Cases | 10 | 10 |
| OPA decisions | 10 | 10 |
| OPA denials | 5 | 6 |
| False allowances | 0 | 0 |
| False blocks | 0 | 0 |
| Correct decisions | 10 | 10 |

G3 moved access-control decisions into OPA while retaining
trusted-attribute construction and schema validation in Python.

#### Policy-version differences

- `public_publish`: v1=True, v2=False

The version change was evaluated against the fixed benchmark.
It does not establish authenticated policy deployment or
tamper-resistant version identification.

### G4 — Candidate policy learning

Staging examples: 5
Held-out examples: 14

| Metric | Coarse | Contextual |
|---|---:|---:|
| False allowances | 3 | 0 |
| False blocks | 1 | 1 |
| Accuracy | 71.4% | 92.9% |
| Legitimate coverage | 83.3% | 83.3% |

OPA reference:
- Cases checked: 14
- Mismatches: 0
- Agreement: 100.0%

The coarse learner ignored approval and workflow context,
producing unsafe generalizations.

The contextual learner prevented those specific false
allowances but still overblocked an unseen legitimate action.

Neither candidate passed the promotion screen.
Neither was automatically deployed.

### G5 — Consolidated validation

Checks passed: 9/9

- PASS: g2_all_trials_accounted_for
- PASS: g2_no_private_publication
- PASS: g2_no_mixed_publication
- PASS: g3_v1_zero_false_allows
- PASS: g3_v2_zero_false_allows
- PASS: g3_v1_zero_false_blocks
- PASS: g3_v2_zero_false_blocks
- PASS: g4_opa_reference_agrees
- PASS: g4_no_automatic_deployment

### Architectural conclusion

The agent proposes actions.

The trusted gateway constructs authorization context from
protected state and rejects malformed tool calls.

OPA evaluates authorization policy.

The executor performs only explicitly authorized operations.

Audit records document the decision source and outcome.

Learned candidate policies remain separate from active
enforcement until independently reviewed.

### Limitations

1. Small synthetic benchmark.
2. Repeated cases are not independent tasks.
3. One local model family.
4. Simplified in-memory executor.
5. No production authentication or process isolation.
6. Trusted document classifications are assumed correct.
7. No concurrent execution or race-condition testing.
8. Policy labels rely on curated reference scenarios.
9. Learned rules use manually selected features.
10. The contextual learner is not equivalent to AgentGuardian.
11. Policy version strings do not authenticate deployed code.
12. JSONL audit files are not tamper-proof.
13. No real exfiltration or adversarial deployment tests.

### Future experiments

- Per-tool structured model schemas
- Real policy-service identity authentication
- Enforced process-level tool isolation
- Trusted state and provenance protection
- Versioned policy releases with approval workflows
- Tamper-evident audit infrastructure
- Cross-model generalization
- Learning candidate policies from richer agent traces
- Multi-agent policy conflict resolution

### Reproducibility

From `module_0_research/chapter_05`:

```bash
python -m pytest -v

python g5_final_analysis.py
```

To regenerate G4's independent OPA reference verification:

```bash
python g4_policy_learning.py --check-opa
```

G3 policies require separate OPA server runs for v1 and v2.

All findings in this report are derived from saved experimental
results in `results/`.
