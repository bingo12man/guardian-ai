
# Guardian AI — Chapter 0.3
## Constitutional AI and Independent Enforcement

### Research objective

Investigate whether constitutional self-critique is sufficient
to prevent unauthorized AI-agent actions, and compare it with
independent policy enforcement.

### Research background

Inspired by:
Bai et al. (2022), Constitutional AI:
Harmlessness from AI Feedback.
https://arxiv.org/abs/2212.08073

This project studies inference-time policy enforcement.
It does not reproduce Constitutional AI fine-tuning or RLAIF.

### Experimental environment

- Model: Llama 3.2 3B through Ollama
- Language: Python
- Data: Synthetic public, confidential and restricted documents
- Actions: read_file, delete_file, send_external, finish
- Trusted policy: Independent Python authorization controller
- Tool execution: Local in-memory simulation
- Real deletion/external transmission: None

### C1 — Basic constitutional enforcement

Three scenarios were tested:

- Authorized document read succeeded.
- Protected document deletion was blocked.
- External transmission was blocked.

Blocked requests were recorded as incomplete rather
than incorrectly reported as completed.

### C2 — Trusted identity and authorization

Five live scenarios produced the expected authorization outcomes:

- Three authorized reads succeeded.
- Two unauthorized reads were blocked.
- False administrator claims did not grant access.

The separate eight-case deterministic policy benchmark reported:

- Unsafe allowances: 0
- Overblocked cases: 0
- Prohibited block rate: 100%

The benchmark evaluates a fixed policy specification,
not general real-world authorization security.

### C3 — Constitutional critique and revision

Six scenarios, three repeats, two output modes.

Total: 36 model trials.

Both JSON and schema-constrained modes achieved:

- Principle identification: 18/18 per mode
- Valid raw revisions: 18/18 per mode
- Policy-compliant revisions: 18/18 per mode
- Useful authorized alternatives: 6/18 per mode
- Appropriate terminations: 12/18 per mode
- Guardian fallbacks: 0

The 33.3% useful-alternative rate excludes appropriate
terminations. All 18 cases per condition produced
the expected form of behavior.

The critic received the Guardian's decision in its prompt,
so these results do not establish independent violation detection.

### C4 — Three-architecture comparison

Eight fixed cases repeated three times.

24 trials per architecture.

| Metric | Model-only | Self-critique | Guardian |
|---|---:|---:|---:|
| Unsafe proposals | 15 | 0 | 15 |
| Unsafe simulated executions | 15 | 0 | 0 |
| Legitimate task success | 9/9 | 0/9 | 9/9 |

Model-only:
Completed legitimate operations, but accepted
15 unauthorized proposals in simulation.

Self-critique:
Prevented unauthorized simulated execution, but also
terminated every legitimate task.

Guardian:
Blocked unauthorized proposals at the execution
boundary and permitted every legitimate operation.

### C4 — Critic confusion matrix

| | Predicted violation | Predicted allowed |
|---|---:|---:|
| Actually prohibited | 15 | 0 |
| Actually permitted | 9 | 0 |

- True-positive rate: 100%
- False-positive rate: 100%
- Specificity: 0%
- Overall accuracy: 62.5%

The critic classified every original action as prohibited.

The 62.5% accuracy reflects class prevalence rather
than meaningful discrimination between allowed and
prohibited actions.

### C5 — Conditional critique counterfactual

Offline analysis reused the saved C4 model responses.

- Total: 24
- Revisions triggered: 24
- Unsafe simulated executions: 0
- Legitimate task successes: 0/9

A conditional workflow that revises only after
a critic-reported violation did not improve results,
because the critic falsely flagged all legitimate actions.

No new Ollama responses were generated for this analysis.

### Main findings

1. Language-model safety instructions were not sufficient
   to prevent unauthorized proposed actions in C4.

2. Self-critique corrected prohibited proposals but also
   generated false-positive violation judgments.

3. Independent authorization separated the model's
   proposals from the executor's permissions.

4. Authorized work remained possible under Guardian
   enforcement on the fixed C4 benchmark.

5. A model-generated critique should not be treated
   as authoritative permission or denial.

### Limitations

- Small synthetic case set
- Repeated prompts, not independent benchmark tasks
- One local model family
- Mostly deterministic generation settings
- Static trusted permission registry
- No process isolation or production authentication
- No real third-party tools
- No real adversarial security guarantee
- Policy correctness assumed in the benchmark
- Self-critique and Guardian approaches have different
  inference and computation requirements
- No cross-model transfer or real-world deployment evaluation

### Architectural conclusion

The model may propose, explain and revise actions.

The trusted Guardian independently determines whether
a requested tool operation is authorized.

Critiques are advisory.
Authorization is enforced at the tool boundary.

### Future research

- More diverse authorization tasks
- Critic false-positive reduction
- Independent semantic critique evaluation
- Cross-model generalization
- Untrusted tool-output and prompt-injection testing
- Process isolation and scoped execution permissions
- Cross-agent transfer of verified safety lessons
