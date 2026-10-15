
# Guardian AI — Chapter 0.2: Reflexion Findings

## Research objective

Investigate whether verbal corrective feedback and
external memory improve autonomous agent performance.

## Setup

- Model: Llama 3.2 3B via Ollama
- Environment: Synthetic document-reading tasks
- Approach: ReAct-style execution with evaluator feedback
- Memory: External JSON lessons; no model-weight updates

## Experimental findings

| Experiment | Finding |
|---|---|
| E2 | Baseline succeeded after prompt improvements |
| E3 | 100% success in both conditions; ceiling effect |
| E4 | 75% in both conditions; no corrective lesson |
| E5 | 75% in both conditions after correcting evaluator |
| E6 | No memory 50%; generic 66.7%; targeted 66.7% |

## E6 details

- Six synthetic evaluation tasks
- Three repetitions per condition
- Eighteen attempts per condition
- Fixed, reviewed feedback lessons
- Corrected evaluator used for final scoring
- No observed regression on previously passing tasks
- Improvements concentrated in one arithmetic task
- No measured advantage for targeted over generic feedback

## Main observations

1. Feedback is not guaranteed to improve agent behavior.
2. Incorrect evaluators can create false failures.
3. Multi-document retrieval remains a weakness.
4. External textual memory is not equivalent to model training.
5. Safety lessons require independent validation.
6. Generalization remains unproven on larger benchmarks.

## Limitations

- Small synthetic dataset
- Simplified rule-based evaluation
- Repeated trials are not necessarily independent
- Evaluation tasks reuse related question structures
- No statistical claim of causal improvement
- No cross-model transfer evaluated
- No real-world threat prevention demonstrated

## Conclusion

Generic and failure-specific guidance were associated
with a 16.7-percentage-point improvement in observed
E6 task success after evaluator correction.

The experiment did not demonstrate that targeted
feedback outperformed generic feedback, nor did it
establish persistent changes to the underlying model.

Further research should prioritize evaluator reliability,
multi-document task completion, and transfer of verified
lessons across agents.
