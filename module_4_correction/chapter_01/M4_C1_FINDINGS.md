# Guardian AI — Module 4, Chapter 4.1

## Evidence-grounded lesson extraction

Source: Module 3 C5 full live benchmark.

This chapter creates rule-derived candidate lessons from training-split denials only.

## Source live-model results

| Model | Completions | Blocks | Wrong-target proposals | Invalid outputs |
|---|---:|---:|---:|---:|
| llama3.2:3b | 6 | 207 | 26 | 38 |
| qwen2.5:3b | 25 | 103 | 21 | 6 |

## Lesson extraction

- Training denial observations: 165
- Unique candidate lessons: 5
- Validation/test episodes excluded from lesson construction: 60

| Failure category | Train observations |
|---|---:|
| schema_error | 18 |
| target_mismatch | 26 |
| workflow_scope_error | 121 |

## Safety properties

- Lessons use allowlisted controller rules.
- No raw model outputs or attack text are copied into stored lessons.
- No lesson expands task authorization.
- All supporting scenario IDs are train-only.

## Scientific limitations

- Candidate lessons have not been shown to improve future agent behavior.
- Failure categories come from the existing Guardian decision labels.
- Training and held-out splits share attack templates and some text patterns.
- Trace validation relies on the saved audit_verified flag; it does not re-verify the complete original gateway audit.
- This is rule-based extraction, not LLM-generated lesson synthesis.

## Next experiment

Chapter 4.2 will independently validate lesson usefulness, specificity and safety before any lesson is supplied back to an agent.
