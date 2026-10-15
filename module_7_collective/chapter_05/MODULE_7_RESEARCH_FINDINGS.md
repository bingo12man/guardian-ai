# Guardian AI — Module 7 Final Research Findings

## Research question

Can multiple agents share verified safety evidence
and improve outcomes without weakening Guardian?

## Live task completions

| Model | Baseline | Isolated | Collective |
|---|---:|---:|---:|
| llama3.2:3b | 15/20 | 15/20 | 15/20 |
| qwen2.5:3b | 10/20 | 9/20 | 9/20 |

## Main conclusion

Verified collective evidence was available and
the synthetic Guardian boundary held, but
no collective task-completion improvement
was observed in this single live benchmark.

## Integrity tests

Five modified claims were rejected, as was
one replay; aggregate manipulation was detected.
Those controlled fixtures are not a production
adversarial security evaluation.

## Limitations

- One live evaluation repetition
- Twenty authorized scenarios per condition
- One source session per contributing model
- Stored source scores lack independent
  raw model-response authentication
- Controller-authored corrections
- Shared templates and overlap across splits
- Identical feedback may be relabeled collective
- Paired comparisons use separate model calls
- No causal collective-learning finding

## Module 8 handoff

Collect only operator-authorized public data
or owned synthetic decoys. Treat collected
content as inert evidence and require trusted
approval before creating any new lesson.
