# Guardian AI — Module 5, Chapter 5.1

## Cross-Agent Lesson Provenance

This chapter builds model-specific evidence support for existing controller-authored safety corrections.

## Donor libraries

| Donor | Supported lessons | Failure observations |
|---|---:|---:|
| llama3.2:3b | 4 | 113 |
| qwen2.5:3b | 3 | 52 |

## Transfer dataset

- Total directional candidates: 14
- Cross-model candidates: 7
- Same-model control candidates: 7

## Security guarantees in this dataset

- Donor provenance is established using training failure observations.
- Lesson text is taken from trusted controller-owned templates.
- No held-out episodes contribute to lesson construction.
- No transfer changes task authorization.
- Raw attack text is not embedded in transfer records.

## Research limitations

- Donor support is not evidence of successful transfer.
- A controller-authored correction is not a novel rule invented by its donor.
- Several lessons may be supported by both models.
- Existing scenario splits reuse some attack templates and text patterns.
- The dataset is not independent of the prior lesson-selection process.

## Next chapter

Chapter 5.2 will implement a donor-aware retriever that prevents recipient evidence from entering donor-specific lesson selection.
