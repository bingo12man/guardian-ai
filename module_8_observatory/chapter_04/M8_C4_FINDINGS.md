# Guardian AI — Chapter 8.4 Findings

## Synthetic robustness evaluation

- Decoys: 24
- Attack variants: 16
- Hard benign negatives: 8
- Multiclass accuracy: 0.2083
- Macro F1: 0.1164
- Binary precision: 0.2000
- Binary recall: 0.0625
- False-positive alerts: 4
- Missed attacks: 15

## Interpretation

The detector was frozen before evaluation.
Hard negatives include benign quotations,
security explanations and encoded examples.
Any false detections remain visible in the
candidate-review queue.

## Limitations

- Synthetic hand-authored examples only
- No independent real-world collection
- One fixed robustness suite
- Patterns may still overlap training design
- Labels reflect author-defined intent
- No LLM or downstream behavior evaluated
- No autonomous lesson publication

## Chapter 8.5

Consolidate the evidence and prepare
the Module 9 durability contract.
