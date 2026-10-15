# Guardian AI — Module 8.2 Findings

## Offline threat classification

- Synthetic examples: 12
- Multiclass accuracy: 1.0000
- Macro F1: 1.0000
- Binary attack F1: 1.0000

## Method

Controller-authored regular-expression rules
operate only on verified offline fixture text.
Ground-truth labels are read after prediction.

## Scientific limitations

- Only 12 synthetic examples.
- Detector rules are tailored to known patterns.
- No independent unseen-template test set.
- No real-world false-positive estimate.
- Encoded strings are not decoded or executed.
- A benign classification is not proof of safety.
- No LLM behavioral improvement has been tested.

## Chapter 8.3 handoff

Use rule IDs, evidence offsets and separately
validated labels to propose controller-reviewed
lessons without treating content as instructions.
