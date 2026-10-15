# Guardian AI — Module 2 Dataset Card

## Dataset description

Synthetic document-access tasks with trusted permission labels and lower-trust search, document and memory content.

## Composition

- Scenarios: 60
- Clean scenarios: 12
- Attack scenarios: 48
- Train / validation / test: 30 / 10 / 20
- Attack techniques: target redirection, authority spoofing, encoded instruction, memory poisoning

## Ground truth

Permission labels are generated from the synthetic policy defined in Chapter 2.1.

## Final scripted evaluation

| Policy | Episodes | Completed | Blocked actions |
|---|---:|---:|---:|
| trusted_only | 60 | 45 | 15 |
| attack_then_recover | 60 | 45 | 51 |

## Integrity and leakage limitations

- Base scenario families are isolated across train, validation and test.
- Chapter 2.4 detected 14 cross-split exact-text groups and 6 normalized-text groups.
- The same attack templates are reused across splits.
- Consequently the test split does NOT establish novel attack-template generalization.
- Labels and checks partly share the same underlying Python policy.

## Intended use

- Guardian policy regression testing
- Synthetic action authorization testing
- Evaluation interface development
- Testing denial and recovery behavior

## Not validated for

- Production security certification
- Novel-attack generalization claims
- Real-world attack-success-rate estimates
- Real confidential data access
- Measuring durable safety learning

## Module 3 handoff

Replace scripted proposals with real agent proposals, while preserving trusted task ground truth, the authorization interface and execution-event accounting.
