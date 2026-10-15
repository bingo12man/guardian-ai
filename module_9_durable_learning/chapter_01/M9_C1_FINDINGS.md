# Guardian AI — Module 9.1 Findings

## Immutable baseline registration

- Frozen detector baseline experiments: 2
- Original fixture dataset and robustness
  dataset are registered separately.
- Replay registrations are rejected.
- Conflicting results cannot replace records.
- Reopening verifies the persisted audit chain.

## Security and durability limitations

- Single-writer local-file research prototype
- No external audit anchoring
- No transactionally atomic multi-file commit
- An operator controlling the filesystem
  can rewrite the entire local history
- Source evidence remains synthetic

## Module 9.2

Build idempotent durable lesson memory with
explicit lifecycle states and rollback.
