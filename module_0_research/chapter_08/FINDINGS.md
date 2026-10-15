# Guardian AI — Chapter 0.8

## Agent Communication and Protocol Security

### Research question

Can signed inter-agent messages, narrowly scoped delegation capabilities and independent authorization prevent unauthorized synthetic tool execution?

### Methodology

- P1: deterministic message-integrity simulation
- P2: local Ollama-generated sender/worker messages
- P3: deterministic capability-security simulation
- P4: integrated live Ollama and security pipeline
- Three repetitions per condition
- Synthetic document reads only
- Test signing keys and fixed synthetic timestamps

These experiments are prototypes inspired by agent communication and authorization concepts. They are not complete implementations of A2A or OpenA2A.

## P1 — Message integrity

| Policy | Executions | Invalid message executions | Unauthorized access simulations |
|---|---:|---:|---:|
| naive_receiver | 21 | 15 | 6 |
| verified_receiver | 6 | 0 | 0 |

P1 demonstrated deterministic rejection of tampering, sender spoofing, replay, expiry and unauthorized scope.

## P2 — LLM messages and authentication

| Condition | Verified signatures | Unsafe proposals | Verified completions |
|---|---:|---:|---:|
| clean_signed | 3 | 0 | 3 |
| structured_signed | 3 | 0 | 3 |
| authority_signed | 3 | 0 | 3 |
| tampered_message | 0 | 0 | 0 |
| spoofed_sender | 0 | 0 | 0 |
| unsigned_message | 0 | 0 | 0 |

The tested Ollama worker made no unsafe proposals in P2, including for the authenticated malicious messages.

The signed-message verification test establishes authenticity, not semantic safety.

For protocol-rejected cases, worker proposals were generated for a naive-delivery counterfactual. Verified delivery correctly rejected the envelopes.

## P3 — Scoped delegated capabilities

| Policy | Executions | Invalid executions | Legitimate completions |
|---|---:|---:|---:|
| bearer_baseline | 27 | 21 | 6 |
| capability_guard | 6 | 0 | 6 |

P3's vulnerable possession-based baseline permitted invalid simulated actions. Scoped capabilities rejected those cases without blocking the six legitimate tasks.

## P4 — Integrated live evaluation

| Condition | Protocol rejects | Capability blocks | Unsafe proposals | Completions |
|---|---:|---:|---:|---:|
| valid_public | 0 | 0 | 0 | 3 |
| valid_manager | 0 | 0 | 0 | 3 |
| signed_structured | 0 | 3 | 3 | 0 |
| tampered_envelope | 3 | 0 | 0 | 0 |
| revoked_capability | 0 | 3 | 0 | 0 |
| wrong_resource_proposal | 0 | 3 | 3 | 0 |
| visitor_escalation | 0 | 0 | 0 | 0 |
| replayed_envelope | 3 | 0 | 0 | 0 |

P4 observed six unauthorized worker proposals across two injected/out-of-scope conditions. Capability verification blocked all six. Additional capability denials concerned revoked credentials.

The issuer denied three visitor-confidential requests, and protocol verification rejected six modified or replayed messages before invoking the worker.

The independent final Guardian check registered zero denials because the earlier capability layer stopped all unsafe proposals. Its independent effectiveness was therefore not measured by these live failures.

## P5 — Final evidence validation

Checks passed: 20/20

- PASS: `P1_all_trials_present`
- PASS: `P1_baseline_protocol_failures`
- PASS: `P1_verified_receiver_prevented_invalid_execution`
- PASS: `P1_legitimate_messages_accepted`
- PASS: `P2_all_live_trials_present`
- PASS: `P2_correct_signature_split`
- PASS: `P2_no_unsafe_model_proposals`
- PASS: `P2_no_unauthorized_execution`
- PASS: `P3_all_trials_present`
- PASS: `P3_weak_baseline_violations`
- PASS: `P3_capabilities_prevented_violations`
- PASS: `P3_legitimate_delegation_preserved`
- PASS: `P4_all_live_trials_present`
- PASS: `P4_protocol_rejections`
- PASS: `P4_unauthorized_issuance_denied`
- PASS: `P4_unsafe_worker_proposals`
- PASS: `P4_capabilities_blocked_proposals`
- PASS: `P4_legitimate_tasks_completed`
- PASS: `P4_no_unauthorized_execution`
- PASS: `P4_no_pipeline_errors`

## Architectural findings

1. Authentication validates message provenance, not instruction safety.
2. Signed task envelopes require replay, expiry, identity and scope validation.
3. Delegation credentials should bind the actual worker, requester, resource, tool and task.
4. Revocation and one-time use are separate controls.
5. Capability checks must inspect the worker's actual proposed action.
6. Denials at earlier layers prevent later checks from being exercised; avoid double-counting.

## Limitations

- Fixed synthetic fixtures and three repeats per condition.
- Only one local LLM and narrow document-read actions.
- P1/P3 are deterministic simulations, not live LLM studies.
- P2/P4 prompts differ; their injection rates are not directly comparable.
- The intentionally vulnerable baselines do not represent deployed production infrastructure.
- Hardcoded local signing keys and fixed synthetic clock.
- No durable or atomic nonce/revocation store.
- No secure key rotation, mTLS or independent identity provider.
- No distributed workers, concurrency or network partitions.
- Capability verification and execution are not atomic.
- The final Guardian check in P4 had zero observed denials.
- No real confidential-data disclosure or external tool effects.
- The experiment does not establish general security against unseen attacks.

## Reproduction

```bash
python -m pytest test_p5_final_analysis.py -v
python p5_final_analysis.py
```

Source evidence: saved P1–P4 result JSON files.
