
"""
Guardian AI — Module 9, Chapter 9.1

Durable, append-only experiment registry.

- Revalidates the Module 8 scientific handoff
- Registers an immutable frozen detector baseline
- Uses deterministic experiment IDs
- Detects replay, collision and altered records
- Persists using atomic create-only files
- Verifies hash-linked audit events after restart

Limitations:
- Local file hashes are not independently anchored
- This prototype assumes a single writer
- Privileged filesystem operators can alter storage
- No authorization or lesson-publication capability
"""

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
M8_C5 = ROOT / "module_8_observatory/chapter_05"

sys.path.insert(0, str(M8_C5))

import m8_5_final_benchmark as m8

FINAL_PATH = (
    M8_C5 / "results/m8_5_final_results.json"
)
CONTRACT_PATH = (
    M8_C5 / "MODULE_9_INTERFACE.json"
)
M8_MANIFEST_PATH = (
    M8_C5 / "results/manifest.json"
)

GENESIS = hashlib.sha256(
    b"GUARDIAN_M9_C1_AUDIT_V1"
).hexdigest()

RECORD_FIELDS = {
    "schema_version",
    "experiment_id",
    "kind",
    "detector_version",
    "detector_sha256",
    "dataset_sha256",
    "configuration",
    "result",
    "status",
    "source_module",
    "source_evidence_sha256",
}

ALLOWED_STATUS = {"FROZEN_BASELINE", "CANDIDATE_PENDING"}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def digest(value):
    return hashlib.sha256(
        canonical(value).encode("utf-8")
    ).hexdigest()


def file_hash(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def write_new_json(path, value):
    """
    Atomic create-only file write.

    'x' guarantees that existing experiment
    records cannot be silently overwritten.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            value, handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.flush()
        os.fsync(handle.fileno())


def validate_module_8_handoff():
    """
    Independently reconstruct Module 8 evidence,
    then compare final artifacts and handoff.

    Prevents using a modified local JSON summary
    as the sole basis for experiment registration.
    """
    source = m8.validate_sources()
    expected_final = m8.make_final(source)
    expected_contract = m8.make_module_9_contract(
        expected_final
    )

    final = read_json(FINAL_PATH)
    contract = read_json(CONTRACT_PATH)
    manifest = read_json(M8_MANIFEST_PATH)

    require(
        final == expected_final,
        "Module 8 final evidence mismatch",
    )
    require(
        contract == expected_contract,
        "Module 9 interface mismatch",
    )
    require(
        manifest["final_sha256"] == file_hash(FINAL_PATH)
        and manifest["contract_sha256"]
        == file_hash(CONTRACT_PATH),
        "Module 8 final manifest mismatch",
    )
    require(
        contract["source_module_8_sha256"]
        == digest(final)
        and contract["consumer_module"] == 9
        and contract["producer_module"] == 8,
        "Invalid Module 9 handoff",
    )

    policy = contract["authorization_constraints"]
    require(
        all(value is False for value in policy.values()),
        "Module 9 authorization contract unsafe",
    )

    return {
        "final": final,
        "contract": contract,
        "final_sha256": digest(final),
        "rule_sha256":
            final["frozen_detector_rule_sha256"],
    }


def experiment_id(payload):
    identity = {
        "schema_version": payload["schema_version"],
        "kind": payload["kind"],
        "detector_version": payload["detector_version"],
        "detector_sha256": payload["detector_sha256"],
        "dataset_sha256": payload["dataset_sha256"],
        "configuration": payload["configuration"],
        "source_module": payload["source_module"],
        "source_evidence_sha256":
            payload["source_evidence_sha256"],
    }
    return "experiment_" + digest(identity)[:24]


def validate_record(record):
    require(
        isinstance(record, dict)
        and set(record) == RECORD_FIELDS,
        "Malformed experiment record",
    )
    require(
        record["schema_version"] == "1.0",
        "Unsupported experiment schema",
    )
    require(
        record["kind"] in (
            "frozen_detector_baseline",
            "candidate_detector",
        ),
        "Unknown experiment kind",
    )
    require(
        record["status"] in ALLOWED_STATUS,
        "Unknown experiment status",
    )
    require(
        record["experiment_id"] == experiment_id(record),
        "Experiment identity mismatch",
    )
    require(
        all(
            isinstance(record[field], str)
            and len(record[field]) == 64
            for field in (
                "detector_sha256",
                "dataset_sha256",
                "source_evidence_sha256",
            )
        ),
        "Invalid digest fields",
    )
    require(
        isinstance(record["configuration"], dict)
        and isinstance(record["result"], dict),
        "Malformed experiment payload",
    )

    if record["kind"] == "frozen_detector_baseline":
        require(
            record["status"] == "FROZEN_BASELINE"
            and record["source_module"] == 8,
            "Incorrect frozen baseline status",
        )
    else:
        require(
            record["status"] == "CANDIDATE_PENDING",
            "Candidate already activated",
        )

    return True


def make_baseline_records(handoff):
    final = handoff["final"]

    # Distinct immutable datasets, same frozen detector.
    datasets = (
        ("original_fixtures", final["original_dataset"]),
        ("robustness_decoys", final["robustness_dataset"]),
    )

    records = []

    for dataset_name, evaluation in datasets:
        dataset_identity = {
            "dataset": dataset_name,
            "module": 8,
            "version": "1.0",
            "source_result_sha256": handoff["final_sha256"],
        }

        base = {
            "schema_version": "1.0",
            "kind": "frozen_detector_baseline",
            "detector_version": "m8_rule_detector_v1",
            "detector_sha256": handoff["rule_sha256"],
            "dataset_sha256": digest(dataset_identity),
            "configuration": {
                "dataset": dataset_name,
                "evaluation_mode": "offline_reconstructed",
                "rules_frozen": True,
                "model_inference": False,
            },
            "result": copy.deepcopy(evaluation),
            "status": "FROZEN_BASELINE",
            "source_module": 8,
            "source_evidence_sha256":
                handoff["final_sha256"],
        }

        base["experiment_id"] = experiment_id(base)
        validate_record(base)
        records.append(base)

    return records


class ExperimentRegistry:
    """
    Local persistent, append-only research registry.

    Each record file is create-only. The audit
    journal is checked on every reopening.

    Designed for a SINGLE writer.
    """

    def __init__(self, storage_dir):
        self.root = Path(storage_dir)
        self.records_dir = self.root / "records"
        self.audit_path = self.root / "audit.json"

        self.records_dir.mkdir(
            parents=True, exist_ok=True
        )

        if self.audit_path.exists():
            self.audit = read_json(self.audit_path)
        else:
            self.audit = []

        self.verify()

    def register(self, record):
        validate_record(record)

        identifier = record["experiment_id"]
        path = self.records_dir / f"{identifier}.json"

        if path.exists():
            existing = read_json(path)
            if existing == record:
                return {
                    "accepted": False,
                    "reason": "DUPLICATE_EXPERIMENT",
                    "experiment_id": identifier,
                }
            return {
                "accepted": False,
                "reason": "EXPERIMENT_ID_CONFLICT",
                "experiment_id": identifier,
            }

        # Verify the complete current registry
        # before accepting another experiment.
        self.verify()

        previous = (
            self.audit[-1]["event_hash"]
            if self.audit else GENESIS
        )

        event = {
            "sequence": len(self.audit) + 1,
            "previous_hash": previous,
            "experiment_id": identifier,
            "record_sha256": digest(record),
            "event_type": "REGISTERED",
        }
        event["event_hash"] = digest(event)

        # Record cannot overwrite an existing file.
        write_new_json(path, record)

        updated = self.audit + [event]

        # Write audit using temporary replacement.
        # A crash between writes can leave an
        # incomplete journal; verify() fails closed.
        temporary = self.root / "audit.pending.json"
        if temporary.exists():
            raise ValueError(
                "Pending audit write requires recovery"
            )

        write_new_json(temporary, updated)
        os.replace(temporary, self.audit_path)
        self.audit = updated
        self.verify()

        return {
            "accepted": True,
            "reason": "REGISTERED",
            "experiment_id": identifier,
        }

    def verify(self):
        require(
            isinstance(self.audit, list),
            "Invalid audit journal",
        )

        previous = GENESIS
        identifiers = set()

        for position, event in enumerate(self.audit, 1):
            require(
                event["sequence"] == position
                and event["previous_hash"] == previous
                and event["event_type"] == "REGISTERED",
                "Audit chain mismatch",
            )

            unsigned = {
                k: v for k, v in event.items()
                if k != "event_hash"
            }
            require(
                event["event_hash"] == digest(unsigned),
                "Audit event checksum mismatch",
            )

            identifier = event["experiment_id"]
            require(
                identifier not in identifiers,
                "Duplicate experiment in audit",
            )
            identifiers.add(identifier)

            path = (
                self.records_dir / f"{identifier}.json"
            )
            require(path.is_file(), "Missing experiment record")

            record = read_json(path)
            validate_record(record)

            require(
                record["experiment_id"] == identifier
                and digest(record) == event["record_sha256"],
                "Experiment evidence altered",
            )

            previous = event["event_hash"]

        disk_ids = {
            path.stem
            for path in self.records_dir.glob("*.json")
        }
        require(
            disk_ids == identifiers,
            "Unregistered or missing record file",
        )

        return True

    def list_records(self):
        self.verify()
        return [
            read_json(
                self.records_dir
                / f"{event['experiment_id']}.json"
            )
            for event in self.audit
        ]

    def audit_head(self):
        return (
            self.audit[-1]["event_hash"]
            if self.audit else GENESIS
        )
