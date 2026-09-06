"""Synthetic-only tests for the R2 evidence-trace diagnostic."""

from __future__ import annotations

import ast
import base64
import contextlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import sqlite3
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for location in (ROOT, SRC):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))


from experiments.runners import (  # noqa: E402
    high_level_multidomain_v1_r2_trace_diagnostic as diagnostic,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ref(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("ascii")).hexdigest()


def _grammar() -> diagnostic.LiteralGrammar:
    return diagnostic.LiteralGrammar(
        family="GLYPH_MACHINE",
        separator=",",
        terminator="STOP",
        allowed_tokens=("A_0000000000000001", "A_0000000000000002"),
        maximum_items=3,
    )


def _attempt(
    task_ordinal: int,
    arm: str,
    *,
    admitted: bool = True,
    success: bool = False,
    receipt_suffix: str = "",
) -> diagnostic.AttemptTrace:
    label = f"task-{task_ordinal}-{arm}-{receipt_suffix}"
    return diagnostic.AttemptTrace(
        task_id=_ref(f"task-{task_ordinal}"),
        arm=arm,
        phase="FINAL",
        attempt_receipt_ref=_ref(f"receipt-{label}"),
        stage_ref=_ref(f"stage-{label}"),
        judgment_ref=_ref(f"judgment-{label}"),
        parser_disposition="ADMITTED" if admitted else "MALFORMED",
        history_class=(
            "NO_PERSISTENT_HISTORY" if arm == "QWEN_ONLY" else "FROZEN_RECALL"
        ),
        recall_batch_ref=None if arm == "QWEN_ONLY" else _ref(f"recall-{task_ordinal}"),
        recalled_record_count=0 if arm == "QWEN_ONLY" else 2,
        candidate_classes=("DECLARED_GRAMMAR_CONFORMANT", "OUTSIDE_DECLARED_GRAMMAR")
        if admitted
        else (),
        selected_index=0 if admitted else None,
        selected_class="DECLARED_GRAMMAR_CONFORMANT" if admitted else "NOT_SELECTED",
        selection_ref=_ref(f"selection-{label}") if admitted else None,
        reservation_ref=(
            _ref(f"reservation-{label}")
            if admitted and arm not in {"QWEN_ONLY", "RETRIEVAL_ONLY"}
            else None
        ),
        response_class="CONFORMANT" if admitted else "NOT_EXECUTED",
        response_length=23 if admitted else 0,
        selected_response_equal=False,
        objective_success=success,
        objective_disposition="SUCCESS" if success else "UNSUCCESSFUL",
        execution_request_ref=_ref(f"request-{label}") if admitted else None,
        execution_receipt_ref=_ref(f"execution-{label}") if admitted else None,
        learner_parent_ref=None,
        learner_child_ref=None,
        learner_sequence=None,
        episode_ref=None,
    )


def _closed_payload() -> dict[str, object]:
    witness_rows = [
        {
            "gid": os.getegid(),
            "label": spec.label,
            "mode": spec.expected_mode,
            "nlink": 1,
            "sha256": "sha256:" + spec.sha256,
            "size_bytes": 1,
            "uid": os.geteuid(),
        }
        for spec in sorted(diagnostic.FROZEN_INPUTS, key=lambda item: item.label)
    ]
    matrix = []
    for task in range(3):
        attempts = [
            _attempt(task, arm, success=arm == "QWEN_ONLY").redacted()
            for arm in diagnostic.EXPECTED_ARMS
        ]
        qwen_attempt = attempts[diagnostic.EXPECTED_ARMS.index("QWEN_ONLY")]
        qwen_attempt["response_class"] = "OUTSIDE_DECLARED_GRAMMAR"
        partition = [list(diagnostic.EXPECTED_ARMS)]
        objective_partition = [
            [arm for arm in diagnostic.EXPECTED_ARMS if arm != "QWEN_ONLY"],
            ["QWEN_ONLY"],
        ]
        matrix.append(
            {
                "attempts": attempts,
                "equality_groups": {
                    "objective_score_equality": objective_partition,
                    "proposal_equality": partition,
                    "proposal_request_equality": partition,
                    "recall_preimage_equality": partition,
                    "response_equality": partition,
                    "selected_trace_equality": partition,
                },
                "first_divergences": [
                    {"arm": arm, "field": "NONE"}
                    for arm in diagnostic.EXPECTED_ARMS
                ],
                "replicate_ref": _ref("replicate"),
                "selected_execution_class_mismatch_count": 1,
                "task_ref": _ref(f"task-{task}"),
            }
        )
    matrix.sort(key=lambda row: str(row["task_ref"]))
    qwen_evidence_refs = {
        str(attempt[name])
        for row in matrix
        for attempt in row["attempts"]
        if attempt["arm"] == "QWEN_ONLY"
        for name in ("stage_ref", "judgment_ref")
    }
    record_ref = _ref("retained-record")
    scope_ref = _ref("scope")
    acquisition_ref = _ref("acquisition")
    source_ref = _ref("source")
    occurrences = [
        {
            "acquisition_ref": acquisition_ref,
            "ancestry_refs": sorted(
                {acquisition_ref, source_ref, _ref("ancestry")}
            ),
            "recall_index": 0,
            "record_bytes_ref": record_ref,
            "record_ref": record_ref,
            "source_contract": "ANG-CTR-COGNITIVE-EPISODE-001@0.1.0",
            "source_ref": source_ref,
            "target_full_replicate_ref": _ref("replicate"),
            "target_lineage_scope_ref": scope_ref,
            "target_task_id": _ref(f"task-{task}"),
        }
        for task in range(3)
    ]
    episode_refs = [_ref(f"episode-{index}") for index in range(12)]
    q5_evidence_refs = qwen_evidence_refs | {
        acquisition_ref,
        record_ref,
        source_ref,
        *episode_refs,
    }
    input_refs = {
        spec.label: "sha256:" + spec.sha256 for spec in diagnostic.FROZEN_INPUTS
    }
    payload: dict[str, object] = {
        "identity": diagnostic.DIAGNOSTIC_IDENTITY,
        "inputs": {
            "after": witness_rows,
            "before": witness_rows,
            "unchanged": True,
        },
        "matrix": matrix,
        "population": {
            "admitted_count": 433,
            "arms_per_task": 7,
            "attempt_count": 468,
            "factory_disposition_count": 468,
            "finalized_receipt_count": 468,
            "malformed_count": 35,
            "matrix_attempt_count": 21,
            "persistent_lineage_count": 4,
            "raw_record_ref_collision_count": 0,
            "recall_occurrence_count": 3,
            "removal_row_count": 60,
            "target_lineage_count": 1,
            "target_task_count": 3,
            "unique_recalled_record_count": 1,
        },
        "predecessor": {
            "arm_factory_audit_ref": input_refs["R2_ARM_FACTORY_AUDIT"],
            "classification": "NOT_SUPPORTED",
            "diagnostic_leaf_ref": input_refs["DIAGNOSTIC_LEAF"],
            "ledger_ref": input_refs["R2_LEDGER"],
            "manifest_ref": input_refs["R2_SOURCE_MANIFEST"],
            "predecessor_leaf_ref": input_refs["R2_PREDECESSOR_LEAF"],
            "result_ref": input_refs["R2_RESULT"],
            "result_report_ref": input_refs["R2_RESULT_REPORT"],
            "runner_ref": input_refs["R2_RUNNER"],
        },
        "provenance": {
            "adaptation_episode_joined": True,
            "all_rejoined": True,
            "frozen_batch_recomputed": False,
            "join_method": "PERSISTED_REF_HASH_CONTENT_TO_RETAINED_RECORD",
            "prospective_contract_recomputed": False,
            "prospective_contract_recomputation_disposition": "INDETERMINATE",
            "lineages": [
                {
                    "adaptation_attempt_count": 12,
                    "admitted_transition_count": 12,
                    "joined_episode_refs": episode_refs,
                    "lineage_scope_ref": scope_ref,
                    "malformed_transition_count": 0,
                    "replicate_ref": _ref("replicate"),
                }
            ],
            "occurrence_count": 3,
            "occurrences": occurrences,
            "raw_record_ref_collision_count": 0,
            "unique_record_count": 1,
        },
        "questions": [
            {
                "disposition": disposition,
                "evidence_refs": sorted(
                    {
                        _ref(f"question-{index}"),
                        *(q5_evidence_refs if index == 5 else ()),
                    }
                ),
                "question": f"Q{index}",
            }
            for index, disposition in enumerate(
                (
                    "ESTABLISHED",
                    "ESTABLISHED",
                    "RULED_OUT",
                    "CONSISTENT_WITH_EVIDENCE",
                    "ESTABLISHED",
                    "INDETERMINATE",
                ),
                start=1,
            )
        ],
        "resources": {
            "cognitive_record_count": 12,
            "input_bytes": 15,
            "no_cognee": True,
            "no_gpu": True,
            "no_model": True,
            "no_network": True,
            "no_subprocess": True,
            "output_size_bytes": 0,
            "peak_rss_bytes": 1,
            "process_count": 1,
            "wall_milliseconds": 1,
        },
        "schema": diagnostic.DIAGNOSTIC_SCHEMA,
    }
    for _ in range(8):
        size = len(diagnostic.canonical_json_bytes(payload))
        resources = payload["resources"]
        assert isinstance(resources, dict)
        if resources["output_size_bytes"] == size:
            break
        resources["output_size_bytes"] = size
    return payload


def _blob(value: object) -> tuple[str, bytes]:
    raw = diagnostic.canonical_json_bytes(value)
    return "sha256:" + hashlib.sha256(raw).hexdigest(), raw


def _create_synthetic_cognitive_db(
    path: Path,
    *,
    mismatched_batch_acquisition: bool = False,
) -> tuple[diagnostic.FrozenInputSpec, dict[str, str]]:
    episode = {
        "child_state_digest": _ref("child-state"),
        "commitment": _ref("commitment"),
        "contract": diagnostic._LEGACY_EPISODE_CONTRACT,
        "encoder_ref": _ref("encoder"),
        "feedback_source_ref": _ref("feedback-source"),
        "feedback_text": "SYNTHETIC_FEEDBACK",
        "model_ref": _ref("model"),
        "observations": ["SYNTHETIC_OBSERVATION"],
        "outcome": "OBSERVED",
        "parent_state_digest": _ref("parent-state"),
        "proposals": ["SYNTHETIC_PROPOSAL"],
        "recalled_refs": [],
        "request": "SYNTHETIC_REQUEST",
        "response": "SYNTHETIC_RESPONSE",
        "selected_index": 0,
        "supporting_evidence_refs": [_ref("support")],
        "task_id": _ref("adaptation-task"),
        "visibility": "LEARNER_VISIBLE",
    }
    episode_ref, episode_bytes = _blob(episode)
    reservation_ref, reservation_bytes = _blob(
        {"contract": "SYNTHETIC_RESERVATION", "request": "SYNTHETIC_REQUEST"}
    )
    batch_ref, batch_bytes = _blob(
        {"contract": "SYNTHETIC_BATCH", "proposals": ["SYNTHETIC_PROPOSAL"]}
    )
    resolution_ref, resolution_bytes = _blob(
        {"contract": "SYNTHETIC_RESOLUTION", "feedback_text": "SYNTHETIC_FEEDBACK"}
    )
    episode_v2_ref, episode_v2_bytes = _blob(
        {"contract": "SYNTHETIC_EPISODE_V2", "response": "SYNTHETIC_RESPONSE"}
    )

    def memory_record(
        ordinal: int,
        *,
        source_contract: str,
        source_ref: str,
        kind: str,
        status: str,
    ) -> dict[str, object]:
        return {
            "acquired_ordinal": ordinal,
            "competence_ref": _ref(f"competence-{ordinal}"),
            "confidence_ppm": 750_000,
            "content": f"SYNTHETIC_RETAINED_CONTENT_{ordinal}",
            "contract": diagnostic._MEMORY_RECORD_CONTRACT,
            "epistemic_status": status,
            "kind": kind,
            "producer_checkpoint_ref": _ref(f"checkpoint-{ordinal}"),
            "producer_id": "synthetic-test",
            "provenance_refs": [source_ref],
            "relations": [],
            "supersedes_refs": [],
            "tombstone_refs": [],
            "visibility": "LEARNER_VISIBLE",
            "world_valid_from": 0,
            "world_valid_until": None,
        }

    acquisition_rows: list[tuple[object, ...]] = []
    projection_rows: list[tuple[object, ...]] = []
    previous: str | None = None
    refs: dict[str, str] = {
        "episode_ref": episode_ref,
        "legacy_reservation_ref": _ref("legacy-reservation"),
        "reservation_ref": reservation_ref,
        "batch_ref": batch_ref,
        "resolution_ref": resolution_ref,
    }
    for ordinal, (label, source_contract, source_ref, kind, status) in enumerate(
        (
            (
                "batch",
                diagnostic._PROSPECTIVE_BATCH_CONTRACT,
                batch_ref,
                "COUNTERFACTUAL",
                "PROPOSED",
            ),
            (
                "resolution",
                diagnostic._PROSPECTIVE_RESOLUTION_CONTRACT,
                resolution_ref,
                "EPISODIC",
                "OBSERVED",
            ),
        )
    ):
        record = memory_record(
            ordinal,
            source_contract=source_contract,
            source_ref=source_ref,
            kind=kind,
            status=status,
        )
        record_ref, record_bytes = _blob(record)
        acquisition = {
            "contract": diagnostic._ACQUISITION_CONTRACT,
            "ordinal": ordinal,
            "predecessor_acquisition_ref": previous,
            "record": record,
            "source_contract": source_contract,
            "source_ref": source_ref,
        }
        acquisition_ref, acquisition_bytes = _blob(acquisition)
        projection = {
            "acquisition_ref": acquisition_ref,
            "contract": diagnostic._GRAPH_PROJECTION_CONTRACT,
            "record": record,
            "source_contract": source_contract,
            "source_ref": source_ref,
        }
        projection_ref, projection_bytes = _blob(projection)
        refs[f"{label}_acquisition_ref"] = acquisition_ref
        refs[f"{label}_record_ref"] = record_ref
        acquisition_rows.append(
            (
                ordinal,
                acquisition_ref,
                previous,
                source_contract,
                source_ref,
                record_ref,
                acquisition_ref,
                acquisition_bytes,
            )
        )
        projection_rows.append(
            (
                ordinal,
                acquisition_ref,
                record_ref,
                projection_ref,
                projection_ref,
                projection_bytes,
                0 if ordinal == 0 else 1,
            )
        )
        self_check = "sha256:" + hashlib.sha256(record_bytes).hexdigest()
        assert self_check == record_ref
        previous = acquisition_ref

    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA user_version = 3")
        connection.execute(
            "CREATE TABLE cognitive_acquisitions ("
            "ordinal INTEGER PRIMARY KEY, acquisition_ref TEXT UNIQUE NOT NULL, "
            "predecessor_acquisition_ref TEXT, source_contract TEXT NOT NULL, "
            "source_ref TEXT NOT NULL, record_ref TEXT UNIQUE NOT NULL, "
            "payload_sha256 TEXT NOT NULL, canonical_payload BLOB NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE acquisition_projection_outbox ("
            "ordinal INTEGER PRIMARY KEY, acquisition_ref TEXT NOT NULL, "
            "record_ref TEXT NOT NULL, projection_ref TEXT UNIQUE NOT NULL, "
            "payload_sha256 TEXT NOT NULL, canonical_payload BLOB NOT NULL, "
            "acknowledged INTEGER NOT NULL, "
            "FOREIGN KEY(ordinal) REFERENCES cognitive_acquisitions(ordinal))"
        )
        connection.execute(
            "CREATE TABLE acquisition_clock ("
            "singleton INTEGER PRIMARY KEY, next_ordinal INTEGER NOT NULL, "
            "acquisition_ref TEXT, record_ref TEXT)"
        )
        connection.execute(
            "CREATE TABLE episodes ("
            "episode_ref TEXT PRIMARY KEY, sequence INTEGER NOT NULL, "
            "parent_state_digest TEXT NOT NULL, child_state_digest TEXT NOT NULL, "
            "child_snapshot_sha256 TEXT NOT NULL, payload_sha256 TEXT NOT NULL, "
            "model_ref TEXT NOT NULL, encoder_ref TEXT NOT NULL, "
            "canonical_payload BLOB NOT NULL)"
        )
        prospective_columns = (
            "reservation_ref TEXT PRIMARY KEY",
            "legacy_reservation_ref TEXT NOT NULL",
            "batch_ref TEXT NOT NULL",
            "parent_lineage_ref TEXT NOT NULL",
            "selected_branch_ref TEXT NOT NULL",
            "status TEXT NOT NULL",
            "parent_sequence INTEGER NOT NULL",
            "parent_event_ref TEXT NOT NULL",
            "parent_state_digest TEXT NOT NULL",
            "parent_snapshot_sha256 TEXT NOT NULL",
            "model_ref TEXT NOT NULL",
            "encoder_ref TEXT NOT NULL",
            "reservation_payload_sha256 TEXT NOT NULL",
            "canonical_reservation BLOB NOT NULL",
            "canonical_batch BLOB NOT NULL",
            "pending_blob_sha256 TEXT",
            "pending_blob_size INTEGER",
            "pending_blob BLOB",
            "execution_request_ref TEXT",
            "canonical_execution_request BLOB",
            "execution_receipt_ref TEXT",
            "canonical_execution_receipt BLOB",
            "feedback_ref TEXT",
            "canonical_feedback BLOB",
            "resolution_ref TEXT",
            "resolution_disposition TEXT",
            "canonical_resolution BLOB",
            "legacy_episode_ref TEXT",
            "episode_v2_ref TEXT",
            "canonical_episode_v2 BLOB",
            "batch_acquisition_ref TEXT NOT NULL",
            "resolution_acquisition_ref TEXT",
        )
        connection.execute(
            "CREATE TABLE prospective_turns_v3 ("
            + ",".join(prospective_columns)
            + ")"
        )
        connection.executemany(
            "INSERT INTO cognitive_acquisitions VALUES (?,?,?,?,?,?,?,?)",
            acquisition_rows,
        )
        connection.executemany(
            "INSERT INTO acquisition_projection_outbox VALUES (?,?,?,?,?,?,?)",
            projection_rows,
        )
        connection.execute(
            "INSERT INTO acquisition_clock VALUES (1,?,?,?)",
            (2, refs["resolution_acquisition_ref"], refs["resolution_record_ref"]),
        )
        connection.execute(
            "INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?,?)",
            (
                episode_ref,
                1,
                _ref("parent-state"),
                _ref("child-state"),
                _ref("snapshot"),
                episode_ref,
                _ref("model"),
                _ref("encoder"),
                episode_bytes,
            ),
        )
        prospective_values = (
            reservation_ref,
            refs["legacy_reservation_ref"],
            batch_ref,
            _ref("parent-lineage"),
            _ref("selected-branch"),
            "RESOLVED",
            1,
            _ref("parent-event"),
            _ref("parent-state"),
            _ref("parent-snapshot"),
            _ref("model"),
            _ref("encoder"),
            reservation_ref,
            reservation_bytes,
            batch_bytes,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            resolution_ref,
            "OBSERVED",
            resolution_bytes,
            episode_ref,
            episode_v2_ref,
            episode_v2_bytes,
            (
                refs["resolution_acquisition_ref"]
                if mismatched_batch_acquisition
                else refs["batch_acquisition_ref"]
            ),
            refs["resolution_acquisition_ref"],
        )
        connection.execute(
            "INSERT INTO prospective_turns_v3 VALUES ("
            + ",".join("?" for _ in prospective_values)
            + ")",
            prospective_values,
        )
        connection.commit()
    finally:
        connection.close()
    path.chmod(0o600)
    spec = diagnostic.FrozenInputSpec(
        label="SYNTHETIC_COGNITIVE",
        path=path,
        sha256=_sha256(path),
        kind="SQLITE",
        expected_mode=0o600,
    )
    return spec, refs


class CanonicalAndRedactionTests(unittest.TestCase):
    def test_canonical_json_is_deterministic_and_round_trips(self) -> None:
        first = {"z": [3, 2, 1], "a": {"status": "ESTABLISHED", "count": 3}}
        second = {"a": {"count": 3, "status": "ESTABLISHED"}, "z": [3, 2, 1]}

        encoded = diagnostic.canonical_json_bytes(first)
        self.assertEqual(encoded, diagnostic.canonical_json_bytes(second))
        self.assertEqual(encoded, diagnostic.canonical_json_bytes(first))
        self.assertEqual(diagnostic.parse_canonical_json(encoded), second)
        self.assertEqual(encoded, json.dumps(second, sort_keys=True, separators=(",", ":")).encode("utf-8"))

    def test_canonical_json_is_exact_utf8_nfc_without_ascii_escaping_or_repair(self) -> None:
        nfc = "caf\N{LATIN SMALL LETTER E WITH ACUTE}"
        encoded = diagnostic.canonical_json_bytes({"label": nfc})
        self.assertEqual(encoded, b'{"label":"caf\xc3\xa9"}')
        self.assertNotIn(b"\\u00e9", encoded)
        self.assertEqual(diagnostic.parse_canonical_json(encoded), {"label": nfc})

        non_nfc = "cafe\N{COMBINING ACUTE ACCENT}"
        self.assertNotEqual(non_nfc, nfc)
        with self.assertRaises(diagnostic.DiagnosticInvariantError):
            diagnostic.canonical_json_bytes({"label": non_nfc})
        with self.assertRaises(diagnostic.DiagnosticInvariantError):
            diagnostic.canonical_json_bytes({non_nfc: "value"})
        with self.assertRaises(diagnostic.DiagnosticInvariantError):
            diagnostic.parse_canonical_json(
                ('{"label":"' + non_nfc + '"}').encode("utf-8")
            )

    def test_parse_rejects_noncanonical_or_duplicate_key_json(self) -> None:
        with self.assertRaises(diagnostic.DiagnosticInvariantError):
            diagnostic.parse_canonical_json(b'{"b": 1, "a": 2}')
        with self.assertRaises(diagnostic.DiagnosticInvariantError):
            diagnostic.parse_canonical_json(b'{"a":1,"a":2}')

    def test_redaction_accepts_refs_counts_and_dispositions(self) -> None:
        payload = _closed_payload()
        diagnostic.validate_redacted_output(payload, forbidden_values=())

    def test_redaction_rejects_unknown_causal_claim_and_bad_question_disposition(self) -> None:
        payload = _closed_payload()
        payload["causal_claim"] = "UNREGISTERED_CAUSAL_ASSERTION"
        with self.assertRaises(diagnostic.DiagnosticInvariantError):
            diagnostic.validate_redacted_output(payload)

        payload = _closed_payload()
        questions = payload["questions"]
        self.assertIsInstance(questions, list)
        questions[0]["disposition"] = "CAUSALLY_ESTABLISHED"
        with self.assertRaises(diagnostic.DiagnosticInvariantError):
            diagnostic.validate_redacted_output(payload)

        payload = _closed_payload()
        questions = payload["questions"]
        self.assertIsInstance(questions, list)
        questions[5]["disposition"] = "ESTABLISHED"
        with self.assertRaises(diagnostic.DiagnosticInvariantError):
            diagnostic.validate_redacted_output(payload)

        payload = _closed_payload()
        provenance = payload["provenance"]
        self.assertIsInstance(provenance, dict)
        provenance["prospective_contract_recomputed"] = True
        with self.assertRaises(diagnostic.DiagnosticInvariantError):
            diagnostic.validate_redacted_output(payload)

        payload = _closed_payload()
        matrix = payload["matrix"]
        self.assertIsInstance(matrix, list)
        qwen = next(
            attempt
            for attempt in matrix[0]["attempts"]
            if attempt["arm"] == "QWEN_ONLY"
        )
        qwen["objective_success"] = False
        qwen["objective_disposition"] = "UNSUCCESSFUL"
        with self.assertRaises(diagnostic.DiagnosticInvariantError):
            diagnostic.validate_redacted_output(payload)

        payload = _closed_payload()
        provenance = payload["provenance"]
        self.assertIsInstance(provenance, dict)
        occurrences = provenance["occurrences"]
        self.assertIsInstance(occurrences, list)
        occurrences[0]["target_task_id"] = _ref("outside-matrix-task")
        with self.assertRaises(diagnostic.DiagnosticInvariantError):
            diagnostic.validate_redacted_output(payload)

        payload = _closed_payload()
        matrix = payload["matrix"]
        self.assertIsInstance(matrix, list)
        matrix[0]["selected_execution_class_mismatch_count"] = 0
        with self.assertRaises(diagnostic.DiagnosticInvariantError):
            diagnostic.validate_redacted_output(payload)

        payload = _closed_payload()
        inputs = payload["inputs"]
        self.assertIsInstance(inputs, dict)
        for section_name in ("before", "after"):
            section = inputs[section_name]
            self.assertIsInstance(section, list)
            repo_witness = next(
                row for row in section if row["label"] == "R2_RUNNER"
            )
            repo_witness["mode"] = 0o600
        with self.assertRaises(diagnostic.DiagnosticInvariantError):
            diagnostic.validate_redacted_output(payload)

    def test_redaction_rejects_forbidden_keys_values_and_reversible_encodings(self) -> None:
        raw = "SYNTHETIC_PRIVATE_TRACE_VALUE"
        encoded_raw = base64.b64encode(raw.encode("utf-8")).decode("ascii")
        base32_raw = base64.b32encode(raw.encode("utf-8")).decode("ascii")
        forbidden_payloads = (
            {"raw_response": raw},
            {"nested": {"recalled_text": raw}},
            {"nested": [{"proposal": raw}]},
            {"leak": raw},
            {"leak": raw.encode("utf-8").hex()},
            {"leak": encoded_raw},
            {"leak": base32_raw},
        )
        for unsafe in forbidden_payloads:
            payload = {
                "schema": diagnostic.DIAGNOSTIC_SCHEMA,
                "identity": diagnostic.DIAGNOSTIC_IDENTITY,
                **unsafe,
            }
            with self.subTest(unsafe=unsafe):
                with self.assertRaises(diagnostic.DiagnosticInvariantError):
                    diagnostic.validate_redacted_output(
                        payload,
                        forbidden_values=(raw,),
                    )

    def test_raw_witness_is_context_dominant_for_enum_and_ref_shaped_values(self) -> None:
        witness = diagnostic.RawMaterialWitness()
        ref_shaped = _ref("raw-shaped")
        witness.observe_document(
            {
                "raw_response": "SEEN",
                "feedback_text": ref_shaped,
                "content": "A_0000000000000001",
            },
            category="SYNTHETIC_RAW",
        )
        self.assertIn("SEEN", witness.forbidden_values())
        self.assertIn(ref_shaped, witness.forbidden_values())
        self.assertIn("A_0000000000000001", witness.forbidden_values())
        self.assertEqual(witness.category_counts, {"SYNTHETIC_RAW": 1})


class FrozenInputAndSQLiteTests(unittest.TestCase):
    def test_frozen_input_requires_each_exact_declared_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for expected_mode, drift_mode in ((0o600, 0o664), (0o664, 0o600)):
                path = Path(temporary) / f"synthetic-{expected_mode:o}.json"
                path.write_bytes(b'{"synthetic":true}')
                path.chmod(expected_mode)
                spec = diagnostic.FrozenInputSpec(
                    label=f"SYNTHETIC_MODE_{expected_mode:o}",
                    path=path,
                    sha256=_sha256(path),
                    kind="JSON",
                    expected_mode=expected_mode,
                )
                witness = diagnostic.capture_frozen_input(spec)
                self.assertEqual(witness.mode, expected_mode)

                path.chmod(drift_mode)
                with self.assertRaises(diagnostic.DiagnosticInvariantError):
                    diagnostic.capture_frozen_input(spec)

    def test_frozen_input_witness_detects_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.json"
            path.write_bytes(b'{"synthetic":true}')
            path.chmod(0o600)
            spec = diagnostic.FrozenInputSpec(
                label="SYNTHETIC_JSON",
                path=path,
                sha256=_sha256(path),
                kind="JSON",
                expected_mode=0o600,
            )
            witness = diagnostic.capture_frozen_input(spec)
            diagnostic.verify_frozen_input_unchanged(spec, witness)

            path.write_bytes(b'{"synthetic":false}')
            with self.assertRaises(diagnostic.DiagnosticInvariantError):
                diagnostic.verify_frozen_input_unchanged(spec, witness)

    def test_immutable_sqlite_is_query_only_and_creates_no_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.sqlite3"
            with sqlite3.connect(path) as connection:
                connection.execute("CREATE TABLE records (ref TEXT PRIMARY KEY, score INTEGER)")
                connection.execute("INSERT INTO records VALUES ('sha256:synthetic', 1)")
            path.chmod(0o600)
            before = _sha256(path)
            spec = diagnostic.FrozenInputSpec(
                label="SYNTHETIC_SQLITE",
                path=path,
                sha256=before,
                kind="SQLITE",
                expected_mode=0o600,
            )

            uri = diagnostic.immutable_sqlite_uri(spec)
            self.assertIn("mode=ro", uri)
            self.assertIn("immutable=1", uri)
            with diagnostic.immutable_sqlite(spec) as connection:
                self.assertEqual(connection.execute("PRAGMA query_only").fetchone(), (1,))
                self.assertEqual(connection.execute("SELECT score FROM records").fetchone(), (1,))
                with self.assertRaises(sqlite3.OperationalError):
                    connection.execute("INSERT INTO records VALUES ('sha256:no-write', 0)")

            self.assertEqual(_sha256(path), before)
            self.assertFalse(Path(f"{path}-wal").exists())
            self.assertFalse(Path(f"{path}-shm").exists())
            self.assertFalse(Path(f"{path}-journal").exists())

    def test_cognitive_scanner_rejoins_scope_local_sources_and_accepts_pending_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cognitive.sqlite3"
            spec, refs = _create_synthetic_cognitive_db(path)
            binding = diagnostic._LineageBinding(
                replicate_ref=_ref("replicate"),
                arm="FULL",
                scope_ref=_ref("scope"),
                spec=spec,
                hashes={
                    "acquisition_sequence": hashlib.sha256(b"sequence").hexdigest(),
                    "journal": hashlib.sha256(b"journal").hexdigest(),
                    "learner": hashlib.sha256(b"learner").hexdigest(),
                    "store": spec.sha256,
                },
            )
            witness = diagnostic.RawMaterialWitness()
            lineage = diagnostic._scan_cognitive_lineage(
                binding,
                raw_witness=witness,
                budget=diagnostic.DiagnosticResourceBudget(),
            )
            self.assertEqual(lineage.acquisition_count, 2)
            self.assertEqual(set(lineage.records), {
                refs["batch_record_ref"],
                refs["resolution_record_ref"],
            })
            joined = lineage.prospective_by_reservation[
                refs["legacy_reservation_ref"]
            ]
            self.assertEqual(joined.batch_ref, refs["batch_ref"])
            self.assertEqual(joined.resolution_ref, refs["resolution_ref"])
            self.assertEqual(joined.legacy_episode_ref, refs["episode_ref"])
            self.assertGreaterEqual(witness.category_counts["COGNITIVE_RECORD"], 2)

    def test_cognitive_scanner_rejects_acquisition_source_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cognitive.sqlite3"
            spec, _ = _create_synthetic_cognitive_db(
                path,
                mismatched_batch_acquisition=True,
            )
            binding = diagnostic._LineageBinding(
                replicate_ref=_ref("replicate"),
                arm="FULL",
                scope_ref=_ref("scope"),
                spec=spec,
                hashes={
                    "acquisition_sequence": hashlib.sha256(b"sequence").hexdigest(),
                    "journal": hashlib.sha256(b"journal").hexdigest(),
                    "learner": hashlib.sha256(b"learner").hexdigest(),
                    "store": spec.sha256,
                },
            )
            with self.assertRaises(diagnostic.DiagnosticInvariantError):
                diagnostic._scan_cognitive_lineage(
                    binding,
                    raw_witness=diagnostic.RawMaterialWitness(),
                    budget=diagnostic.DiagnosticResourceBudget(),
                )


class GrammarAndTraceTests(unittest.TestCase):
    def test_public_literal_grammar_classifies_shape_without_semantic_scoring(self) -> None:
        grammar = _grammar()
        self.assertEqual(
            diagnostic.classify_declared_grammar(grammar, "A_0000000000000001,STOP"),
            "CONFORMANT",
        )
        self.assertEqual(
            diagnostic.classify_declared_grammar(grammar, "STOP"),
            "CONFORMANT",
        )
        self.assertEqual(
            diagnostic.classify_declared_grammar(grammar, "A_ffffffffffffffff,STOP"),
            "FOREIGN_TOKEN",
        )
        self.assertEqual(
            diagnostic.classify_declared_grammar(grammar, "A_0000000000000001"),
            "MALFORMED_SHAPE",
        )
        self.assertEqual(diagnostic.classify_declared_grammar(grammar, "free form"), "OUTSIDE_DECLARED_GRAMMAR")
        self.assertEqual(diagnostic.classify_declared_grammar(grammar, ""), "EMPTY")
        self.assertEqual(
            diagnostic.classify_candidate(grammar, "A_0000000000000002,STOP"),
            "DECLARED_GRAMMAR_CONFORMANT",
        )
        self.assertEqual(
            diagnostic.classify_candidate(grammar, "STOP"),
            "DECLARED_GRAMMAR_CONFORMANT",
        )
        self.assertEqual(
            diagnostic.classify_declared_grammar(grammar, "STOP,STOP"),
            "MALFORMED_SHAPE",
        )

        public_a = {
            "actions": ["A_0000000000000001", "A_0000000000000002"],
            "family": "glyph-machine",
            "maximum_steps": 3,
            "synthetic_goal": "first",
        }
        public_b = {**public_a, "synthetic_goal": "second"}
        self.assertEqual(
            diagnostic.LiteralGrammar.from_public_glyph_task(public_a),
            diagnostic.LiteralGrammar.from_public_glyph_task(public_b),
        )

    def _evidence(
        self,
        *,
        arm: str,
        admitted: bool,
        success: bool,
    ) -> tuple[bytes, bytes, str, str]:
        task_id = _ref(f"extract-task-{arm}-{admitted}")
        receipt_ref = _ref(f"extract-receipt-{arm}-{admitted}")
        public_task = {
            "actions": ["A_0000000000000001", "A_0000000000000002"],
            "family": "glyph-machine",
            "maximum_steps": 3,
            "synthetic_goal": "never-inspected-for-correctness",
        }
        proposals = (
            ["A_0000000000000001,STOP", "explain a generic procedure"]
            if admitted
            else []
        )
        if arm == "QWEN_ONLY":
            recalled = [diagnostic.NO_HISTORY_SENTINEL]
            recall_ref = None
        else:
            recalled = ["synthetic retained record one", "synthetic retained record two"]
            recall_ref = _ref("synthetic-recall-batch")
        selection: dict[str, object] | None
        if not admitted:
            selection = None
        elif arm in {"QWEN_ONLY", "RETRIEVAL_ONLY"}:
            selection = {
                "selected_index": 0,
                "selected_trace": proposals[0],
                "selection_ref": _ref(f"control-selection-{arm}"),
            }
        else:
            selection = {
                "decision_evidence_ref": _ref(f"decision-{arm}"),
                "reservation_ref": _ref(f"reservation-{arm}"),
                "selected_index": 0,
                "selected_trace": proposals[0],
            }
        raw_response = "A_0000000000000002,STOP" if admitted else ""
        stage = {
            "arm": arm,
            "attempt_receipt_ref": receipt_ref,
            "execution_receipt": (
                {"execution_receipt_ref": _ref(f"execution-{arm}")}
                if admitted
                else None
            ),
            "execution_request": (
                {"execution_request_ref": _ref(f"request-{arm}")}
                if admitted
                else None
            ),
            "frozen_recall_ref": recall_ref,
            "parser_disposition": "ADMITTED" if admitted else "MALFORMED",
            "phase": "final",
            "proposal_request": {
                "recalled_evidence": recalled,
                "task": diagnostic.canonical_json_bytes(public_task).decode("ascii"),
            },
            "proposals": proposals,
            "raw_response": raw_response,
            "selection": selection,
            "task_id": task_id,
        }
        objective = {
            "arm": arm,
            "attempt_receipt_ref": receipt_ref,
            "disposition": "SUCCESS" if success else "UNSUCCESSFUL",
            "raw_response": raw_response,
            "score": 1.0 if success else 0.0,
            "task_id": task_id,
        }
        judgment = {
            "arm": arm,
            "attempt_receipt_ref": receipt_ref,
            "learner_transition": None,
            "objective_judgment": objective,
            "task_id": task_id,
        }
        return (
            diagnostic.canonical_json_bytes(stage),
            diagnostic.canonical_json_bytes(judgment),
            diagnostic.content_ref("attempt-stage", stage),
            diagnostic.content_ref("attempt-judgment", judgment),
        )

    def test_extracts_admitted_and_malformed_without_retaining_raw_text(self) -> None:
        stage, judgment, stage_ref, judgment_ref = self._evidence(
            arm="FULL",
            admitted=True,
            success=False,
        )
        admitted = diagnostic.extract_attempt_trace(
            stage,
            judgment,
            stage_ref=stage_ref,
            judgment_ref=judgment_ref,
            recalled_record_count=2,
        )
        self.assertEqual(admitted.parser_disposition, "ADMITTED")
        self.assertEqual(admitted.history_class, "FROZEN_RECALL")
        self.assertEqual(admitted.candidate_classes[0], "DECLARED_GRAMMAR_CONFORMANT")
        self.assertEqual(admitted.selected_index, 0)
        self.assertEqual(admitted.response_class, "CONFORMANT")
        self.assertFalse(admitted.selected_response_equal)
        serialized = diagnostic.canonical_json_bytes(admitted.redacted())
        for raw in (
            b"synthetic retained record",
            b"explain a generic procedure",
            b"never-inspected-for-correctness",
        ):
            self.assertNotIn(raw, serialized)

        stage, judgment, stage_ref, judgment_ref = self._evidence(
            arm="QWEN_ONLY",
            admitted=False,
            success=False,
        )
        malformed = diagnostic.extract_attempt_trace(
            stage,
            judgment,
            stage_ref=stage_ref,
            judgment_ref=judgment_ref,
        )
        self.assertEqual(malformed.parser_disposition, "MALFORMED")
        self.assertEqual(malformed.history_class, "NO_PERSISTENT_HISTORY")
        self.assertEqual(malformed.candidate_classes, ())
        self.assertEqual(malformed.selected_class, "NOT_SELECTED")
        self.assertEqual(malformed.response_class, "NOT_EXECUTED")
        self.assertIsNone(malformed.execution_receipt_ref)

    def test_extract_accepts_irrelevant_exact_nfc_utf8_public_task_field(self) -> None:
        stage, judgment, _, judgment_ref = self._evidence(
            arm="FULL",
            admitted=True,
            success=False,
        )
        parsed_stage = diagnostic.parse_canonical_json(stage)
        self.assertIsInstance(parsed_stage, dict)
        proposal_request = parsed_stage["proposal_request"]  # type: ignore[index]
        self.assertIsInstance(proposal_request, dict)
        public_task = diagnostic.parse_canonical_json(proposal_request["task"])
        self.assertIsInstance(public_task, dict)
        public_task["display_label"] = "caf\N{LATIN SMALL LETTER E WITH ACUTE}"
        proposal_request["task"] = diagnostic.canonical_json_bytes(public_task).decode(
            "utf-8"
        )
        changed_stage = diagnostic.canonical_json_bytes(parsed_stage)
        changed_stage_ref = diagnostic.content_ref("attempt-stage", parsed_stage)

        observed = diagnostic.extract_attempt_trace(
            changed_stage,
            judgment,
            stage_ref=changed_stage_ref,
            judgment_ref=judgment_ref,
            recalled_record_count=2,
        )
        self.assertEqual(observed.parser_disposition, "ADMITTED")
        self.assertEqual(observed.response_class, "CONFORMANT")
        self.assertNotIn(
            "caf\N{LATIN SMALL LETTER E WITH ACUTE}".encode("utf-8"),
            diagnostic.canonical_json_bytes(observed.redacted()),
        )

    def test_extraction_rejects_binding_or_recall_cardinality_drift(self) -> None:
        stage, judgment, stage_ref, judgment_ref = self._evidence(
            arm="FULL",
            admitted=True,
            success=False,
        )
        with self.assertRaises(diagnostic.DiagnosticInvariantError):
            diagnostic.extract_attempt_trace(
                stage,
                judgment,
                stage_ref=stage_ref,
                judgment_ref=judgment_ref,
                recalled_record_count=1,
            )

        parsed = diagnostic.parse_canonical_json(judgment)
        self.assertIsInstance(parsed, dict)
        parsed["task_id"] = _ref("different-task")  # type: ignore[index]
        with self.assertRaises(diagnostic.DiagnosticInvariantError):
            diagnostic.extract_attempt_trace(
                stage,
                diagnostic.canonical_json_bytes(parsed),
                stage_ref=stage_ref,
                judgment_ref=judgment_ref,
            )

    def test_attempt_stage_and_judgment_domains_recompute_exactly_and_reject_mutation(self) -> None:
        stage, judgment, stage_ref, judgment_ref = self._evidence(
            arm="FULL",
            admitted=True,
            success=False,
        )
        self.assertEqual(
            stage_ref,
            "sha256:"
            + hashlib.sha256(b"attempt-stage\x00" + stage).hexdigest(),
        )
        self.assertEqual(
            judgment_ref,
            "sha256:"
            + hashlib.sha256(b"attempt-judgment\x00" + judgment).hexdigest(),
        )
        self.assertNotEqual(
            diagnostic.content_ref("attempt-stage", diagnostic.parse_canonical_json(judgment)),
            judgment_ref,
        )

        changed_stage = diagnostic.parse_canonical_json(stage)
        self.assertIsInstance(changed_stage, dict)
        changed_stage["phase"] = "development"  # type: ignore[index]
        with self.assertRaises(diagnostic.DiagnosticInvariantError):
            diagnostic.extract_attempt_trace(
                diagnostic.canonical_json_bytes(changed_stage),
                judgment,
                stage_ref=stage_ref,
                judgment_ref=judgment_ref,
            )

        changed_judgment = diagnostic.parse_canonical_json(judgment)
        self.assertIsInstance(changed_judgment, dict)
        changed_judgment["audit_marker"] = "MUTATED"  # type: ignore[index]
        with self.assertRaises(diagnostic.DiagnosticInvariantError):
            diagnostic.extract_attempt_trace(
                stage,
                diagnostic.canonical_json_bytes(changed_judgment),
                stage_ref=stage_ref,
                judgment_ref=judgment_ref,
            )

    def test_attempt_receipt_and_removal_arm_rejoin_decoded_ledger_evidence(self) -> None:
        task_id = _ref("removal-task")
        execution_receipt_ref = _ref("source-execution-receipt")
        receipt_payload = {
            "arm": "FULL",
            "nonauthorization": diagnostic._ATTEMPT_RECEIPT_NONAUTHORIZATION,
            "parser_disposition": "ADMITTED",
            "phase": "final",
            "purpose": "evaluation",
            "schema": diagnostic._ATTEMPT_RECEIPT_SCHEMA,
            "source_kind": "EXECUTION_RECEIPT",
            "source_receipt_ref": execution_receipt_ref,
            "task_id": task_id,
        }
        attempt_receipt_ref = diagnostic.content_ref(
            "arm-attempt-receipt", receipt_payload
        )
        receipt_stage = {
            "arm": "FULL",
            "attempt_receipt": receipt_payload,
            "attempt_receipt_ref": attempt_receipt_ref,
            "execution_receipt": {
                "execution_receipt_ref": execution_receipt_ref,
            },
            "parser_disposition": "ADMITTED",
            "phase": "final",
            "proposal_generation": {"generation_ref": _ref("proposal-generation")},
            "purpose": "evaluation",
            "task_id": task_id,
        }
        self.assertEqual(
            diagnostic._validate_attempt_receipt(receipt_stage),
            attempt_receipt_ref,
        )
        changed_receipt = {**receipt_payload, "source_kind": "PROPOSAL_GENERATION"}
        with self.assertRaises(diagnostic.DiagnosticInvariantError):
            diagnostic._validate_attempt_receipt(
                {**receipt_stage, "attempt_receipt": changed_receipt}
            )

        request_payload = {
            "contract": "SYNTHETIC_EXECUTION_REQUEST",
            "request": "SYNTHETIC_REQUEST",
        }
        request_ref, request_bytes = _blob(request_payload)
        response_payload = {
            "contract": "SYNTHETIC_EXECUTION_RECEIPT",
            "response": "SYNTHETIC_RESPONSE",
        }
        response_ref, response_bytes = _blob(response_payload)
        proposal_generation = {
            "generation_ref": _ref("proposal-generation"),
            "prompt_ref": _ref("proposal-prompt"),
            "response": "SYNTHETIC_PROPOSAL",
        }
        proposal_request = {
            "recalled_evidence": ["SYNTHETIC_RECALL"],
            "task": "SYNTHETIC_TASK",
        }
        selection = {
            "selected_index": 0,
            "selected_trace": "SYNTHETIC_PROPOSAL",
            "selection_ref": _ref("selection"),
        }
        task_generation = {
            "generation_ref": _ref("response-generation"),
            "prompt_ref": _ref("execution-prompt"),
            "response": "SYNTHETIC_RESPONSE",
        }
        quiescence = {"status": "QUIESCENT"}
        stage = {
            "execution_receipt": {
                "execution_receipt_ref": response_ref,
                **response_payload,
            },
            "execution_request": {
                "execution_request_ref": request_ref,
                **request_payload,
            },
            "frozen_recall_ref": _ref("recall-batch"),
            "proposal_generation": proposal_generation,
            "proposal_request": proposal_request,
            "proposals": ["SYNTHETIC_PROPOSAL"],
            "public_task_ref": _ref("public-task"),
            "raw_response": "SYNTHETIC_RESPONSE",
            "selection": selection,
            "task_response_generation": task_generation,
        }
        judgment = {
            "objective_judgment": {"score": 1.0},
            "probe_integrity_ref": _ref("probe"),
            "runtime_quiescence": quiescence,
            "runtime_quiescence_ref": diagnostic.content_ref(
                "runtime-quiescence", quiescence
            ),
        }
        stage_ref = diagnostic.content_ref("attempt-stage", stage)
        judgment_ref = diagnostic.content_ref("attempt-judgment", judgment)
        ledger = diagnostic._LedgerRow(
            task_id=task_id,
            arm="FULL",
            phase="final",
            family="glyph-machine",
            replicate_ref=_ref("replicate"),
            receipt_ref=attempt_receipt_ref,
            stage_ref=stage_ref,
            judgment_ref=judgment_ref,
            stage_bytes=diagnostic.canonical_json_bytes(stage),
            judgment_bytes=diagnostic.canonical_json_bytes(judgment),
            stage=stage,
            judgment=judgment,
            success=True,
        )
        recalled_ref = _ref("retained-record")
        removal = {
            "backend_raw_hit_count": 1,
            "backend_rejected_hit_count": 0,
            "backend_search_calls": 1,
            "evidence_final_ref": judgment_ref,
            "evidence_stage_ref": stage_ref,
            "execution_prompt_ref": task_generation["prompt_ref"],
            "execution_receipt_bytes": response_bytes.hex(),
            "execution_receipt_ref": response_ref,
            "execution_request_bytes": request_bytes.hex(),
            "execution_request_ref": request_ref,
            "frozen_recall_ref": stage["frozen_recall_ref"],
            "proposal_generation_bytes": diagnostic.canonical_json_bytes(
                proposal_generation
            ).hex(),
            "proposal_generation_ref": proposal_generation["generation_ref"],
            "proposal_prompt_ref": proposal_generation["prompt_ref"],
            "proposal_request_bytes": diagnostic.canonical_json_bytes(
                proposal_request
            ).hex(),
            "probe_integrity_ref": judgment["probe_integrity_ref"],
            "proposals": stage["proposals"],
            "public_task_ref": stage["public_task_ref"],
            "raw_response": stage["raw_response"],
            "recalled_record_bytes_sha256": [recalled_ref.removeprefix("sha256:")],
            "recalled_record_refs": [recalled_ref],
            "runtime_quiescence_ref": judgment["runtime_quiescence_ref"],
            "score": 1.0,
            "selection_bytes": diagnostic.canonical_json_bytes(selection).hex(),
            "selection_ref": selection["selection_ref"],
            "selected_trace": selection["selected_trace"],
            "task_id": task_id,
            "task_response_generation_bytes": diagnostic.canonical_json_bytes(
                task_generation
            ).hex(),
            "task_response_generation_ref": task_generation["generation_ref"],
        }
        self.assertEqual(
            diagnostic._validate_removal_arm(removal, arm="FULL", ledger=ledger),
            removal,
        )
        tampered = {**removal, "execution_request_bytes": response_bytes.hex()}
        with self.assertRaises(diagnostic.DiagnosticInvariantError):
            diagnostic._validate_removal_arm(tampered, arm="FULL", ledger=ledger)

        stateful_selection = {
            "decision_evidence_ref": _ref("stateful-decision"),
            "reservation_ref": _ref("stateful-reservation"),
            "selected_index": 0,
            "selected_trace": "SYNTHETIC_PROPOSAL",
        }
        stateful_stage = {**stage, "selection": stateful_selection}
        stateful_stage_ref = diagnostic.content_ref("attempt-stage", stateful_stage)
        stateful_ledger = replace(
            ledger,
            stage=stateful_stage,
            stage_bytes=diagnostic.canonical_json_bytes(stateful_stage),
            stage_ref=stateful_stage_ref,
        )
        stateful_removal = {
            **removal,
            "evidence_stage_ref": stateful_stage_ref,
            "selection_bytes": diagnostic.canonical_json_bytes(
                stateful_selection
            ).hex(),
            "selection_ref": stateful_selection["reservation_ref"],
        }
        self.assertEqual(
            diagnostic._validate_removal_arm(
                stateful_removal,
                arm="FULL",
                ledger=stateful_ledger,
            ),
            stateful_removal,
        )
        with self.assertRaises(diagnostic.DiagnosticInvariantError):
            diagnostic._validate_removal_arm(
                {
                    **stateful_removal,
                    "selection_ref": stateful_selection["decision_evidence_ref"],
                },
                arm="FULL",
                ledger=stateful_ledger,
            )

    def test_public_task_grammar_rejoins_exact_manifest_commitment(self) -> None:
        first_replicate = _ref("first-replicate")
        second_replicate = _ref("second-replicate")
        payload = {
            "actions": ["A_0000000000000001", "A_0000000000000002"],
            "family": "glyph-machine",
            "maximum_steps": 3,
        }
        rendered = diagnostic.canonical_json_bytes(payload).decode("utf-8")
        public_ref = diagnostic._evaluator_record_digest(
            "public-task",
            {
                "family": "glyph-machine",
                "ordinal": 0,
                "payload_json": rendered,
                "phase": "final",
                "replicate": "replicate-01",
                "replicate_commitment": first_replicate,
            },
        )
        second_public_ref = diagnostic._evaluator_record_digest(
            "public-task",
            {
                "family": "glyph-machine",
                "ordinal": 0,
                "payload_json": rendered,
                "phase": "final",
                "replicate": "replicate-02",
                "replicate_commitment": second_replicate,
            },
        )
        task = {
            "family": "glyph-machine",
            "ordinal": 0,
            "phase": "final",
            "public_commitment": public_ref,
            "replicate_commitment": first_replicate,
        }
        tasks = {
            _ref("task-a"): task,
            _ref("task-b"): {
                **task,
                "public_commitment": second_public_ref,
                "replicate_commitment": second_replicate,
            },
        }
        labels = diagnostic._manifest_replicate_labels(tasks)
        self.assertEqual(
            labels,
            {
                first_replicate: "replicate-01",
                second_replicate: "replicate-02",
            },
        )
        stage = {
            "proposal_request": {"task": rendered},
            "public_task_ref": public_ref,
        }
        self.assertEqual(
            diagnostic._validate_public_task_binding(
                stage,
                task,
                replicate_labels=labels,
            ),
            payload,
        )
        second_task = tasks[_ref("task-b")]
        self.assertEqual(
            diagnostic._validate_public_task_binding(
                {
                    "proposal_request": {"task": rendered},
                    "public_task_ref": second_public_ref,
                },
                second_task,
                replicate_labels=labels,
            ),
            payload,
        )
        with self.assertRaises(diagnostic.DiagnosticInvariantError):
            diagnostic._validate_public_task_binding(
                {
                    "proposal_request": {"task": rendered},
                    "public_task_ref": second_public_ref,
                },
                second_task,
                replicate_labels={
                    **labels,
                    second_replicate: "replicate-01",
                },
            )

        changed_payload = {**payload, "maximum_steps": 4}
        changed_stage = {
            **stage,
            "proposal_request": {
                "task": diagnostic.canonical_json_bytes(changed_payload).decode("utf-8")
            },
        }
        with self.assertRaises(diagnostic.DiagnosticInvariantError):
            diagnostic._validate_public_task_binding(
                changed_stage,
                task,
                replicate_labels=labels,
            )
        with self.assertRaises(diagnostic.DiagnosticInvariantError):
            diagnostic._validate_public_task_binding(
                {**stage, "public_task_ref": _ref("substituted-public-task")},
                task,
                replicate_labels=labels,
            )


class CohortEqualityAndProvenanceTests(unittest.TestCase):
    def _cohort(self) -> list[diagnostic.AttemptTrace]:
        return [
            _attempt(
                task,
                arm,
                success=arm == "QWEN_ONLY",
            )
            for task in range(3)
            for arm in diagnostic.EXPECTED_ARMS
        ]

    def test_exact_three_by_seven_cohort_and_uniqueness(self) -> None:
        cohort = self._cohort()
        self.assertEqual(
            diagnostic.validate_unique_cohort(cohort),
            tuple(sorted({_ref(f"task-{item}") for item in range(3)})),
        )

        with self.assertRaises(diagnostic.DiagnosticInvariantError):
            diagnostic.validate_unique_cohort(cohort[:-1])

        duplicate_arm = list(cohort)
        duplicate_arm[-1] = _attempt(
            2,
            "FULL",
            receipt_suffix="second-full",
        )
        with self.assertRaises(diagnostic.DiagnosticInvariantError):
            diagnostic.validate_unique_cohort(duplicate_arm)

        no_qwen_success = [
            replace(item, objective_success=False, objective_disposition="UNSUCCESSFUL")
            if item.arm == "QWEN_ONLY"
            else item
            for item in cohort
        ]
        with self.assertRaises(diagnostic.DiagnosticInvariantError):
            diagnostic.validate_unique_cohort(no_qwen_success)

    def test_equality_groups_and_first_divergence_are_exact_and_ordered(self) -> None:
        self.assertEqual(
            diagnostic.equality_groups(
                {
                    "FULL": {"ref": _ref("same")},
                    "QWEN_ONLY": {"ref": _ref("other")},
                    "RETRIEVAL_ONLY": {"ref": _ref("same")},
                }
            ),
            (("FULL", "RETRIEVAL_ONLY"), ("QWEN_ONLY",)),
        )
        reference = {
            "parser_disposition": "ADMITTED",
            "history_preimage": _ref("history-a"),
            "candidate_classes": ["DECLARED_GRAMMAR_CONFORMANT"],
            "selection": 0,
            "response_conformance": "CONFORMANT",
            "objective_disposition": "SUCCESS",
            "objective_score": 1.0,
        }
        observed = {**reference, "candidate_classes": ["OUTSIDE_DECLARED_GRAMMAR"]}
        self.assertEqual(diagnostic.first_divergence(reference, observed), "CANDIDATE_CLASSES")
        observed["parser_disposition"] = "MALFORMED"
        self.assertEqual(diagnostic.first_divergence(reference, observed), "PARSER_DISPOSITION")
        self.assertEqual(diagnostic.first_divergence(reference, reference), "NONE")

    def test_provenance_rejoins_scope_local_repeated_occurrences(self) -> None:
        record = _ref("record")
        ancestry = _ref("adaptation-episode")
        first_lineage = _ref("lineage")
        second_lineage = _ref("second-lineage")
        row = diagnostic.ProvenanceRow(
            record_ref=record,
            lineage_ref=first_lineage,
            ancestry_refs=(ancestry,),
        )
        clone_row = diagnostic.ProvenanceRow(
            record_ref=record,
            lineage_ref=second_lineage,
            ancestry_refs=(ancestry,),
        )
        summary = diagnostic.rejoin_provenance(
            ((first_lineage, record), (first_lineage, record), (second_lineage, record)),
            (row, clone_row),
            required_ancestry_refs=(ancestry,),
        )
        self.assertEqual(
            summary,
            {
                "all_rejoined": True,
                "ancestry_ref_count": 1,
                "lineage_count": 2,
                "occurrence_count": 3,
                "provenance_classes": ["RETAINED_ACQUISITION"],
                "rejoined_ancestry_ref_count": 1,
                "unique_record_count": 2,
            },
        )
        with self.assertRaises(diagnostic.DiagnosticInvariantError):
            diagnostic.rejoin_provenance(((first_lineage, _ref("missing")),), (row,))
        with self.assertRaises(diagnostic.DiagnosticInvariantError):
            diagnostic.rejoin_provenance(((first_lineage, record),), (row, row))


class PublicationTests(unittest.TestCase):
    def _payload(self) -> dict[str, object]:
        return _closed_payload()

    def test_direct_publication_is_create_once_mode_0600_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "diagnostic.json"
            payload = self._payload()
            encoded = diagnostic.canonical_json_bytes(payload)
            witness = diagnostic.publish_create_once(
                path,
                payload,
                expected_path=path,
            )
            metadata = path.lstat()
            self.assertEqual(path.read_bytes(), encoded)
            self.assertEqual(stat.S_IMODE(metadata.st_mode), 0o600)
            self.assertTrue(stat.S_ISREG(metadata.st_mode))
            self.assertEqual(metadata.st_nlink, 1)
            self.assertEqual(witness.sha256, _sha256(path))
            self.assertEqual(witness.size_bytes, len(encoded))
            self.assertEqual(tuple(path.parent.iterdir()), (path,))

            with self.assertRaises(diagnostic.DiagnosticInvariantError):
                diagnostic.publish_create_once(path, payload, expected_path=path)
            self.assertEqual(path.read_bytes(), encoded)

    def test_publication_rejects_wrong_expected_path_symlink_and_oversize(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "diagnostic.json"
            payload = self._payload()
            with self.assertRaises(diagnostic.DiagnosticInvariantError):
                diagnostic.publish_create_once(
                    path,
                    payload,
                    expected_path=root / "other.json",
                )
            self.assertFalse(path.exists())

            target = root / "target.json"
            target.write_bytes(b"preserve")
            path.symlink_to(target)
            with self.assertRaises(diagnostic.DiagnosticInvariantError):
                diagnostic.publish_create_once(path, payload, expected_path=path)
            self.assertEqual(target.read_bytes(), b"preserve")

            path.unlink()
            oversized = self._payload()
            with self.assertRaises(diagnostic.DiagnosticInvariantError):
                diagnostic.publish_create_once(
                    path,
                    oversized,
                    expected_path=path,
                    maximum_bytes=128,
                )
            self.assertFalse(path.exists())

    def test_publication_revalidates_forbidden_values_before_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "diagnostic.json"
            raw = "SYNTHETIC_PRIVATE_RESPONSE"
            payload = {**self._payload(), "leak": raw}
            with self.assertRaises(diagnostic.DiagnosticInvariantError):
                diagnostic.publish_create_once(
                    path,
                    payload,
                    expected_path=path,
                    forbidden_values=(raw,),
                )
            self.assertFalse(path.exists())

    def test_two_publishers_race_to_exactly_one_exclusive_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "diagnostic.json"
            payload = self._payload()

            def publish() -> bool:
                try:
                    diagnostic.publish_create_once(path, payload, expected_path=path)
                except diagnostic.DiagnosticInvariantError:
                    return False
                return True

            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = tuple(pool.map(lambda _: publish(), range(2)))
            self.assertEqual(sorted(outcomes), [False, True])
            self.assertEqual(path.read_bytes(), diagnostic.canonical_json_bytes(payload))
            self.assertEqual(tuple(path.parent.iterdir()), (path,))


class ComposerIntegrationTests(unittest.TestCase):
    def test_injected_composition_reaches_exclusive_publication_and_join_failure_stops_before_open(self) -> None:
        payload = _closed_payload()
        target_ids = tuple(
            str(row["task_ref"]) for row in payload["matrix"]  # type: ignore[index]
        )
        ledger_rows = {
            (f"synthetic-{index}", "FULL"): SimpleNamespace(
                stage={
                    "parser_disposition": "ADMITTED" if index < 433 else "MALFORMED"
                }
            )
            for index in range(468)
        }
        ledger = SimpleNamespace(
            rows=ledger_rows,
            receipt_refs=frozenset(_ref(f"receipt-{index}") for index in range(468)),
            run_intent_ref=_ref("run-intent"),
            metric_counts={},
        )
        removals = SimpleNamespace(
            records={_ref(f"removal-{index}"): {} for index in range(60)},
            evidence_refs=tuple(_ref(f"removal-evidence-{index}") for index in range(60)),
        )
        lineage_keys = {
            (_ref(f"replicate-{replicate}"), arm): SimpleNamespace(
                label=f"binding-{replicate}-{arm}"
            )
            for replicate in range(2)
            for arm in ("FULL", "RANDOM_FEEDBACK")
        }
        factory = SimpleNamespace(
            lineages=lineage_keys,
            disposition_refs=tuple(_ref(f"disposition-{index}") for index in range(468)),
        )
        cognitive_lineage = SimpleNamespace(records={_ref("record"): object()})
        fake_raw_witness = mock.Mock(spec=diagnostic.RawMaterialWitness)
        fake_raw_witness.forbidden_values.return_value = ()
        events: list[str] = []

        def file_witness(spec: diagnostic.FrozenInputSpec) -> diagnostic.FileWitness:
            return diagnostic.FileWitness(
                label=spec.label,
                path=spec.path,
                sha256=spec.sha256,
                size_bytes=1,
                mode=spec.expected_mode,
                uid=os.geteuid(),
                gid=os.getegid(),
                nlink=1,
                device=1,
                inode=1 + tuple(diagnostic.FROZEN_INPUTS).index(spec),
                mtime_ns=1,
            )

        def capture(spec: diagnostic.FrozenInputSpec) -> diagnostic.FileWitness:
            events.append("capture")
            return file_witness(spec)

        def verify(
            _spec: diagnostic.FrozenInputSpec,
            before: diagnostic.FileWitness,
        ) -> diagnostic.FileWitness:
            events.append("verify")
            return before

        def read_json(
            spec: diagnostic.FrozenInputSpec,
        ) -> tuple[dict[str, object], diagnostic.FileWitness]:
            events.append(f"read:{spec.label}")
            return {"synthetic": spec.label}, file_witness(spec)

        def sequenced(name: str, value: object):
            def call(*_args: object, **_kwargs: object) -> object:
                events.append(name)
                return value

            return call

        real_publish = diagnostic.publish_create_once
        real_validate = diagnostic.validate_redacted_output

        def publish(*args: object, **kwargs: object) -> diagnostic.FileWitness:
            events.append("publish")
            return real_publish(*args, **kwargs)  # type: ignore[arg-type]

        def validate(*args: object, **kwargs: object) -> None:
            events.append("validate")
            real_validate(*args, **kwargs)  # type: ignore[arg-type]

        with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
            output = Path(temporary) / "diagnostic.json"
            stack.enter_context(mock.patch.object(diagnostic, "OUTPUT_PATH", output))
            stack.enter_context(
                mock.patch.object(diagnostic, "RawMaterialWitness", return_value=fake_raw_witness)
            )
            stack.enter_context(
                mock.patch.object(diagnostic, "capture_frozen_input", side_effect=capture)
            )
            stack.enter_context(
                mock.patch.object(
                    diagnostic,
                    "verify_frozen_input_unchanged",
                    side_effect=verify,
                )
            )
            stack.enter_context(
                mock.patch.object(diagnostic, "_read_frozen_json", side_effect=read_json)
            )
            stack.enter_context(
                mock.patch.object(
                    diagnostic,
                    "_manifest_task_map",
                    side_effect=sequenced("manifest", {}),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    diagnostic,
                    "_result_attempt_bindings",
                    side_effect=sequenced("result", ({}, set(), {})),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    diagnostic,
                    "_stream_ledger",
                    side_effect=sequenced("ledger", ledger),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    diagnostic,
                    "_validate_removal_dataset",
                    side_effect=sequenced("removal", removals),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    diagnostic,
                    "_validate_factory_audit",
                    side_effect=sequenced("factory", factory),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    diagnostic,
                    "_scan_cognitive_lineage",
                    side_effect=sequenced("cognitive", cognitive_lineage),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    diagnostic,
                    "_target_task_ids",
                    side_effect=sequenced("targets", target_ids),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    diagnostic,
                    "_build_trace_matrix",
                    side_effect=sequenced("matrix", (payload["matrix"], {})),
                )
            )
            provenance_mock = stack.enter_context(
                mock.patch.object(
                    diagnostic,
                    "_rejoin_target_provenance",
                    side_effect=sequenced("provenance", payload["provenance"]),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    diagnostic,
                    "_question_rows",
                    side_effect=sequenced("questions", payload["questions"]),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    diagnostic,
                    "validate_redacted_output",
                    side_effect=validate,
                )
            )
            publish_mock = stack.enter_context(
                mock.patch.object(
                    diagnostic,
                    "publish_create_once",
                    side_effect=publish,
                )
            )

            witness = diagnostic._compose_diagnostic(
                diagnostic.DiagnosticResourceBudget()
            )
            self.assertEqual(witness.path, output)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            composed = diagnostic.parse_canonical_json(output.read_bytes())
            self.assertIsInstance(composed, dict)
            self.assertFalse(
                composed["provenance"]["prospective_contract_recomputed"]  # type: ignore[index]
            )
            required_order = (
                "manifest",
                "result",
                "ledger",
                "removal",
                "factory",
                "cognitive",
                "targets",
                "matrix",
                "provenance",
                "questions",
                "validate",
                "publish",
            )
            positions = [events.index(name) for name in required_order]
            self.assertEqual(positions, sorted(positions))

            failure_output = Path(temporary) / "join-failure.json"
            diagnostic.OUTPUT_PATH = failure_output
            provenance_mock.side_effect = diagnostic.DiagnosticInvariantError(
                "synthetic provenance join failure"
            )
            publish_mock.reset_mock()
            with self.assertRaises(diagnostic.DiagnosticInvariantError):
                diagnostic._compose_diagnostic(diagnostic.DiagnosticResourceBudget())
            publish_mock.assert_not_called()
            self.assertFalse(failure_output.exists())


class StaticBoundaryTests(unittest.TestCase):
    def test_frozen_input_inventory_is_exact_and_includes_predecessor_leaf(self) -> None:
        expected = {
            "R2_RESULT": (
                "/opt/angler/results/high-level-multidomain-v1-r2.json",
                "c2316da22208cf0a8693f32f4ea28060c529329df1c6d7d67841b56720aca676",
                "JSON",
            ),
            "R2_RESULT_REPORT": (
                "/opt/angler/src/angler/docs/reports/HIGH_LEVEL_MULTIDOMAIN_ADAPTATION_V1_R2_RESULT.md",
                "4aae9d581333eebc468ed5c44f52c62b3784cc98ac6896afada70628c95960d1",
                "REPORT",
            ),
            "R2_LEDGER": (
                "/opt/angler/state/project-angler/high-level-multidomain-v1-r2/evaluation-v1-r2/attempt-evidence.sqlite3",
                "914e1695cc4c9187a60f48a035632dbe5e849056e113bab4295ca36f5bbb4eca",
                "SQLITE",
            ),
            "R2_ARM_FACTORY_AUDIT": (
                "/opt/angler/state/project-angler/high-level-multidomain-v1-r2/evaluation-v1-r2/arm-factory-audit.json",
                "eb6f4e16336f83b4cee95ecc9ec4803af99045dd1ba904373599db8a3f0b4e82",
                "JSON",
            ),
            "R2_SOURCE_MANIFEST": (
                "/opt/angler/src/angler/experiments/manifests/high-level-multidomain-v1-r2.json",
                "842e39e713c87352a9dbc5e04c8cd73ab74a1b32fceb9aaa72c1d1fd75a4344d",
                "JSON",
            ),
            "R2_RUNNER": (
                "/opt/angler/src/angler/experiments/runners/high_level_multidomain_v1_r2.py",
                "fbd10104ee566e327e812f24e6ea44d30e405d4a2dea2650ebd94dad58fee574",
                "SOURCE",
            ),
            "COGNITIVE_TRANSACTION_STORE_SOURCE": (
                "/opt/angler/src/angler/src/angler/runtime/cognitive_transaction_store.py",
                "4c8add9e1b49649b381b0f4dad439709f7b4691cec514c88b39b386bc8a28c0a",
                "SOURCE",
            ),
            "COGNITIVE_ACQUISITION_SOURCE": (
                "/opt/angler/src/angler/src/angler/memory/cognitive_acquisition.py",
                "e93c9f181fe1938a4c033a55a88827ba3d0ca5b8b9f46cb1b5569bc4987d36e7",
                "SOURCE",
            ),
            "COGNITIVE_CONTRACTS_SOURCE": (
                "/opt/angler/src/angler/src/angler/cognition/contracts.py",
                "cb2fe5d597c10550425a03951aaecf2e9b104065e6ea32adaba30d1d61b1eb77",
                "SOURCE",
            ),
            "R2_PREDECESSOR_LEAF": (
                "/opt/angler/src/angler/docs/blueprints/branches/science/work/ANG-WORK-SCIENCE-HIGH-LEVEL-MULTIDOMAIN-ADAPTATION-V1-R2-LINEAGE-SEQUENCE-RECOVERY-001.md",
                "a07ce25685bfa3dd501edde2e1df84229faa379ab111ee5291e8f62fd8764455",
                "LEAF",
            ),
            "DIAGNOSTIC_LEAF": (
                "/opt/angler/src/angler/docs/blueprints/branches/science/work/ANG-WORK-SCIENCE-HIGH-LEVEL-MULTIDOMAIN-ADAPTATION-V1-R2-EVIDENCE-TRACE-DIAGNOSIS-001.md",
                "eedc848394a088e347db68db23f24fabab409315a0775747597edc76abd5f960",
                "LEAF",
            ),
            "LINEAGE_102E3196": (
                "/opt/angler/state/project-angler/high-level-multidomain-v1-r2/evaluation-v1-r2/arm-runtime/lineages/102e3196e05f0ce031e8585c2eef0f94b48acf1e91e272bc998d5b0cbc30a395/cognitive.sqlite3",
                "54a58c587c2c79ec27999ff8f9c48e1607f64e4f7c8d1fcca3e17af6f2d72a57",
                "SQLITE",
            ),
            "LINEAGE_75CADC26": (
                "/opt/angler/state/project-angler/high-level-multidomain-v1-r2/evaluation-v1-r2/arm-runtime/lineages/75cadc260b52f873a7893ea7c18a3bb027ed25d98fabb300942e429ac08df706/cognitive.sqlite3",
                "c27f14fce291334eb785cd02ca77560021a61546c5cabbc0c0db87738d5bd022",
                "SQLITE",
            ),
            "LINEAGE_E4ACB76A": (
                "/opt/angler/state/project-angler/high-level-multidomain-v1-r2/evaluation-v1-r2/arm-runtime/lineages/e4acb76aae39f554ee54ad98854eb67f83c7ffc7a0ab2961c7709d31bfb248de/cognitive.sqlite3",
                "748a7499667382c754fa389b15c76af57e20f29bacab25a30d51200f7205cfda",
                "SQLITE",
            ),
            "LINEAGE_F4F478B1": (
                "/opt/angler/state/project-angler/high-level-multidomain-v1-r2/evaluation-v1-r2/arm-runtime/lineages/f4f478b15524d0d88cb6363ae2484c05acfbc0c608d8ab1153cdebc01f324312/cognitive.sqlite3",
                "534191cc6e404da2880681074906f550a91c811e5d80df45442212eb834dc9fe",
                "SQLITE",
            ),
        }
        observed = {
            spec.label: (os.fspath(spec.path), spec.sha256, spec.kind)
            for spec in diagnostic.FROZEN_INPUTS
        }
        self.assertEqual(observed, expected)
        expected_modes = {
            "R2_RESULT": 0o600,
            "R2_RESULT_REPORT": 0o664,
            "R2_LEDGER": 0o600,
            "R2_ARM_FACTORY_AUDIT": 0o600,
            "R2_SOURCE_MANIFEST": 0o600,
            "R2_RUNNER": 0o664,
            "COGNITIVE_TRANSACTION_STORE_SOURCE": 0o664,
            "COGNITIVE_ACQUISITION_SOURCE": 0o664,
            "COGNITIVE_CONTRACTS_SOURCE": 0o664,
            "R2_PREDECESSOR_LEAF": 0o664,
            "DIAGNOSTIC_LEAF": 0o664,
            "LINEAGE_102E3196": 0o600,
            "LINEAGE_75CADC26": 0o600,
            "LINEAGE_E4ACB76A": 0o600,
            "LINEAGE_F4F478B1": 0o600,
        }
        self.assertEqual(
            {spec.label: spec.expected_mode for spec in diagnostic.FROZEN_INPUTS},
            expected_modes,
        )
        self.assertEqual(len(observed), len(diagnostic.FROZEN_INPUTS))
        self.assertEqual(len(observed), 15)
        self.assertEqual(
            len({spec.path for spec in diagnostic.FROZEN_INPUTS}),
            len(diagnostic.FROZEN_INPUTS),
        )

    def test_public_api_and_literal_live_output_are_frozen(self) -> None:
        expected = {
            "DiagnosticInvariantError",
            "DiagnosticResourceBudget",
            "DIAGNOSTIC_IDENTITY",
            "DIAGNOSTIC_SCHEMA",
            "OUTPUT_PATH",
            "MAX_OUTPUT_BYTES",
            "EXPECTED_ARMS",
            "FROZEN_INPUTS",
            "FrozenInputSpec",
            "FileWitness",
            "LiteralGrammar",
            "AttemptTrace",
            "ProvenanceRow",
            "canonical_json_bytes",
            "content_ref",
            "parse_canonical_json",
            "capture_frozen_input",
            "verify_frozen_input_unchanged",
            "immutable_sqlite_uri",
            "immutable_sqlite",
            "classify_declared_grammar",
            "classify_candidate",
            "equality_groups",
            "extract_attempt_trace",
            "first_divergence",
            "validate_unique_cohort",
            "rejoin_provenance",
            "validate_redacted_output",
            "publish_create_once",
            "run_live_diagnostic",
        }
        self.assertFalse(expected - set(dir(diagnostic)))
        self.assertEqual(
            diagnostic.OUTPUT_PATH,
            Path("/opt/angler/results/high-level-multidomain-v1-r2-trace-diagnosis.json"),
        )
        self.assertEqual(diagnostic.MAX_OUTPUT_BYTES, 1_048_576)
        self.assertEqual(
            tuple(inspect.signature(diagnostic.run_live_diagnostic).parameters),
            (),
        )

    def test_private_composer_inventory_rejects_same_label_substitution(self) -> None:
        substituted = list(diagnostic.FROZEN_INPUTS)
        substituted[0] = replace(
            substituted[0],
            path=Path("/tmp/substituted-r2-result.json"),
        )
        with self.assertRaises(diagnostic.DiagnosticInvariantError):
            diagnostic._spec_map(tuple(substituted))

    def test_source_has_no_model_network_subprocess_or_task_solver_import(self) -> None:
        source_path = ROOT / "experiments/runners/high_level_multidomain_v1_r2_trace_diagnostic.py"
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
        self.assertFalse(
            imported
            & {
                "accelerate",
                "cognee",
                "httpx",
                "requests",
                "socket",
                "subprocess",
                "torch",
                "transformers",
                "urllib",
            }
        )
        lowered = source.lower()
        for forbidden in (
            "auto_model",
            "generate(",
            "judge_response",
            "subprocess.",
            "socket.",
            "cuda",
            "shortest_path",
            "correct_answer",
            "response_repair",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, lowered)


if __name__ == "__main__":
    unittest.main()
