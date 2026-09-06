from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import struct
import tempfile
from threading import Event
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from angler.cognition.prospective import (
    CognitiveExecutionReceipt,
    CognitiveExecutionRequest,
)
from angler.reasoning import SituatedFeatureSpec, encode_situated_features
from angler.runtime.prospective_observation import (
    QwenReceiptObservedStateEncoderV1,
)
from angler.runtime.qwen_cognitive import (
    FrozenQwenCognitiveExecutorV1,
    FrozenQwenCycleManifestV1,
    FrozenQwenProcedureAdapterV1,
    FrozenQwenProcedureRelationAdapterV1,
    QwenExecutionJournalEntry,
    SQLiteQwenExecutionJournal,
    build_label_free_projection,
    parse_qwen_procedure_proposals,
)
from angler.runtime.situated_qwen import LocalQwenIO


def _ref(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


_GENESIS_CONFIG_REF = (
    "sha256:ec5df84ea8e3946ce9e2bc387ef874c5bd719d962ee0c9c08bd2ab6a279c05b7"
)
_PROSPECTIVE_CONFIG_REF = (
    "sha256:f3bab0f86a3671588a3bd66538448cf276e3a6e965d7d866f7698a092d1e7313"
)
_LEARNER_CHECKPOINT_REF = (
    "sha256:5b2de1bb50091570dd92a790fe179cf3681f1202a1660d843863c46845f14f4c"
)
_INITIAL_STATE_DIGEST = (
    "sha256:5a8bac6f60ba9d0d73c4cbc3965479a7f1b1f8d4fe4b3e9c9bb4530a924a4286"
)
_INITIAL_SNAPSHOT_SHA256 = (
    "93ab9deee7b0b67eabade8d43bb2b09e466f1dcd25405931d5a3d55eca91902d"
)


def _manifest(**changes) -> FrozenQwenCycleManifestV1:
    values = {
        "model_ref": _ref("fake-qwen-model"),
        "tokenizer_ref": _ref("fake-qwen-tokenizer"),
        "genesis_config_ref": _GENESIS_CONFIG_REF,
        "prospective_config_ref": _PROSPECTIVE_CONFIG_REF,
        "learner_checkpoint_ref": _LEARNER_CHECKPOINT_REF,
        "initial_competence_state_digest": _INITIAL_STATE_DIGEST,
        "genesis_snapshot_sha256": _INITIAL_SNAPSHOT_SHA256,
        "genesis_snapshot_bytes": 139_913,
        "genesis_seed": 2_026_083_101,
        "prospective_lesion": False,
        "input_width": 8,
        "relation_width": 4,
        "temporal_width": 8,
        "latent_width": 3,
        "max_input_tokens": 4_096,
        "max_new_tokens": 64,
        "embedding_batch_size": 1,
    }
    values.update(changes)
    return FrozenQwenCycleManifestV1(**values)


class _Batch(dict):
    def to(self, device):
        return _Batch({name: value.to(device) for name, value in self.items()})


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self.chat_prompts: list[str] = []

    def apply_chat_template(self, messages, **kwargs):
        if kwargs != {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }:
            raise AssertionError("unexpected chat-template settings")
        prompt = messages[0]["content"]
        self.chat_prompts.append(prompt)
        return prompt

    def __call__(
        self,
        texts,
        *,
        return_tensors,
        padding=False,
        add_special_tokens=False,
    ):
        if return_tensors != "pt":
            raise AssertionError("unexpected tensor format")
        rows = [texts] if isinstance(texts, str) else list(texts)
        encoded_rows = [
            [2, *(2 + (value % 250) for value in row.encode("utf-8"))]
            if add_special_tokens
            else [2 + (value % 250) for value in row.encode("utf-8")]
            for row in rows
        ]
        maximum = max(len(row) for row in encoded_rows)
        input_ids = torch.zeros((len(rows), maximum), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, values in enumerate(encoded_rows):
            input_ids[index, : len(values)] = torch.tensor(values)
            attention_mask[index, : len(values)] = 1
        return _Batch(input_ids=input_ids, attention_mask=attention_mask)

    def batch_decode(self, generated, **kwargs):
        if kwargs != {"skip_special_tokens": True}:
            raise AssertionError("unexpected decode settings")
        return [
            bytes(int(value) for value in row.detach().cpu().tolist()).decode("utf-8")
            for row in generated
        ]


class _Backbone(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.linspace(0.25, 1.0, width))

    def forward(self, input_ids, attention_mask, *, use_cache, return_dict):
        del attention_mask
        if use_cache or not return_dict:
            raise AssertionError("unexpected representation settings")
        positions = torch.arange(input_ids.shape[1]).view(1, -1, 1)
        hidden = (
            input_ids.float().unsqueeze(-1) + positions.float()
        ) * self.anchor.view(1, 1, -1)
        return SimpleNamespace(last_hidden_state=hidden)


class _Model(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.model = _Backbone(width)
        self.responses: list[str] = []
        self.generate_calls = 0

    def queue(self, *responses: str) -> None:
        self.responses.extend(responses)

    def generate(self, *, input_ids, attention_mask, **kwargs):
        del attention_mask
        if kwargs["do_sample"] or kwargs["max_new_tokens"] != 64:
            raise AssertionError("unexpected generation settings")
        self.generate_calls += 1
        if not self.responses:
            raise AssertionError("no fake response is queued")
        values = torch.tensor(
            [list(self.responses.pop(0).encode("utf-8"))],
            dtype=torch.long,
            device=input_ids.device,
        )
        return torch.cat((input_ids, values), dim=1)


def _io(model: _Model | None = None, tokenizer: _Tokenizer | None = None):
    manifest = _manifest()
    model = model or _Model()
    tokenizer = tokenizer or _Tokenizer()
    model.requires_grad_(False)
    model.eval()
    io = LocalQwenIO(
        model,
        tokenizer,
        embedding_batch_size=manifest.embedding_batch_size,
        max_input_tokens=manifest.max_input_tokens,
        max_new_tokens=manifest.max_new_tokens,
        enable_thinking=False,
        model_ref=manifest.model_ref,
        tokenizer_ref=manifest.tokenizer_ref,
    )
    return manifest, model, tokenizer, io


@dataclass(frozen=True, slots=True)
class _RecallItem:
    artifact_ref: str
    text: str
    acquired_ordinal: int
    age: int
    backend_score: float | None
    world_valid_at_query: bool | None
    landmark_relations: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class _RecallBatch:
    items: tuple[_RecallItem, ...]


def _recall() -> _RecallBatch:
    return _RecallBatch(
        (
            _RecallItem(_ref("record-b"), "evidence beta", 1, 2, 0.5, True),
            _RecallItem(_ref("record-a"), "evidence alpha", 2, 1, None, None),
        )
    )


def _request(label: str = "one") -> CognitiveExecutionRequest:
    reservation_ref = _ref("reservation-" + label)
    return CognitiveExecutionRequest(
        reservation_ref=reservation_ref,
        commitment_ref=_ref("commitment-" + label),
        idempotency_key=reservation_ref,
        task_id="qwen-fixture",
        request="Use the public task input.",
        selected_trace="Apply the learned-selected public procedure.",
        input_observations=("public observation",),
    )


class FrozenQwenManifestAndIOTests(unittest.TestCase):
    def test_manifest_retains_default_and_accepts_bounded_output_cap(self) -> None:
        self.assertEqual(_manifest().max_new_tokens, 64)
        self.assertEqual(_manifest(max_new_tokens=160).max_new_tokens, 160)
        self.assertEqual(_manifest(max_new_tokens=256).max_new_tokens, 256)
        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            _manifest(max_new_tokens=257)

    def test_manifest_binds_encoder_runtime_and_exact_learner_lineage(self) -> None:
        manifest = _manifest()
        for value in (
            manifest.encoder_ref,
            manifest.manifest_ref,
            manifest.knowledge_encoder_ref,
            manifest.projection_seed_ref,
            manifest.generation_config_ref,
        ):
            self.assertRegex(value, r"^sha256:[0-9a-f]{64}$")

        mutations = {
            "genesis_config_ref": _ref("changed-genesis"),
            "prospective_config_ref": _ref("changed-prospective"),
            "learner_checkpoint_ref": _ref("changed-checkpoint"),
            "initial_competence_state_digest": _ref("changed-state"),
            "genesis_snapshot_sha256": "0" * 64,
            "genesis_snapshot_bytes": 139_914,
            "genesis_seed": 2_026_083_102,
            "prospective_lesion": True,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                changed = replace(manifest, **{field: value})
                self.assertNotEqual(manifest.manifest_ref, changed.manifest_ref)

        changed_encoder = replace(manifest, relation_width=5)
        self.assertNotEqual(manifest.encoder_ref, changed_encoder.encoder_ref)
        self.assertNotEqual(manifest.manifest_ref, changed_encoder.manifest_ref)

    def test_exact_path_rejects_missing_or_drifted_identity(self) -> None:
        manifest = _manifest()
        model = _Model()
        model.requires_grad_(False)
        legacy = LocalQwenIO(model, _Tokenizer())
        with self.assertRaisesRegex(ValueError, "requires model and tokenizer"):
            FrozenQwenProcedureAdapterV1(legacy, manifest)

        _manifest_value, _model, _tokenizer, exact = _io()
        FrozenQwenProcedureAdapterV1(exact, manifest)
        with self.assertRaisesRegex(ValueError, "drifted"):
            FrozenQwenProcedureAdapterV1(
                exact,
                replace(manifest, model_ref=_ref("different-model")),
            )

    def test_generation_record_is_bounded_auditable_and_legacy_generate_survives(self) -> None:
        manifest, model, _tokenizer, io = _io()
        self.assertEqual(io.generation_calls, 0)
        model.queue("bounded response")
        record = io.generate_record("bounded prompt")
        self.assertEqual(io.generation_calls, 1)
        self.assertEqual(record.response, "bounded response")
        self.assertEqual(record.model_ref, manifest.model_ref)
        self.assertEqual(record.tokenizer_ref, manifest.tokenizer_ref)
        self.assertEqual(record.generation_config_ref, manifest.generation_config_ref)
        self.assertEqual(record.generated_token_ids, tuple(b"bounded response"))
        self.assertRegex(record.generation_ref, r"^sha256:[0-9a-f]{64}$")

        legacy_model = _Model()
        legacy_model.requires_grad_(False)
        legacy_model.queue("legacy response")
        legacy = LocalQwenIO(legacy_model, _Tokenizer(), max_new_tokens=64)
        self.assertEqual(legacy.generate("legacy prompt"), "legacy response")
        self.assertEqual(legacy.generation_calls, 1)

    def test_generation_call_counter_counts_actual_failed_invocation_and_is_read_only(
        self,
    ) -> None:
        _manifest_value, model, _tokenizer, io = _io()
        with self.assertRaisesRegex(ValueError, "non-empty"):
            io.generate_record(" ")
        self.assertEqual(io.generation_calls, 0)

        with self.assertRaisesRegex(AssertionError, "no fake response"):
            io.generate_record("invoke the model")
        self.assertEqual(model.generate_calls, 1)
        self.assertEqual(io.generation_calls, 1)

        model.queue("recovered response")
        self.assertEqual(io.generate("invoke it again"), "recovered response")
        self.assertEqual(model.generate_calls, 2)
        self.assertEqual(io.generation_calls, 2)
        with self.assertRaises(AttributeError):
            io.generation_calls = 0  # type: ignore[misc]

    def test_input_ceiling_rejects_without_truncation_or_generation(self) -> None:
        model = _Model()
        model.requires_grad_(False)
        tokenizer = _Tokenizer()
        io = LocalQwenIO(
            model,
            tokenizer,
            embedding_batch_size=1,
            max_input_tokens=8,
            max_new_tokens=64,
            model_ref=_ref("bounded-model"),
            tokenizer_ref=_ref("bounded-tokenizer"),
        )
        model.queue("must not run")
        with self.assertRaisesRegex(ValueError, "token ceiling"):
            io.generate_record("this prompt is too long")
        with self.assertRaisesRegex(ValueError, "token ceiling"):
            io.embed(("this segment is too long",))
        self.assertEqual(model.generate_calls, 0)
        self.assertEqual(io.generation_calls, 0)

        model.train()
        with self.assertRaisesRegex(RuntimeError, "evaluation mode"):
            io.generate_record("short")


class FrozenQwenProposalAndRelationTests(unittest.TestCase):
    def test_strict_json_supports_two_seven_and_sixty_four_without_repair(self) -> None:
        for count in (2, 7, 64):
            response = json.dumps(
                {"procedures": [f"Procedure {index}." for index in range(count)]},
                separators=(",", ":"),
            )
            self.assertEqual(len(parse_qwen_procedure_proposals(response, count=count)), count)

        invalid = (
            '```json\n{"procedures":["A","B"]}\n```',
            '{"procedures":["A","A"]}',
            '{"procedures":["A"]}',
            '{"procedures":[" A","B"]}',
            '{"procedures":["A","B"],"answer":"hidden"}',
        )
        for response in invalid:
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    parse_qwen_procedure_proposals(response, count=2)

    def test_qwen_supplies_every_public_proposal(self) -> None:
        manifest, model, tokenizer, io = _io()
        model.queue('{"procedures":["Inspect evidence.","Compare constraints."]}')
        adapter = FrozenQwenProcedureAdapterV1(io, manifest)
        proposals = adapter.propose_procedure_traces(
            "Handle the current public request.",
            ("public precedent",),
            count=2,
        )
        self.assertEqual(proposals, ("Inspect evidence.", "Compare constraints."))
        self.assertIn("Do not answer the task", tokenizer.chat_prompts[-1])
        self.assertEqual(
            adapter.last_proposal_generation.response,
            '{"procedures":["Inspect evidence.","Compare constraints."]}',
        )

    def test_malformed_proposal_retains_exact_consumed_generation_without_retry(self) -> None:
        manifest, model, _tokenizer, io = _io()
        raw = '{"procedures":["Only one procedure."]}'
        model.queue(raw)
        adapter = FrozenQwenProcedureAdapterV1(io, manifest)

        with self.assertRaisesRegex(ValueError, "cardinality"):
            adapter.propose_procedure_traces(
                "Handle the current public request.",
                ("public precedent",),
                count=2,
            )

        retained = adapter.last_proposal_generation
        self.assertIsNotNone(retained)
        self.assertEqual(retained.response, raw)
        self.assertEqual(io.generation_calls, 1)
        self.assertEqual(model.generate_calls, 1)

    def test_projection_counter_signs_and_candidate_rows_are_exactly_bounded(self) -> None:
        manifest, _model, _tokenizer, io = _io()
        projection = build_label_free_projection(manifest)
        self.assertEqual(projection.shape, (4, 8))
        seed = bytes.fromhex(manifest.projection_seed_ref[7:])
        scale = 1.0 / math.sqrt(8.0)
        independent = []
        for input_index in range(8):
            digest = hashlib.sha256(
                b"angler.qwen-label-free-projection.v1\0"
                + seed
                + struct.pack(">II", 0, input_index)
            ).digest()
            independent.append(scale if digest[0] & 1 else -scale)
        self.assertTrue(torch.equal(projection[0], torch.tensor(independent)))

        recall = _recall()
        adapter = FrozenQwenProcedureRelationAdapterV1(io, manifest)
        candidates = adapter.encode_candidates(
            "Relate this public task.",
            ("First public procedure.", "Second public procedure."),
            recall,
        )
        self.assertEqual(len(candidates), 2)
        temporal, mask = encode_situated_features(
            (recall,),
            now=(3,),
            spec=SituatedFeatureSpec((), include_acquired_ordinal=True),
        )
        expected_temporal = temporal[0, mask[0]].mean(dim=0)
        expected_support = tuple(sorted(item.artifact_ref for item in recall.items))
        for candidate in candidates:
            self.assertEqual(candidate.relation_features.shape, (4,))
            self.assertTrue(torch.isfinite(candidate.relation_features).all())
            self.assertFalse(candidate.relation_features.requires_grad)
            self.assertTrue(torch.equal(candidate.temporal_features, expected_temporal))
            self.assertEqual(float(candidate.base_logit), 0.0)
            self.assertEqual(candidate.supporting_evidence_refs, expected_support)

        inconsistent = _RecallBatch(
            (recall.items[0], replace(recall.items[1], age=2))
        )
        shifted = adapter.encode_candidates(
            "Relate this public task.",
            ("First public procedure.", "Second public procedure."),
            inconsistent,
        )
        self.assertFalse(
            torch.equal(
                candidates[0].temporal_features,
                shifted[0].temporal_features,
            )
        )


class FrozenQwenJournalExecutorAndObservationTests(unittest.TestCase):
    def test_populated_version_one_reopen_is_strictly_byte_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, model, _tokenizer, io = _io()
            journal_path = root / "qwen-journal.sqlite3"
            journal = SQLiteQwenExecutionJournal(journal_path)
            executor = FrozenQwenCognitiveExecutorV1(io, manifest, journal)
            request = _request("byte-stable-reopen")
            model.queue("durable byte-stable response")
            executor.execute(request)
            journal.audit_integrity()

            before = journal_path.read_bytes()
            before_sha256 = hashlib.sha256(before).hexdigest()
            before_inventory = tuple(sorted(item.name for item in root.iterdir()))
            reopened = SQLiteQwenExecutionJournal(journal_path)
            reopened.audit_integrity()
            self.assertEqual(
                reopened.get(request.idempotency_key),
                journal.get(request.idempotency_key),
            )
            after = journal_path.read_bytes()

            self.assertEqual(after, before)
            self.assertEqual(hashlib.sha256(after).hexdigest(), before_sha256)
            self.assertEqual(
                tuple(sorted(item.name for item in root.iterdir())),
                before_inventory,
            )

    def test_version_zero_journal_is_initialized_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal_path = root / "version-zero.sqlite3"
            journal_path.touch(mode=0o600)

            journal = SQLiteQwenExecutionJournal(journal_path)
            journal.audit_integrity()
            self.assertIsNone(journal.get(_ref("missing-version-zero-entry")))
            connection = sqlite3.connect(
                journal_path.as_uri() + "?mode=ro&immutable=1",
                uri=True,
            )
            try:
                self.assertEqual(
                    connection.execute("PRAGMA user_version").fetchone(),
                    (SQLiteQwenExecutionJournal.SCHEMA_VERSION,),
                )
            finally:
                connection.close()

    def test_existing_journal_reopen_observes_exclusive_schema_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal_path = root / "locked-reopen.sqlite3"
            journal = SQLiteQwenExecutionJournal(journal_path)
            journal.audit_integrity()
            before = journal_path.read_bytes()
            before_inventory = tuple(sorted(item.name for item in root.iterdir()))

            writer = sqlite3.connect(journal_path, isolation_level=None)
            started = Event()

            def reopen() -> SQLiteQwenExecutionJournal:
                started.set()
                return SQLiteQwenExecutionJournal(journal_path)

            try:
                writer.execute("BEGIN EXCLUSIVE")
                writer.execute("CREATE TABLE pending_schema_change (value TEXT)")
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pending = pool.submit(reopen)
                    self.assertTrue(started.wait(timeout=1.0))
                    with self.assertRaises(TimeoutError):
                        pending.result(timeout=0.1)
                    writer.rollback()
                    reopened = pending.result(timeout=2.0)
                    reopened.audit_integrity()
            finally:
                if writer.in_transaction:
                    writer.rollback()
                writer.close()

            self.assertEqual(journal_path.read_bytes(), before)
            self.assertEqual(
                tuple(sorted(item.name for item in root.iterdir())),
                before_inventory,
            )

    def test_schema_lookalikes_fail_without_mutation(self) -> None:
        valid_table = """
            CREATE TABLE qwen_execution_journal (
                idempotency_key TEXT PRIMARY KEY NOT NULL,
                manifest_ref TEXT NOT NULL,
                execution_request_ref TEXT UNIQUE NOT NULL,
                execution_receipt_ref TEXT UNIQUE NOT NULL,
                entry_bytes BLOB NOT NULL
            ) WITHOUT ROWID
        """
        wrong_table = valid_table.replace(
            "qwen_execution_journal", "wrong_qwen_execution_journal", 1
        )
        wrong_constraints = """
            CREATE TABLE qwen_execution_journal (
                idempotency_key TEXT PRIMARY KEY NOT NULL,
                manifest_ref TEXT NOT NULL,
                execution_request_ref TEXT NOT NULL,
                execution_receipt_ref TEXT NOT NULL,
                entry_bytes BLOB NOT NULL
            ) WITHOUT ROWID
        """
        rowid_table = valid_table.replace(") WITHOUT ROWID", ")")
        cases = (
            ("unsupported-version", valid_table, 2, ()),
            ("wrong-table", wrong_table, 1, ()),
            ("wrong-constraints", wrong_constraints, 1, ()),
            ("wrong-storage-shape", rowid_table, 1, ()),
            (
                "extra-index",
                valid_table,
                1,
                (
                    "CREATE INDEX unexpected_qwen_index "
                    "ON qwen_execution_journal(manifest_ref)",
                ),
            ),
            (
                "extra-trigger",
                valid_table,
                1,
                (
                    "CREATE TRIGGER unexpected_qwen_trigger "
                    "AFTER INSERT ON qwen_execution_journal BEGIN SELECT 1; END",
                ),
            ),
            (
                "extra-view",
                valid_table,
                1,
                (
                    "CREATE VIEW unexpected_qwen_view AS "
                    "SELECT idempotency_key FROM qwen_execution_journal",
                ),
            ),
            ("malformed-bytes", None, None, ()),
        )
        for label, table_sql, version, extras in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                journal_path = root / "lookalike.sqlite3"
                if table_sql is None:
                    journal_path.write_bytes(b"not a SQLite database")
                else:
                    connection = sqlite3.connect(journal_path)
                    try:
                        connection.execute(table_sql)
                        for statement in extras:
                            connection.execute(statement)
                        connection.execute(f"PRAGMA user_version={version}")
                        connection.commit()
                    finally:
                        connection.close()

                before = journal_path.read_bytes()
                before_sha256 = hashlib.sha256(before).hexdigest()
                before_inventory = tuple(
                    sorted(item.name for item in root.iterdir())
                )
                with self.assertRaises((ValueError, sqlite3.DatabaseError)):
                    SQLiteQwenExecutionJournal(journal_path)
                after = journal_path.read_bytes()
                self.assertEqual(after, before)
                self.assertEqual(hashlib.sha256(after).hexdigest(), before_sha256)
                self.assertEqual(
                    tuple(sorted(item.name for item in root.iterdir())),
                    before_inventory,
                )

    def test_selected_trace_is_the_only_procedure_and_recovery_is_byte_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, model, tokenizer, io = _io()
            journal_path = Path(directory) / "qwen-journal.sqlite3"
            journal = SQLiteQwenExecutionJournal(journal_path)
            executor = FrozenQwenCognitiveExecutorV1(io, manifest, journal)
            request = _request()
            model.queue("exact public Qwen response")

            executed = executor.execute(request)
            self.assertEqual(executed.executed_trace, request.selected_trace)
            self.assertEqual(executed.response, "exact public Qwen response")
            prompt = tokenizer.chat_prompts[-1]
            self.assertIn(request.selected_trace, prompt)
            self.assertNotIn("unselected hidden alternative", prompt)
            durable = journal.get(request.idempotency_key)
            self.assertIsNotNone(durable)
            self.assertEqual(durable.receipt.response, executed.response)

            restarted_journal = SQLiteQwenExecutionJournal(journal_path)
            restarted = FrozenQwenCognitiveExecutorV1(io, manifest, restarted_journal)
            recovered = restarted.recover(request)
            self.assertEqual(recovered.status, "RECORDED")
            self.assertEqual(recovered.receipt, durable.receipt)
            self.assertEqual(
                recovered.receipt.canonical_bytes(),
                durable.receipt.canonical_bytes(),
            )
            self.assertEqual(model.generate_calls, 1)

            tampered_request = _request("tampered-prompt")
            tampered_generation = replace(
                durable.generation,
                prompt_ref=_ref("wrong-execution-prompt"),
            )
            tampered_receipt = CognitiveExecutionReceipt.from_request(
                tampered_request,
                status="COMPLETED",
                executed_trace=tampered_request.selected_trace,
                response=tampered_generation.response,
            )
            restarted_journal.put_if_absent(
                QwenExecutionJournalEntry(
                    manifest.manifest_ref,
                    tampered_request,
                    tampered_generation,
                    tampered_receipt,
                )
            )
            with self.assertRaisesRegex(ValueError, "exact context"):
                restarted.recover(tampered_request)
            journal.audit_integrity()

    def test_journal_rejects_conflict_and_concurrent_identity_race(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, model, _tokenizer, io = _io()
            journal = SQLiteQwenExecutionJournal(
                Path(directory) / "qwen-journal.sqlite3"
            )
            self.assertIsInstance(
                SQLiteQwenExecutionJournal(
                    str(Path(directory) / "qwen-string-path.sqlite3")
                ),
                SQLiteQwenExecutionJournal,
            )
            request = _request("conflict")
            model.queue("first response")
            executor = FrozenQwenCognitiveExecutorV1(io, manifest, journal)
            executor.execute(request)
            first = journal.get(request.idempotency_key)

            different_generation = replace(
                first.generation,
                response="different response",
            )
            different_receipt = CognitiveExecutionReceipt.from_request(
                request,
                status="COMPLETED",
                executed_trace=request.selected_trace,
                response="different response",
            )
            conflicting = QwenExecutionJournalEntry(
                manifest.manifest_ref,
                request,
                different_generation,
                different_receipt,
            )
            with self.assertRaisesRegex(ValueError, "identity conflict"):
                journal.put_if_absent(conflicting)

            race_request = _request("race")
            race_entries = []
            for response in ("race alpha", "race beta"):
                generation = replace(
                    first.generation,
                    prompt_ref=_ref("prompt-" + response),
                    response=response,
                    generated_token_ids=tuple(response.encode("utf-8")),
                )
                receipt = CognitiveExecutionReceipt.from_request(
                    race_request,
                    status="COMPLETED",
                    executed_trace=race_request.selected_trace,
                    response=response,
                )
                race_entries.append(
                    QwenExecutionJournalEntry(
                        manifest.manifest_ref,
                        race_request,
                        generation,
                        receipt,
                    )
                )

            def put(entry):
                try:
                    return journal.put_if_absent(entry)
                except ValueError as error:
                    return error

            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = tuple(pool.map(put, race_entries))
            self.assertEqual(
                sum(isinstance(value, QwenExecutionJournalEntry) for value in outcomes),
                1,
            )
            self.assertEqual(sum(isinstance(value, ValueError) for value in outcomes), 1)
            journal.audit_integrity()

    def test_manifest_bound_receipt_observation_is_exact_and_restart_stable(self) -> None:
        manifest = _manifest()
        request = _request("observation")
        receipt = CognitiveExecutionReceipt.from_request(
            request,
            status="COMPLETED",
            executed_trace=request.selected_trace,
            response="observed Qwen response",
        )
        encoder = QwenReceiptObservedStateEncoderV1(
            model_ref=manifest.model_ref,
            encoder_ref=manifest.encoder_ref,
            manifest_ref=manifest.manifest_ref,
            latent_width=manifest.latent_width,
        )
        first = encoder.encode(request, receipt, latent_width=manifest.latent_width)
        second = encoder.encode(request, receipt, latent_width=manifest.latent_width)
        self.assertEqual(first, second)
        self.assertEqual(
            first.evidence_refs,
            tuple(sorted((request.execution_request_ref, receipt.execution_receipt_ref))),
        )

        manifest_bytes = manifest.manifest_ref.encode("ascii")
        framed = b"".join(
            (
                b"angler.qwen-receipt-observed-state.v1\0",
                struct.pack(">Q", len(manifest_bytes)),
                manifest_bytes,
                struct.pack(">Q", manifest.latent_width),
                struct.pack(">Q", len(request.canonical_bytes())),
                request.canonical_bytes(),
                struct.pack(">Q", len(receipt.canonical_bytes())),
                receipt.canonical_bytes(),
            )
        )
        independent = []
        for index in range(manifest.latent_width):
            leading = int.from_bytes(
                hashlib.sha256(framed + struct.pack(">Q", index)).digest()[:8],
                "big",
            )
            independent.append(((leading >> 11) / 2**52) - 1.0)
        self.assertEqual(first.latent, tuple(independent))

        changed_manifest = replace(manifest, genesis_seed=manifest.genesis_seed + 1)
        changed_encoder = QwenReceiptObservedStateEncoderV1(
            model_ref=changed_manifest.model_ref,
            encoder_ref=changed_manifest.encoder_ref,
            manifest_ref=changed_manifest.manifest_ref,
            latent_width=changed_manifest.latent_width,
        )
        self.assertNotEqual(
            first.latent,
            changed_encoder.encode(
                request, receipt, latent_width=changed_manifest.latent_width
            ).latent,
        )


if __name__ == "__main__":
    unittest.main()
