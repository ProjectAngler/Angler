from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest

from angler.runtime.jenny_genesis import JennyGenesis
from angler.runtime.persistent_autonomy import (
    Affordance,
    AffordanceRequest,
    AffordanceReceipt,
    AutonomyHeartbeat,
    CHOICE_CONTEXT_MAX_CHARACTERS,
    CONSEQUENCE_NAMES,
    CycleChoice,
    CycleObservation,
    CycleUpdate,
    DynamicAffordanceRegistry,
    ObservableConsequence,
    PermissionGate,
    PersistentAutonomySupervisor,
    RankedAffordance,
)
from angler.runtime.temporal_v2 import TrustedClock


QUALIFICATION = "sha256:" + "7" * 64


class _ClockSource:
    def __init__(self) -> None:
        self.wall = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
        self.mono = 20_000_000_000

    def wall_now(self):
        value = self.wall
        self.wall += timedelta(milliseconds=1)
        return value

    def mono_now(self):
        value = self.mono
        self.mono += 1_000_000
        return value


def _clock(source=None):
    source = source or _ClockSource()
    return TrustedClock(wall_clock=source.wall_now, monotonic_clock=source.mono_now)


def _genesis(created="2026-09-02T12:00:00Z"):
    return JennyGenesis.owner_approved(created_at_utc=created)


def _state(values=None):
    return json.dumps(
        {"utility": values or {}}, separators=(",", ":"), sort_keys=True
    ).encode()


class _LearnedCycle:
    def __init__(self, *, qualified=True, forced=()):
        self._qualification = QUALIFICATION if qualified else None
        self.forced = list(forced)
        self.seen_rankings = []
        self.learn_calls = 0

    @property
    def qualification_ref(self):
        return self._qualification

    def choose(self, *, observation, temporal, affordances, state):
        values = json.loads(state)["utility"]
        rankings = tuple(
            RankedAffordance(item.affordance_id, float(values.get(item.affordance_id, 0.0)))
            for item in sorted(
                affordances,
                key=lambda item: (-float(values.get(item.affordance_id, 0.0)), item.affordance_id),
            )
        )
        selected = self.forced.pop(0) if self.forced else rankings[0].affordance_id
        self.seen_rankings.append(rankings)
        return CycleChoice(selected, rankings, "Outcome-sensitive synthetic prediction.", 0.25)

    def learn(self, *, observation, choice, receipt, temporal, parent_state):
        self.learn_calls += 1
        values = json.loads(parent_state)["utility"]
        consequence = dict(receipt.consequence)
        values[choice.selected_affordance_id] = float(
            values.get(choice.selected_affordance_id, 0.0)
        ) + float(consequence.get("reward", 0.0))
        return CycleUpdate(
            _state(values),
            "The observed consequence changed the selected affordance estimate.",
            "Proposal: retain the outcome-linked utility update.",
        )


class _CheckpointedInitiativeCycle(_LearnedCycle):
    def __init__(self, *, proposals=(), fail_proposals=0):
        super().__init__()
        self.proposals = list(proposals)
        self.fail_proposals = fail_proposals
        self.readiness_calls = 0
        self.proposal_calls = 0

    def has_autonomous_intent(self, *, state):
        self.readiness_calls += 1
        return False

    def propose_autonomous_request(self, *, state, temporal, affordances):
        self.proposal_calls += 1
        if self.fail_proposals:
            self.fail_proposals -= 1
            raise RuntimeError("injected initiative failure")
        return self.proposals.pop(0) if self.proposals else None


class _CounterExecutor:
    def __init__(self, reward=1.0):
        self.calls = 0
        self.reward = reward

    def __call__(self, request):
        self.calls += 1
        return AffordanceReceipt(
            "COMPLETED", f"internal effect {self.calls}", (("reward", self.reward),)
        )


class _UnevaluatedExecutor:
    def __init__(self):
        self.calls = 0

    def __call__(self, request):
        self.calls += 1
        return AffordanceReceipt("COMPLETED_UNEVALUATED", "candidate output", ())


class _FailingExecutor:
    def __call__(self, request):
        raise ValueError(f"invalid payload for {request.affordance_id}")


def _registry(*items):
    registry = DynamicAffordanceRegistry()
    for affordance, executor in items:
        registry.register(affordance, executor)
    return registry


def _act(name):
    return Affordance(name, "ACT", f"Internal {name} operation.", "internal.cognition")


def _wait(name="wait.context"):
    return Affordance(name, "WAIT", "Wait for a meaningful state change.", "internal.cognition")


def _deferred_registry(*, mapper=None, mapper_ref=None, source_ref=None):
    registry = DynamicAffordanceRegistry()
    registry.register(
        _act("tool.research"),
        execution_mode="DEFERRED",
        observable_source_ref=source_ref,
        observable_source_kind="TOOL" if source_ref is not None else None,
        consequence_mapper=mapper,
        consequence_mapper_ref=mapper_ref,
    )
    return registry


def _host_guarded_affordance():
    return Affordance(
        "web.research",
        "ACT",
        "Request bounded research from a separately guarded host.",
        "external.web.read",
        external_effect=True,
        authorization_mode="HOST_GUARDED",
    )


class PersistentAutonomyTests(unittest.TestCase):
    def test_choice_and_reserved_request_share_an_aggregate_context_bound(self) -> None:
        context_json = json.dumps(
            {"formation": "x" * 70_000},
            separators=(",", ":"),
            sort_keys=True,
        )
        choice = CycleChoice(
            "reflect.internal",
            (RankedAffordance("reflect.internal", 1.0),),
            "Preserve one normalized transaction envelope.",
            0.1,
            context_json=context_json,
        )
        request = AffordanceRequest(
            idempotency_key="sha256:" + "1" * 64,
            trigger_ref="scheduler:aggregate-context",
            affordance_id="reflect.internal",
            observation_ref="sha256:" + "2" * 64,
            state_head_ref="sha256:" + "3" * 64,
            choice_context_json=context_json,
        )
        self.assertEqual(request.choice_context_json, choice.context_json)
        self.assertGreater(len(context_json), 65_536)

        oversized = json.dumps(
            {"formation": "x" * CHOICE_CONTEXT_MAX_CHARACTERS},
            separators=(",", ":"),
            sort_keys=True,
        )
        with self.assertRaisesRegex(ValueError, "context_json must be bounded"):
            CycleChoice(
                "reflect.internal",
                (RankedAffordance("reflect.internal", 1.0),),
                "Reject an actually oversized aggregate envelope.",
                0.1,
                context_json=oversized,
            )
        with self.assertRaisesRegex(
            ValueError, "choice_context_json must be bounded"
        ):
            AffordanceRequest(
                idempotency_key="sha256:" + "4" * 64,
                trigger_ref="scheduler:oversized-context",
                affordance_id="reflect.internal",
                observation_ref="sha256:" + "5" * 64,
                state_head_ref="sha256:" + "6" * 64,
                choice_context_json=oversized,
            )

    def test_event_substrate_commits_once_and_survives_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "substrate.sqlite3"
            registry = _registry((_act("reflect.internal"), _CounterExecutor()))
            first = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=_LearnedCycle(),
                affordances=registry,
            )
            result = first.human_ingress("human:substrate", "private input")
            duplicate = first.human_ingress("human:substrate", "private input")
            samples = first.event_substrate_items()

            self.assertEqual(duplicate.episode_ref, result.episode_ref)
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].episode_ref, result.episode_ref)
            payload = json.loads(samples[0].payload_json)
            self.assertEqual(payload["observation"]["source"], "HUMAN")
            self.assertEqual(payload["closure"]["receipt_status"], "COMPLETED")
            self.assertNotIn("private input", samples[0].payload_json)

            restarted = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=b"ignored-existing-state",
                clock=_clock(),
                cycle=_LearnedCycle(),
                affordances=registry,
            )
            self.assertEqual(restarted.event_substrate_items(), samples)

    def test_schema_v1_episode_history_backfills_exact_event_substrate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "substrate-migration.sqlite3"
            registry = _registry((_act("reflect.internal"), _CounterExecutor()))
            first = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=_LearnedCycle(),
                affordances=registry,
            )
            first.human_ingress("human:before-migration", "preserved event")
            expected = first.event_substrate_items()
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("DROP TABLE affective_substrate")
                connection.execute("PRAGMA user_version = 1")
                connection.commit()

            migrated = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=b"ignored-existing-state",
                clock=_clock(),
                cycle=_LearnedCycle(),
                affordances=registry,
            )
            self.assertEqual(migrated.event_substrate_items(), expected)
            with closing(sqlite3.connect(path)) as connection:
                version = connection.execute("PRAGMA user_version").fetchone()[0]
            self.assertEqual(version, 2)

    def test_synchronous_executor_error_commits_and_does_not_strand_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "executor-error.sqlite3",
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=_LearnedCycle(),
                affordances=_registry((_act("tool.fail"), _FailingExecutor())),
            )

            result = supervisor.human_ingress("human:first", "malformed action")
            self.assertEqual(result.status, "COMMITTED")
            self.assertIsNone(supervisor.pending_bytes())
            episode = json.loads(supervisor.episode_item(result.episode_ref).payload_json)
            self.assertEqual(episode["receipt"]["status"], "ERROR")
            self.assertIn("ValueError", episode["receipt"]["output"])

            second = supervisor.human_ingress("human:second", "try again")
            self.assertEqual(second.status, "COMMITTED")
            self.assertIsNone(supervisor.pending_bytes())

    def test_restart_reconciles_stranded_synchronous_reservation_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "interrupted-sync.sqlite3"
            registry = _registry((_act("tool.read"), _CounterExecutor()))
            first = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=_LearnedCycle(),
                affordances=registry,
            )
            observation = CycleObservation("human:interrupted", "HUMAN", "read")
            temporal = first.clock.sample(-1)
            choice = first.cycle.choose(
                observation=observation,
                temporal=temporal,
                affordances=first.affordances.definitions(),
                state=first.state_bytes(),
            )
            affordance = first._validate_choice(choice)  # noqa: SLF001
            first._reserve_effect(  # noqa: SLF001
                observation=observation,
                temporal=temporal,
                choice=choice,
                affordance=affordance,
            )

            restarted = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=_LearnedCycle(),
                affordances=registry,
            )
            result = restarted.reconcile_interrupted_synchronous()
            self.assertIsNotNone(result)
            self.assertEqual(result.status, "COMMITTED")
            self.assertIsNone(restarted.pending_bytes())
            self.assertIsNone(restarted.reconcile_interrupted_synchronous())

    def test_local_authorization_default_and_pre_field_pending_compatibility(self) -> None:
        implicit = _act("tool.research")
        explicit = Affordance(
            "tool.research",
            "ACT",
            "Internal tool.research operation.",
            "internal.cognition",
            authorization_mode="LOCAL",
        )
        self.assertEqual(implicit, explicit)
        self.assertEqual(implicit.authorization_mode, "LOCAL")
        self.assertTrue(PermissionGate().authorize(implicit).allowed)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pre-authorization-mode.sqlite3"
            registry = _deferred_registry()
            first = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=_LearnedCycle(),
                affordances=registry,
            )
            first.scheduler_tick("tool:legacy-pending")
            request = first.pending_effect_request()
            pending = json.loads(first.pending_bytes())
            pending["affordance"].pop("authorization_mode")
            legacy_bytes = json.dumps(
                pending,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            with closing(sqlite3.connect(path)) as connection:
                connection.execute(
                    "UPDATE supervisor_state SET pending_json=? WHERE singleton=1",
                    (legacy_bytes,),
                )
                connection.commit()

            restarted = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=_LearnedCycle(),
                affordances=registry,
            )
            self.assertEqual(restarted.resume_pending().status, "TOOL_PENDING")
            self.assertEqual(restarted.pending_effect_request(), request)

    def test_host_guarded_registration_and_allowlist_are_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "must declare an external effect"):
            Affordance(
                "web.invalid",
                "ACT",
                "Invalid host-guarded declaration.",
                "external.web.read",
                authorization_mode="HOST_GUARDED",
            )
        guarded = _host_guarded_affordance()
        invalid_registry = DynamicAffordanceRegistry()
        with self.assertRaisesRegex(ValueError, "require deferred execution"):
            invalid_registry.register(guarded, _CounterExecutor())

        with tempfile.TemporaryDirectory() as directory:
            registry = DynamicAffordanceRegistry()
            registry.register(guarded, execution_mode="DEFERRED")
            cycle = _LearnedCycle()
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "host-guarded-denied.sqlite3",
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=cycle,
                affordances=registry,
            )
            denied = supervisor.scheduler_tick("web:not-allowlisted")
            self.assertEqual(denied.status, "COMMITTED")
            self.assertEqual(cycle.learn_calls, 1)
            self.assertIsNone(supervisor.pending_effect_request())
            denied_episode = json.loads(
                supervisor.episode_item(denied.episode_ref).payload_json
            )
            self.assertEqual(denied_episode["receipt"]["status"], "DENIED")

        with tempfile.TemporaryDirectory() as directory:
            registry = DynamicAffordanceRegistry()
            registry.register(guarded, execution_mode="DEFERRED")
            cycle = _LearnedCycle()
            gate = PermissionGate(
                allowed_host_guarded_scopes=("external.web.read",)
            )
            self.assertFalse(gate.external_effects_enabled)
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "host-guarded-allowed.sqlite3",
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=cycle,
                affordances=registry,
                permission_gate=gate,
            )
            allowed = supervisor.scheduler_tick("web:allowlisted")
            self.assertEqual(allowed.status, "TOOL_PENDING")
            self.assertEqual(supervisor.pending_deferred_effect()[0], guarded)
            self.assertEqual(supervisor.state_head().moving_origin_ordinal, -1)
            self.assertEqual(cycle.learn_calls, 0)

    def test_local_external_affordance_remains_denied_by_default(self) -> None:
        local_external = Affordance(
            "network.local-mode",
            "ACT",
            "Local authorization does not bypass the external-effect gate.",
            "external.network",
            external_effect=True,
        )
        decision = PermissionGate().authorize(local_external)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "external effects are disabled")

    def test_deferred_act_reserves_without_execution_learning_or_time_advance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cycle = _LearnedCycle()
            registry = _deferred_registry()
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "deferred.sqlite3",
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=cycle,
                affordances=registry,
            )

            result = supervisor.scheduler_tick("tool:reserve")

            self.assertEqual(result.status, "TOOL_PENDING")
            self.assertEqual(result.moving_origin_ordinal, -1)
            self.assertEqual(supervisor.state_head().moving_origin_ordinal, -1)
            self.assertEqual(cycle.learn_calls, 0)
            affordance, request = supervisor.pending_deferred_effect()
            self.assertEqual(affordance, registry.definition("tool.research"))
            self.assertEqual(request, supervisor.pending_effect_request())
            pending = json.loads(supervisor.pending_bytes())
            self.assertEqual(pending["phase"], "RESERVED")
            self.assertEqual(pending["execution_mode"], "DEFERRED")
            self.assertEqual(
                result.pending_ref,
                "sha256:" + hashlib.sha256(supervisor.pending_bytes()).hexdigest(),
            )

    def test_deferred_reservation_restores_exactly_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deferred-restart.sqlite3"
            registry = _deferred_registry()
            first = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=_LearnedCycle(),
                affordances=registry,
            )
            reserved = first.agent_ingress(
                "agent:research", "Find grounded information for the owner's goal."
            )
            request = first.pending_effect_request()
            pending = first.pending_bytes()

            restarted = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=b"ignored-existing-state",
                clock=_clock(),
                cycle=_LearnedCycle(),
                affordances=registry,
            )
            restored = restarted.resume_pending()
            self.assertEqual(restored.status, "TOOL_PENDING")
            self.assertEqual(restored.pending_ref, reserved.pending_ref)
            self.assertEqual(restarted.pending_bytes(), pending)
            self.assertEqual(restarted.pending_effect_request(), request)
            self.assertEqual(restarted.state_head().moving_origin_ordinal, -1)

    def test_deferred_completion_rejects_request_mode_and_binding_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deferred-mismatch.sqlite3"
            source_ref = "sha256:" + "8" * 64
            mapper_ref = "sha256:" + "9" * 64

            def mapper(observed):
                return tuple((name, 0.0) for name in CONSEQUENCE_NAMES)

            registry = _deferred_registry(
                mapper=mapper,
                mapper_ref=mapper_ref,
                source_ref=source_ref,
            )
            supervisor = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=_LearnedCycle(),
                affordances=registry,
            )
            supervisor.scheduler_tick("tool:mismatch")
            request = supervisor.pending_effect_request()
            raw_receipt = AffordanceReceipt(
                "COMPLETED",
                "Observed result.",
                (),
                ObservableConsequence(
                    request_ref=request.idempotency_key,
                    source_kind="TOOL",
                    source_ref=source_ref,
                    observation_json='{"answer":"grounded"}',
                ),
            )
            with self.assertRaisesRegex(RuntimeError, "exact reservation"):
                supervisor.complete_deferred_effect(
                    replace(request, action_payload="tampered"), raw_receipt
                )

            changed = _deferred_registry(
                mapper=mapper,
                mapper_ref="sha256:" + "6" * 64,
                source_ref=source_ref,
            )
            changed_supervisor = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=_LearnedCycle(),
                affordances=changed,
            )
            with self.assertRaisesRegex(RuntimeError, "binding changed"):
                changed_supervisor.complete_deferred_effect(request, raw_receipt)

        with tempfile.TemporaryDirectory() as directory:
            affordance = _act("reflect.internal")
            registry = _registry((affordance, _CounterExecutor()))
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "sync.sqlite3",
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=_LearnedCycle(),
                affordances=registry,
            )
            observation = CycleObservation("sync:reserved", "SCHEDULER", "")
            request, _ = supervisor._reserve_effect(  # noqa: SLF001
                observation=observation,
                temporal=supervisor.clock.sample(-1),
                choice=CycleChoice(
                    affordance.affordance_id,
                    (RankedAffordance(affordance.affordance_id, 1.0),),
                    "Use the synchronous affordance.",
                    0.0,
                ),
                affordance=affordance,
            )
            with self.assertRaisesRegex(RuntimeError, "synchronous reservation"):
                supervisor.resume_pending()
            with self.assertRaisesRegex(RuntimeError, "synchronous reservation"):
                supervisor.complete_deferred_effect(
                    request, AffordanceReceipt("COMPLETED", "wrong route", ())
                )

    def test_deferred_completion_maps_once_commits_once_and_replays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_ref = "sha256:" + "8" * 64
            mapper_ref = "sha256:" + "9" * 64
            mapper_calls = []

            def mapper(observed):
                mapper_calls.append(observed.observation_ref)
                return tuple(
                    (name, 1.0 if name == "objective_progress" else 0.0)
                    for name in CONSEQUENCE_NAMES
                )

            cycle = _LearnedCycle()
            registry = _deferred_registry(
                mapper=mapper,
                mapper_ref=mapper_ref,
                source_ref=source_ref,
            )
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "deferred-complete.sqlite3",
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=cycle,
                affordances=registry,
            )
            supervisor.continuation_ingress(
                "continuation:research", "Continue the unresolved research operation."
            )
            request = supervisor.pending_effect_request()
            raw_receipt = AffordanceReceipt(
                "COMPLETED",
                "One grounded observation.",
                (),
                ObservableConsequence(
                    request_ref=request.idempotency_key,
                    source_kind="TOOL",
                    source_ref=source_ref,
                    observation_json='{"answer":"grounded","citations":2}',
                ),
            )

            committed = supervisor.complete_deferred_effect(request, raw_receipt)
            self.assertEqual(committed.status, "COMMITTED")
            self.assertEqual(committed.moving_origin_ordinal, 0)
            self.assertEqual(cycle.learn_calls, 1)
            self.assertEqual(len(mapper_calls), 1)
            self.assertIsNone(supervisor.pending_effect_request())
            episode = json.loads(supervisor.episode_item(committed.episode_ref).payload_json)
            self.assertEqual(
                tuple(name for name, _ in episode["receipt"]["consequence"]),
                CONSEQUENCE_NAMES,
            )

            replay = supervisor.complete_deferred_effect(request, raw_receipt)
            self.assertEqual(replay.episode_ref, committed.episode_ref)
            self.assertEqual(cycle.learn_calls, 1)
            self.assertEqual(len(mapper_calls), 1)
            with self.assertRaisesRegex(RuntimeError, "conflicts"):
                supervisor.complete_deferred_effect(
                    request, replace(raw_receipt, output="Different observation.")
                )

    def test_agent_and_continuation_sources_do_not_change_human_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = _registry((_act("reflect.internal"), _CounterExecutor()))
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "observation-sources.sqlite3",
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=_LearnedCycle(),
                affordances=registry,
            )
            results = (
                supervisor.human_ingress("human:source", "Ordinary conversation."),
                supervisor.agent_ingress("agent:source", "A delegated non-chat goal."),
                supervisor.continuation_ingress(
                    "continuation:source", "A model-authored continuation."
                ),
            )
            sources = tuple(
                json.loads(supervisor.episode_item(result.episode_ref).payload_json)[
                    "observation"
                ]["source"]
                for result in results
            )
            self.assertEqual(sources, ("HUMAN", "AGENT", "CONTINUATION"))

    def test_receipt_must_match_the_full_reserved_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            affordance = _act("reflect.internal")
            registry = _registry((affordance, _CounterExecutor()))
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "exact-request.sqlite3",
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=_LearnedCycle(),
                affordances=registry,
            )
            observation = CycleObservation("test:exact-request", "SCHEDULER", "")
            temporal = supervisor.clock.sample(-1)
            choice = CycleChoice(
                "reflect.internal",
                (RankedAffordance("reflect.internal", 1.0),),
                "Try one bounded operation.",
                0.1,
                action_payload="original payload",
            )
            request, _ = supervisor._reserve_effect(  # noqa: SLF001
                observation=observation,
                temporal=temporal,
                choice=choice,
                affordance=affordance,
            )
            tampered = replace(request, action_payload="different payload")
            with self.assertRaisesRegex(RuntimeError, "exact reservation"):
                supervisor._persist_receipt(  # noqa: SLF001
                    tampered,
                    AffordanceReceipt("COMPLETED", "candidate", (("reward", 1.0),)),
                )
            self.assertEqual(supervisor.state_head().moving_origin_ordinal, -1)

    def test_source_bound_executor_cannot_assign_its_own_scalar_credit(self) -> None:
        source_ref = "sha256:" + "8" * 64

        def self_scoring_executor(request):
            return AffordanceReceipt(
                "COMPLETED",
                "The observer claims success.",
                (("reward", 1.0),),
                ObservableConsequence(
                    request_ref=request.idempotency_key,
                    source_kind="TEST",
                    source_ref=source_ref,
                    observation_json='{"exit_code":0}',
                ),
            )

        with tempfile.TemporaryDirectory() as directory:
            cycle = _LearnedCycle()
            registry = DynamicAffordanceRegistry()
            registry.register(
                _act("test.observe"),
                self_scoring_executor,
                observable_source_ref=source_ref,
                observable_source_kind="TEST",
            )
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "self-scoring.sqlite3",
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=cycle,
                affordances=registry,
            )
            with self.assertRaisesRegex(ValueError, "cannot supply scalar"):
                supervisor.scheduler_tick("test:self-scoring")
            self.assertEqual(cycle.learn_calls, 0)
            self.assertEqual(supervisor.state_head().moving_origin_ordinal, -1)

    def test_source_bound_observation_must_match_request_source_and_kind(self) -> None:
        source_ref = "sha256:" + "8" * 64
        mismatches = (
            ("request", "sha256:" + "0" * 64, source_ref, "TEST"),
            ("source", None, "sha256:" + "1" * 64, "TEST"),
            ("kind", None, source_ref, "WORLD"),
        )
        for label, wrong_request, returned_source, returned_kind in mismatches:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                def executor(request):
                    return AffordanceReceipt(
                        "COMPLETED",
                        "Raw bounded observation.",
                        (),
                        ObservableConsequence(
                            request_ref=wrong_request or request.idempotency_key,
                            source_kind=returned_kind,
                            source_ref=returned_source,
                            observation_json='{"observed":true}',
                        ),
                    )

                cycle = _LearnedCycle()
                registry = DynamicAffordanceRegistry()
                registry.register(
                    _act("test.observe"),
                    executor,
                    observable_source_ref=source_ref,
                    observable_source_kind="TEST",
                )
                supervisor = PersistentAutonomySupervisor(
                    Path(directory) / "mismatch.sqlite3",
                    genesis=_genesis(),
                    initial_state=_state(),
                    clock=_clock(),
                    cycle=cycle,
                    affordances=registry,
                )
                with self.assertRaisesRegex(ValueError, "binding differs"):
                    supervisor.scheduler_tick(f"test:mismatch-{label}")
                self.assertEqual(cycle.learn_calls, 0)
                self.assertEqual(supervisor.state_head().moving_origin_ordinal, -1)

    def test_late_feedback_is_append_only_restartable_and_never_reruns_effect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feedback.sqlite3"
            executor = _UnevaluatedExecutor()
            registry = _registry((_act("reflect.internal"), executor))
            first = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=_LearnedCycle(),
                affordances=registry,
            )
            attempt = first.human_ingress("human:attempt", "Produce one candidate.")
            self.assertEqual(attempt.status, "COMMITTED")
            self.assertEqual(executor.calls, 1)

            def crash(stage):
                if stage == "after_feedback_persisted":
                    raise RuntimeError("injected feedback crash")

            crashing = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=_LearnedCycle(),
                affordances=registry,
                fault_injector=crash,
            )
            source_ref = "sha256:" + "9" * 64
            with self.assertRaisesRegex(RuntimeError, "feedback crash"):
                crashing.submit_feedback(
                    "human:feedback",
                    target_episode_ref=attempt.episode_ref,
                    feedback_text="The attempt was substantially useful.",
                    feedback_source_ref=source_ref,
                    consequence=(("reward", 0.75),),
                )
            restarted = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=_LearnedCycle(),
                affordances=registry,
            )
            feedback = restarted.resume_pending()
            self.assertEqual(feedback.status, "COMMITTED")
            self.assertEqual(executor.calls, 1)
            self.assertEqual(restarted.state_head().moving_origin_ordinal, 1)
            self.assertEqual(json.loads(restarted.state_bytes())["utility"]["reflect.internal"], 0.75)
            payload = json.loads(restarted.episode_items(after_ordinal=0)[0].payload_json)
            self.assertEqual(payload["feedback"]["target_episode_ref"], attempt.episode_ref)
            replay = restarted.submit_feedback(
                "human:feedback",
                target_episode_ref=attempt.episode_ref,
                feedback_text="The attempt was substantially useful.",
                feedback_source_ref=source_ref,
                consequence=(("reward", 0.75),),
            )
            self.assertEqual(replay.episode_ref, feedback.episode_ref)
            with self.assertRaisesRegex(ValueError, "already has"):
                restarted.submit_feedback(
                    "human:feedback-other",
                    target_episode_ref=attempt.episode_ref,
                    feedback_text="Duplicate feedback must not double-train.",
                    feedback_source_ref=source_ref,
                    consequence=(("reward", 0.5),),
                )

    def test_genesis_mismatch_rejected_and_wait_restores_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jenny.sqlite3"
            registry = _registry((_wait(), None))
            first = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=_LearnedCycle(forced=("wait.context",)),
                affordances=registry,
            )
            result = first.scheduler_tick("tick:wait")
            self.assertEqual(result.status, "WAITING")
            pending = first.pending_bytes()
            restarted = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=b"ignored-existing-state",
                clock=_clock(),
                cycle=_LearnedCycle(forced=("wait.context",)),
                affordances=registry,
            )
            self.assertEqual(restarted.pending_bytes(), pending)
            self.assertEqual(restarted.resume_pending().pending_ref, result.pending_ref)
            with self.assertRaisesRegex(RuntimeError, "genesis"):
                PersistentAutonomySupervisor(
                    path,
                    genesis=_genesis("2026-09-02T12:00:01Z"),
                    initial_state=_state(),
                    clock=_clock(),
                    cycle=_LearnedCycle(),
                    affordances=registry,
                )

    def test_receipt_recovery_does_not_duplicate_effect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "crash.sqlite3"
            executor = _CounterExecutor()
            registry = _registry((_act("reflect.internal"), executor))

            def crash(stage):
                if stage == "after_receipt_persisted":
                    raise RuntimeError("injected crash")

            first = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=_LearnedCycle(),
                affordances=registry,
                fault_injector=crash,
            )
            with self.assertRaisesRegex(RuntimeError, "injected crash"):
                first.scheduler_tick("tick:crash")
            self.assertEqual(executor.calls, 1)
            restarted = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=_LearnedCycle(),
                affordances=registry,
            )
            committed = restarted.resume_pending()
            self.assertEqual(committed.status, "COMMITTED")
            self.assertEqual(executor.calls, 1)
            replay = restarted.scheduler_tick("tick:crash")
            self.assertEqual(replay.episode_ref, committed.episode_ref)
            self.assertEqual(executor.calls, 1)

    def test_staged_learned_update_restarts_without_rerunning_learning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "update-crash.sqlite3"
            executor = _CounterExecutor()
            registry = _registry((_act("reflect.internal"), executor))
            cycle = _LearnedCycle()

            def crash(stage):
                if stage == "after_update_persisted":
                    raise RuntimeError("injected update crash")

            first = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=cycle,
                affordances=registry,
                fault_injector=crash,
            )
            with self.assertRaisesRegex(RuntimeError, "injected update crash"):
                first.scheduler_tick("tick:update-crash")
            self.assertEqual(executor.calls, 1)
            self.assertEqual(cycle.learn_calls, 1)
            restarted = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=cycle,
                affordances=registry,
            )
            self.assertEqual(restarted.resume_pending().status, "COMMITTED")
            self.assertEqual(executor.calls, 1)
            self.assertEqual(cycle.learn_calls, 1)

    def test_human_ingress_preempts_wait_and_scheduler_disable_preserves_chat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "human.sqlite3"
            executor = _CounterExecutor()
            registry = _registry(
                (_wait(), None), (_act("reflect.internal"), executor)
            )
            cycle = _LearnedCycle(forced=("wait.context", "reflect.internal", "reflect.internal"))
            supervisor = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=cycle,
                affordances=registry,
            )
            self.assertEqual(supervisor.scheduler_tick("tick:one").status, "WAITING")
            human = supervisor.human_ingress("human:one", "Please continue the internal reflection.")
            self.assertEqual(human.status, "COMMITTED")
            self.assertEqual(executor.calls, 1)
            supervisor.set_scheduler_enabled(False)
            self.assertEqual(supervisor.scheduler_tick("tick:off").status, "DISABLED")
            chat = supervisor.human_ingress("human:two", "Chat remains usable.")
            self.assertEqual(chat.status, "COMMITTED")
            self.assertEqual(executor.calls, 2)

    def test_bounded_userless_continuation_and_outcomes_change_rankings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "continue.sqlite3"
            alpha, beta = _CounterExecutor(-1.0), _CounterExecutor(1.0)
            registry = _registry(
                (_act("act.alpha"), alpha), (_act("act.beta"), beta)
            )
            cycle = _LearnedCycle()
            supervisor = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=cycle,
                affordances=registry,
            )
            results = supervisor.run_internal("life", max_steps=3)
            self.assertEqual([item.status for item in results], ["COMMITTED"] * 3)
            self.assertEqual(results[-1].moving_origin_ordinal, 2)
            self.assertEqual(cycle.seen_rankings[0][0].affordance_id, "act.alpha")
            self.assertEqual(cycle.seen_rankings[1][0].affordance_id, "act.beta")

    def test_shadow_ticks_do_not_advance_semantic_age(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shadow.sqlite3"
            executor = _CounterExecutor()
            registry = _registry((_act("reflect.internal"), executor))
            qualified = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=_LearnedCycle(),
                affordances=registry,
            )
            qualified.scheduler_tick("seed:zero")
            self.assertEqual(qualified.semantic_age(0), 0)
            shadow = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=_LearnedCycle(qualified=False),
                affordances=registry,
            )
            for index in range(50):
                self.assertEqual(shadow.scheduler_tick(f"idle:{index}").status, "SHADOW")
            self.assertEqual(shadow.state_head().moving_origin_ordinal, 0)
            self.assertEqual(shadow.semantic_age(0), 0)
            self.assertEqual(executor.calls, 1)

    def test_external_affordance_is_denied_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external.sqlite3"
            executor = _CounterExecutor()
            registry = _registry(
                (
                    Affordance(
                        "network.publish",
                        "ACT",
                        "Synthetic external publication attempt.",
                        "external.network",
                        external_effect=True,
                    ),
                    executor,
                )
            )
            supervisor = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=_LearnedCycle(),
                affordances=registry,
            )
            result = supervisor.scheduler_tick("external:denied")
            self.assertEqual(result.status, "COMMITTED")
            self.assertEqual(executor.calls, 0)
            self.assertEqual(supervisor.state_head().moving_origin_ordinal, 0)

    def test_unchanged_quiescent_wake_skips_learned_boundary_and_db_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quiescent-checkpoint.sqlite3"
            cycle = _CheckpointedInitiativeCycle(proposals=(None,))
            supervisor = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=cycle,
                affordances=_registry((_act("reflect.internal"), _CounterExecutor())),
            )

            first = supervisor.scheduler_tick("heartbeat:empty-1")
            self.assertEqual(first.status, "QUIESCENT")
            self.assertEqual(cycle.readiness_calls, 1)
            self.assertEqual(cycle.proposal_calls, 1)
            first_head = supervisor.state_head()
            database_after_inspection = path.read_bytes()

            second = supervisor.scheduler_tick("heartbeat:empty-2")
            self.assertEqual(second.status, "QUIESCENT")
            self.assertIn("no canonical state or event change", second.detail)
            self.assertEqual(cycle.readiness_calls, 1)
            self.assertEqual(cycle.proposal_calls, 1)
            self.assertEqual(supervisor.state_head(), first_head)
            self.assertEqual(path.read_bytes(), database_after_inspection)

            restarted_cycle = _CheckpointedInitiativeCycle(proposals=(None,))
            restarted = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=b"ignored-existing-state",
                clock=_clock(),
                cycle=restarted_cycle,
                affordances=_registry((_act("reflect.internal"), _CounterExecutor())),
            )
            after_restart = restarted.scheduler_tick("heartbeat:empty-3")
            self.assertEqual(after_restart.status, "QUIESCENT")
            self.assertIn("no canonical state or event change", after_restart.detail)
            self.assertEqual(restarted_cycle.readiness_calls, 0)
            self.assertEqual(restarted_cycle.proposal_calls, 0)
            self.assertEqual(restarted.state_head(), first_head)

    def test_state_or_event_change_invites_one_new_initiative_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "event-driven-checkpoint.sqlite3"
            cycle = _CheckpointedInitiativeCycle(proposals=(None, None))
            executor = _CounterExecutor()
            supervisor = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=cycle,
                affordances=_registry((_act("reflect.internal"), executor)),
            )

            self.assertEqual(
                supervisor.scheduler_tick("heartbeat:before-human").status,
                "QUIESCENT",
            )
            self.assertEqual(cycle.proposal_calls, 1)
            human = supervisor.human_ingress(
                "human:new-evidence", "A new owner observation changed canonical state."
            )
            self.assertEqual(human.status, "COMMITTED")
            self.assertEqual(executor.calls, 1)

            invited = supervisor.scheduler_tick("heartbeat:after-human")
            self.assertEqual(invited.status, "QUIESCENT")
            self.assertEqual(cycle.readiness_calls, 2)
            self.assertEqual(cycle.proposal_calls, 2)
            deduped = supervisor.scheduler_tick("heartbeat:after-human-again")
            self.assertEqual(deduped.status, "QUIESCENT")
            self.assertEqual(cycle.readiness_calls, 2)
            self.assertEqual(cycle.proposal_calls, 2)

    def test_failed_initiative_inspection_remains_retriable_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failed-checkpoint.sqlite3"
            failed_cycle = _CheckpointedInitiativeCycle(fail_proposals=1)
            first = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=failed_cycle,
                affordances=_registry((_act("reflect.internal"), _CounterExecutor())),
            )
            before = first.state_head()
            with self.assertRaisesRegex(RuntimeError, "injected initiative failure"):
                first.scheduler_tick("heartbeat:failed-inspection")
            self.assertEqual(first.state_head(), before)
            with closing(sqlite3.connect(path)) as connection:
                count = connection.execute(
                    "SELECT COUNT(*) FROM autonomy_wake_checkpoint"
                ).fetchone()[0]
            self.assertEqual(count, 0)

            retry_cycle = _CheckpointedInitiativeCycle(proposals=(None,))
            restarted = PersistentAutonomySupervisor(
                path,
                genesis=_genesis(),
                initial_state=b"ignored-existing-state",
                clock=_clock(),
                cycle=retry_cycle,
                affordances=_registry((_act("reflect.internal"), _CounterExecutor())),
            )
            retried = restarted.scheduler_tick("heartbeat:retry-inspection")
            self.assertEqual(retried.status, "QUIESCENT")
            self.assertEqual(retry_cycle.readiness_calls, 1)
            self.assertEqual(retry_cycle.proposal_calls, 1)
            self.assertEqual(restarted.state_head(), before)

    def test_heartbeat_runs_bounded_internal_life_without_user_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executor = _CounterExecutor()
            registry = _registry((_act("reflect.internal"), executor))
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "heartbeat.sqlite3",
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=_LearnedCycle(),
                affordances=registry,
            )
            heartbeat = AutonomyHeartbeat(
                supervisor, interval_seconds=0.001, max_steps_per_session=3
            )
            heartbeat.start()
            results = heartbeat.join(timeout=2.0)
            self.assertEqual([item.status for item in results], ["COMMITTED"] * 3)
            self.assertEqual(executor.calls, 3)
            self.assertEqual(supervisor.state_head().moving_origin_ordinal, 2)

    def test_foreground_activity_restarts_the_heartbeat_idle_interval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executor = _CounterExecutor()
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "foreground-idle.sqlite3",
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=_LearnedCycle(),
                affordances=_registry((_act("reflect.internal"), executor)),
            )
            heartbeat = AutonomyHeartbeat(
                supervisor, interval_seconds=0.08, max_steps_per_session=1
            )
            heartbeat.start()
            time.sleep(0.05)
            heartbeat.defer_for_foreground()
            # This is later than the original deadline but earlier than the
            # restarted owner-idle deadline.
            time.sleep(0.05)
            self.assertEqual(executor.calls, 0)

            results = heartbeat.join(timeout=1.0)
            self.assertEqual([item.status for item in results], ["COMMITTED"])
            self.assertEqual(executor.calls, 1)

    def test_heartbeat_never_queues_behind_a_busy_foreground_lock(self) -> None:
        class ProbeLock:
            def __init__(self) -> None:
                self.acquire_modes: list[bool] = []
                self.held = False

            def acquire(self, blocking: bool = True) -> bool:
                self.acquire_modes.append(blocking)
                if len(self.acquire_modes) == 1:
                    return False
                self.held = True
                return True

            def release(self) -> None:
                if not self.held:
                    raise RuntimeError("probe lock is not held")
                self.held = False

        with tempfile.TemporaryDirectory() as directory:
            executor = _CounterExecutor()
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "nonblocking-heartbeat.sqlite3",
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=_LearnedCycle(),
                affordances=_registry((_act("reflect.internal"), executor)),
            )
            lock = ProbeLock()
            heartbeat = AutonomyHeartbeat(
                supervisor,
                interval_seconds=0.001,
                max_steps_per_session=1,
                operation_lock=lock,
            )
            heartbeat.start()
            results = heartbeat.join(timeout=1.0)

            self.assertEqual(lock.acquire_modes, [False, False])
            self.assertEqual([item.status for item in results], ["COMMITTED"])
            self.assertEqual(executor.calls, 1)

    def test_foreground_preempts_inflight_target_before_any_canonical_action(self) -> None:
        class BlockingTargetCycle(_LearnedCycle):
            def __init__(self) -> None:
                super().__init__()
                self.formation_started = threading.Event()
                self.release_formation = threading.Event()
                self.pending_target = False
                self.cancel_calls = 0
                self.choose_calls = 0

            def has_autonomous_intent(self, *, state) -> bool:
                return False

            def propose_autonomous_request(self, *, state, temporal, affordances):
                self.formation_started.set()
                if not self.release_formation.wait(2.0):
                    raise TimeoutError("blocking target former was not released")
                self.pending_target = True
                return "Examine one state-grounded unresolved pattern."

            def cancel_pending_autonomous_request(self) -> None:
                self.pending_target = False
                self.cancel_calls += 1

            def choose(self, *, observation, temporal, affordances, state):
                self.choose_calls += 1
                return super().choose(
                    observation=observation,
                    temporal=temporal,
                    affordances=affordances,
                    state=state,
                )

        class CountingPermissionGate(PermissionGate):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def authorize(self, affordance):
                self.calls += 1
                return super().authorize(affordance)

        with tempfile.TemporaryDirectory() as directory:
            cycle = BlockingTargetCycle()
            permission_gate = CountingPermissionGate()
            executor = _CounterExecutor()
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "foreground-preemption.sqlite3",
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=cycle,
                affordances=_registry((_act("reflect.internal"), executor)),
                permission_gate=permission_gate,
            )
            state_before = supervisor.state_bytes()
            head_before = supervisor.state_head()
            episodes_before = supervisor.episode_items()
            heartbeat = AutonomyHeartbeat(
                supervisor,
                interval_seconds=0.001,
                max_steps_per_session=1,
                operation_lock=threading.Lock(),
            )

            heartbeat.start()
            self.assertTrue(cycle.formation_started.wait(1.0))
            heartbeat.defer_for_foreground()
            cycle.release_formation.set()
            results = heartbeat.join(timeout=2.0)

            self.assertEqual([item.status for item in results], ["QUIESCENT"])
            self.assertIn("foreground ingress preempted", results[0].detail)
            self.assertEqual(cycle.cancel_calls, 1)
            self.assertFalse(cycle.pending_target)
            self.assertEqual(cycle.choose_calls, 0)
            self.assertEqual(permission_gate.calls, 0)
            self.assertEqual(executor.calls, 0)
            self.assertEqual(supervisor.state_bytes(), state_before)
            self.assertEqual(supervisor.state_head(), head_before)
            self.assertEqual(supervisor.episode_items(), episodes_before)
            self.assertIsNone(supervisor.pending_bytes())

    def test_foreground_preempts_inflight_selection_before_permission_or_effect(self) -> None:
        class BlockingChoiceCycle(_LearnedCycle):
            def __init__(self) -> None:
                super().__init__()
                self.selection_started = threading.Event()
                self.release_selection = threading.Event()
                self.choose_calls = 0

            def has_autonomous_intent(self, *, state) -> bool:
                return True

            def choose(self, *, observation, temporal, affordances, state):
                self.choose_calls += 1
                self.selection_started.set()
                if not self.release_selection.wait(2.0):
                    raise TimeoutError("blocking selector was not released")
                return super().choose(
                    observation=observation,
                    temporal=temporal,
                    affordances=affordances,
                    state=state,
                )

        class CountingPermissionGate(PermissionGate):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def authorize(self, affordance):
                self.calls += 1
                return super().authorize(affordance)

        with tempfile.TemporaryDirectory() as directory:
            cycle = BlockingChoiceCycle()
            permission_gate = CountingPermissionGate()
            executor = _CounterExecutor()
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "selection-preemption.sqlite3",
                genesis=_genesis(),
                initial_state=_state(),
                clock=_clock(),
                cycle=cycle,
                affordances=_registry((_act("reflect.internal"), executor)),
                permission_gate=permission_gate,
            )
            state_before = supervisor.state_bytes()
            head_before = supervisor.state_head()
            episodes_before = supervisor.episode_items()
            heartbeat = AutonomyHeartbeat(
                supervisor,
                interval_seconds=0.001,
                max_steps_per_session=1,
                operation_lock=threading.Lock(),
            )

            heartbeat.start()
            self.assertTrue(cycle.selection_started.wait(1.0))
            heartbeat.defer_for_foreground()
            cycle.release_selection.set()
            results = heartbeat.join(timeout=2.0)

            self.assertEqual([item.status for item in results], ["QUIESCENT"])
            self.assertIn("after selection", results[0].detail)
            self.assertEqual(cycle.choose_calls, 1)
            self.assertEqual(permission_gate.calls, 0)
            self.assertEqual(executor.calls, 0)
            self.assertEqual(supervisor.state_bytes(), state_before)
            self.assertEqual(supervisor.state_head(), head_before)
            self.assertEqual(supervisor.episode_items(), episodes_before)
            self.assertIsNone(supervisor.pending_bytes())


if __name__ == "__main__":
    unittest.main()
