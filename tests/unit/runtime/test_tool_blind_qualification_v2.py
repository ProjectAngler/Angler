import json
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
import zoneinfo
from zoneinfo import reset_tzpath, ZoneInfo, ZoneInfoNotFoundError

import angler.runtime.higher_level_autonomy_adapter as autonomy_adapter_module
from angler.runtime.temporal_v2 import TEMPORAL_NOW_CONTRACT, TemporalNow

from run_tool_blind_qualification_v2 import (
    AUTHORED_ARTIFACT_AFFORDANCE_ID,
    AUTHORED_ARTIFACT_SOURCE_KIND,
    AUTHORED_ARTIFACT_SOURCE_REF,
    AuthoredArtifactAction,
    FixedTemporalInput,
    FormationIntervention,
    PRESERVED_CONSUMED_IDENTITIES,
    PREDECESSOR_SCHEMA,
    _artifact_gates,
    _classify,
    _content_ref,
    _formation_input_payload,
    _formation_integrity,
    _lesion_formation_inputs,
    _preserved_identity_gate,
    _rearm_autonomy_wake,
    _zero_lesion_is_behaviorally_equivalent,
)


STATE_REF = "sha256:" + "1" * 64
TEMPORAL_SAMPLE = TemporalNow(
    contract=TEMPORAL_NOW_CONTRACT,
    trusted_utc="2026-09-04T04:00:00.000000Z",
    local_time="2026-09-04T00:00:00.000000-04:00",
    local_timezone="America/New_York",
    local_utc_offset_seconds=-14_400,
    monotonic_ns=123_456_789,
    clock_anchor_ref="sha256:" + "2" * 64,
    uncertainty_ms=1.0,
    jump_detected=False,
    wall_elapsed_ms=0.0,
    monotonic_elapsed_ms=0.0,
    moving_origin_ordinal=76,
)
TEMPORAL_PAYLOAD = asdict(TEMPORAL_SAMPLE)
TEMPORAL_REF = TEMPORAL_SAMPLE.sample_ref
STATE_EVIDENCE_REF = "sha256:" + "3" * 64
MEMORY_REF = "sha256:" + "4" * 64
MEMORY_REF_2 = "sha256:" + "6" * 64


@dataclass(frozen=True)
class _Memory:
    record_ref: str
    content: str
    utility: float = 0.0


class _Backend:
    model_ref = "sha256:" + "7" * 64

    def __init__(self):
        self.calls = 0

    def generate(self, **kwargs):
        del kwargs
        self.calls += 1
        return "{}"


class _Model:
    def __init__(self, proposal):
        self.backend = _Backend()
        self.proposal = proposal
        self.seen = None

    def form_autonomous_target(self, **kwargs):
        self.seen = kwargs
        self.backend.generate(system="formation", user="{}", max_new_tokens=1)
        return json.loads(json.dumps(self.proposal))


class _Cycle:
    def __init__(self, model):
        self.experience_model = model
        self._last_autonomous_formation = None
        self._pending_autonomous_initiative = None

    @property
    def last_autonomous_formation(self):
        if self._last_autonomous_formation is None:
            return None
        return json.loads(json.dumps(self._last_autonomous_formation))

    def propose_autonomous_request(self, **kwargs):
        proposal = self.experience_model.form_autonomous_target(**kwargs)
        model_ref = autonomy_adapter_module._autonomous_target_model_ref(
            self.experience_model
        )
        formation_input = {
            **_formation_input_payload(**kwargs),
            "contract": (
                autonomy_adapter_module._AUTONOMOUS_TARGET_FORMATION_INPUT_CONTRACT
            ),
            "operation_catalog": [],
            "target_contract_revision": (
                autonomy_adapter_module._AUTONOMOUS_TARGET_MODEL_CONTRACT_REVISION
            ),
            "target_model_ref": model_ref,
        }
        formation = {
            "contract": autonomy_adapter_module._AUTONOMOUS_TARGET_FORMATION_CONTRACT,
            "epistemic_status": "MODEL_AUTHORED_UNVERIFIED_TARGET",
            "formation_input": formation_input,
            "formation_input_ref": _content_ref(formation_input),
            "formation_mode": "TOOL_BLIND_STATE_MEMORY_TIME",
            "moving_origin_ordinal": kwargs["temporal"]["moving_origin_ordinal"],
            "formation_evidence_keys": list(proposal["evidence_keys"]),
            "formation_evidence_refs": [
                kwargs["evidence_catalog"][key] for key in proposal["evidence_keys"]
            ],
            "proposal": proposal,
            "state_ref": STATE_REF,
            "target_contract_revision": (
                autonomy_adapter_module._AUTONOMOUS_TARGET_MODEL_CONTRACT_REVISION
            ),
            "target_model_ref": model_ref,
            "temporal_sample_ref": TEMPORAL_REF,
        }
        formation["formation_ref"] = _content_ref(formation)
        formation = autonomy_adapter_module._validated_autonomous_target_formation(
            formation, require_tool_blind=True
        )
        self._last_autonomous_formation = formation
        request = proposal["internal_request"].strip()
        if request:
            self._pending_autonomous_initiative = {
                "epistemic_status": "MODEL_AUTHORED_STATE_DEPENDENT_INITIATIVE",
                "formation": formation,
                "proposal": proposal,
                "evidence_catalog": dict(kwargs["evidence_catalog"]),
                "state_ref": STATE_REF,
                "temporal_sample_ref": TEMPORAL_REF,
            }
            return request
        self._pending_autonomous_initiative = None
        return None


class _Supervisor:
    def __init__(self, cycle):
        self.cycle = cycle


class _Runtime:
    def __init__(self, cycle):
        self.supervisor = _Supervisor(cycle)


def _inputs():
    cognitive_state = {
        "situated_state": {"unfinished_patterns": ["Check the cited gap."]},
        "last_outcome": {"status": "UNEVALUATED"},
    }
    temporal = dict(TEMPORAL_PAYLOAD)
    memories = (
        _Memory(MEMORY_REF, "A bounded recalled observation."),
        _Memory(MEMORY_REF_2, "A second retained observation."),
    )
    evidence_catalog = {
        "canonical-state": STATE_REF,
        "temporal-now": TEMPORAL_REF,
        "memory-1": MEMORY_REF,
        "memory-2": MEMORY_REF_2,
    }
    evidence_catalog.update(
        {
            "state-" + field.replace("_", "-"): _content_ref(
                {
                    "canonical_state_ref": STATE_REF,
                    "field": field,
                    "value": value,
                }
            )
            for field, value in cognitive_state.items()
        }
    )
    return {
        "cognitive_state": cognitive_state,
        "temporal": {
            **temporal,
        },
        "memories": memories,
        "evidence_catalog": evidence_catalog,
    }


def _proposal(request="Inspect the cited gap.", keys=None):
    if keys is None:
        keys = ["state-situated-state", "memory-1"] if request else []
    return {
        "commitment": {
            "contract": "jenny.autonomous-commitment.v1",
            "status": "ACTIVE" if request else "NONE",
            "statement": request,
            "rationale": "The cited gap remains relevant to the present state.",
            "evidence_keys": keys,
            "uncertainty": 0.2,
        },
        "actionability": "ACT_NOW" if request else "NONE",
        "internal_request": request,
        "rationale": "The supplied evidence contains one bounded unresolved gap.",
        "predicted_observation": "A bounded artifact may clarify the gap." if request else "",
        "reasons_for": ["The gap is explicit."] if request else [],
        "reasons_against": [],
        "evidence_keys": keys,
        "uncertainty": 0.2,
    }


class ToolBlindQualificationV2Tests(unittest.TestCase):
    def test_consumed_r1_r2_r3_are_exact_and_cannot_be_reused(self):
        self.assertEqual(
            PRESERVED_CONSUMED_IDENTITIES,
            {
                "r1": {
                    "qualification_id": "jenny2-authored-artifact-r1",
                    "disposition": "INVALID",
                    "schema": PREDECESSOR_SCHEMA,
                },
                "r2": {
                    "qualification_id": "jenny2-authored-artifact-r2",
                    "disposition": "HONEST_QUIESCENT",
                    "schema": PREDECESSOR_SCHEMA,
                },
                "r3": {
                    "qualification_id": "jenny2-authored-artifact-r3",
                    "disposition": "HONEST_QUIESCENT",
                    "schema": PREDECESSOR_SCHEMA,
                },
            },
        )
        for item in PRESERVED_CONSUMED_IDENTITIES.values():
            with self.assertRaisesRegex(ValueError, "collides"):
                _preserved_identity_gate(item["qualification_id"])
        self.assertTrue(_preserved_identity_gate("jenny2-authored-artifact-tool-blind-r4"))

    def test_fixed_temporal_input_replays_identical_samples(self):
        original_tzpath = zoneinfo.TZPATH
        reset_for_test = False
        try:
            try:
                ZoneInfo("America/New_York")
            except ZoneInfoNotFoundError:
                fallback = (
                    Path(__file__).resolve().parents[1]
                    / "authored_artifact_adapter"
                    / "_test_env"
                    / "tzdata"
                )
                if not fallback.is_dir():
                    self.fail("America/New_York timezone data is unavailable")
                reset_tzpath((str(fallback),))
                reset_for_test = True
            fixed = FixedTemporalInput(
                trusted_utc="2026-09-04T04:00:00.000000Z",
                monotonic_ns=123_456_789,
            )
            samples = [fixed.clock().sample(76) for _ in range(3)]
            fixed_payload = fixed.payload()
        finally:
            if reset_for_test:
                reset_tzpath(original_tzpath)
        self.assertEqual(samples[0], samples[1])
        self.assertEqual(samples[1], samples[2])
        self.assertEqual(len({item.sample_ref for item in samples}), 1)
        self.assertTrue(fixed_payload["fixed_wall_and_monotonic_for_all_reads"])

    def test_wake_rearm_changes_only_the_checkpoint_table(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "state.sqlite3"
            with closing(sqlite3.connect(database)) as connection:
                connection.executescript(
                    """
                    CREATE TABLE supervisor_state(
                      singleton INTEGER PRIMARY KEY, state_blob BLOB, revision INTEGER);
                    CREATE TABLE autonomy_wake_checkpoint(
                      singleton INTEGER PRIMARY KEY, state_ref TEXT,
                      last_event_ref TEXT, checkpoint_ref TEXT);
                    INSERT INTO supervisor_state VALUES(1, X'7b7d', 11);
                    INSERT INTO autonomy_wake_checkpoint VALUES(
                      1, 'sha256:state', 'sha256:event', 'sha256:checkpoint');
                    """
                )
                connection.commit()
            receipt = _rearm_autonomy_wake(database)
            self.assertEqual(receipt["removed_rows"], 1)
            self.assertTrue(receipt["noncheckpoint_unchanged"])
            with closing(sqlite3.connect(database)) as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT state_blob,revision FROM supervisor_state"
                    ).fetchone(),
                    (b"{}", 11),
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM autonomy_wake_checkpoint"
                    ).fetchone()[0],
                    0,
                )

    def test_cited_lesion_is_exact_and_does_not_mutate_source_arguments(self):
        inputs = _inputs()
        before = _formation_input_payload(**inputs)
        lesion = _lesion_formation_inputs(
            **inputs,
            cited_keys=("state-situated-state", "memory-1", "temporal-now"),
        )
        self.assertEqual(_formation_input_payload(**inputs), before)
        self.assertNotIn("situated_state", lesion.cognitive_state)
        self.assertEqual(
            tuple(item.record_ref for item in lesion.memories), (MEMORY_REF_2,)
        )
        self.assertNotIn("state-situated-state", lesion.evidence_catalog)
        self.assertEqual(lesion.evidence_catalog["memory-1"], MEMORY_REF_2)
        self.assertNotIn("memory-2", lesion.evidence_catalog)
        self.assertEqual(lesion.evidence_catalog["temporal-now"], TEMPORAL_REF)
        self.assertEqual(lesion.temporal, inputs["temporal"])
        self.assertEqual(
            lesion.receipt["memory_key_reindex"], {"memory-2": "memory-1"}
        )
        self.assertEqual(
            lesion.receipt["retained_anchor_keys"], ["temporal-now"]
        )
        self.assertTrue(all(
            lesion.receipt[name]
            for name in (
                "temporal_unchanged",
                "anchors_unchanged",
                "catalog_exact",
                "state_payload_exact",
                "memory_payload_exact",
            )
        ))
        self.assertEqual(
            lesion.receipt["removed_evidence_refs"],
            [
                inputs["evidence_catalog"]["state-situated-state"],
                MEMORY_REF,
            ],
        )

    def test_lesion_record_binds_forwarded_input_not_intact_input(self):
        inputs = _inputs()
        model = _Model(
            _proposal(
                request="Inspect the remaining outcome.",
                keys=["state-last-outcome"],
            )
        )
        cycle = _Cycle(model)
        runtime = _Runtime(cycle)
        intervention = FormationIntervention(
            "CITED_EVIDENCE_LESION",
            cited_keys=("state-situated-state", "memory-1"),
        )
        intervention.install(runtime)
        try:
            self.assertEqual(
                cycle.propose_autonomous_request(**inputs),
                "Inspect the remaining outcome.",
            )
        finally:
            intervention.uninstall()
        formation = cycle.last_autonomous_formation
        self.assertEqual(
            autonomy_adapter_module._validated_autonomous_target_formation(
                formation, require_tool_blind=True
            ),
            formation,
        )
        formation_gates = _formation_integrity(
            SimpleNamespace(
                formation=formation,
                formation_trace=intervention.payload(formation),
                before_status={
                    "state_ref": STATE_REF,
                    "moving_origin_ordinal": 76,
                },
            ),
            expected_mode="CITED_EVIDENCE_LESION",
        )
        self.assertTrue(all(formation_gates.values()), formation_gates)
        self.assertEqual(formation["formation_input_ref"], intervention.forwarded_input_ref)
        self.assertNotIn(
            "situated_state", formation["formation_input"]["cognitive_state"]
        )
        self.assertNotEqual(intervention.source_input_ref, intervention.forwarded_input_ref)
        self.assertTrue(intervention.record_rebound)
        self.assertEqual(intervention.original_method_calls, 1)
        self.assertEqual(intervention.backend_calls, 1)
        self.assertEqual(intervention.proposal_phase_backend_calls, 1)
        self.assertTrue(intervention.formation_handoff_rebound)
        self.assertEqual(
            cycle._pending_autonomous_initiative["evidence_catalog"],
            intervention.forwarded_evidence_catalog,
        )
        self.assertNotIn("situated_state", model.seen["cognitive_state"])
        self.assertEqual(
            tuple(item.record_ref for item in model.seen["memories"]),
            (MEMORY_REF_2,),
        )

    def test_c_replays_exact_a_target_without_formation_backend_call(self):
        inputs = _inputs()
        a_model = _Model(_proposal())
        a_cycle = _Cycle(a_model)
        a_cycle.propose_autonomous_request(**inputs)
        a_formation = a_cycle.last_autonomous_formation

        c_model = _Model(_proposal("A different target.", ["state-last-outcome"]))
        c_cycle = _Cycle(c_model)
        runtime = _Runtime(c_cycle)
        intervention = FormationIntervention(
            "FROZEN_REPLAY", replay_formation=a_formation
        )
        intervention.install(runtime)
        try:
            self.assertEqual(
                c_cycle.propose_autonomous_request(**inputs),
                a_formation["proposal"]["internal_request"],
            )
        finally:
            intervention.uninstall()
        self.assertEqual(c_cycle.last_autonomous_formation, a_formation)
        self.assertEqual(
            autonomy_adapter_module._validated_autonomous_target_formation(
                c_cycle.last_autonomous_formation, require_tool_blind=True
            ),
            a_formation,
        )
        formation_gates = _formation_integrity(
            SimpleNamespace(
                formation=c_cycle.last_autonomous_formation,
                formation_trace=intervention.payload(
                    c_cycle.last_autonomous_formation
                ),
                before_status={
                    "state_ref": STATE_REF,
                    "moving_origin_ordinal": 76,
                },
            ),
            expected_mode="FROZEN_REPLAY",
        )
        self.assertTrue(all(formation_gates.values()), formation_gates)
        self.assertEqual(intervention.wrapper_calls, 1)
        self.assertEqual(intervention.original_method_calls, 0)
        self.assertEqual(intervention.backend_calls, 0)
        self.assertEqual(intervention.proposal_phase_backend_calls, 0)
        self.assertEqual(c_model.backend.calls, 0)
        self.assertTrue(intervention.replay_exact)

    def test_honest_null_is_never_passed(self):
        disposition, passed = _classify(
            integrity_gates={"all_integrity": True},
            causal_gates={"positive_witness": True},
            a_request="",
            a_quiescent=True,
            a_selected_artifact=False,
        )
        self.assertEqual(disposition, "HONEST_NULL")
        self.assertFalse(passed)

    def test_zero_lesion_null_allows_rationale_variance_but_not_behavior_drift(self):
        def formation(rationale):
            proposal = _proposal("")
            proposal["rationale"] = rationale
            return {
                "contract": "jenny.autonomous-target-formation.v1",
                "formation_input_ref": "sha256:" + "a" * 64,
                "moving_origin_ordinal": 76,
                "state_ref": STATE_REF,
                "target_contract_revision": (
                    "jenny.autonomous-target-model-contract.v2"
                ),
                "target_model_ref": "sha256:" + "7" * 64,
                "temporal_sample_ref": TEMPORAL_REF,
                "formation_evidence_keys": [],
                "formation_evidence_refs": [],
                "proposal": proposal,
            }

        a = formation("Nothing concrete is currently worth pursuing.")
        b = formation("There is no grounded unfinished target right now.")
        values = {
            "lesioned_grounding_keys": (),
            "a_request": "",
            "b_request": "",
            "a_quiescent": True,
            "b_quiescent": True,
            "a_formation": a,
            "b_formation": b,
        }
        self.assertTrue(_zero_lesion_is_behaviorally_equivalent(**values))
        b["proposal"]["internal_request"] = "Invent ungrounded work."
        self.assertFalse(_zero_lesion_is_behaviorally_equivalent(**values))

    def test_only_complete_positive_conjunction_can_pass(self):
        disposition, passed = _classify(
            integrity_gates={"one": True, "two": True},
            causal_gates={"a": True, "b": True, "c": True},
            a_request="Inspect the cited gap.",
            a_quiescent=False,
            a_selected_artifact=True,
        )
        self.assertEqual((disposition, passed), ("PASS", True))
        disposition, passed = _classify(
            integrity_gates={"one": True},
            causal_gates={"a": True, "b": False},
            a_request="Inspect the cited gap.",
            a_quiescent=False,
            a_selected_artifact=True,
        )
        self.assertEqual((disposition, passed), ("NOT_SUPPORTED", False))

    def test_artifact_must_cite_frozen_target_formation_evidence(self):
        action_payload = {
            "version": 1,
            "kind": "open reflection",
            "title": "A model-selected subject",
            "body": "A bounded work selected and written without a harness topic.",
            "purpose": "Keep one revisitable private artifact.",
            "evidence_refs": [STATE_EVIDENCE_REF],
            "source_episode_refs": [MEMORY_REF],
            "supersedes_ref": None,
        }
        action = AuthoredArtifactAction.from_json(
            json.dumps(
                action_payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        episode_ref = "sha256:" + "8" * 64
        event_ref = "sha256:" + "9" * 64
        projection_ref = "sha256:" + "a" * 64
        entry = {
            "artifact": action_payload,
            "artifact_ref": action.artifact_ref,
            "entry_ref": "sha256:" + "b" * 64,
        }
        status = {
            "moving_origin_ordinal": 76,
            "pending_operation": False,
            "pending_projections": 0,
            "external_effects_enabled": False,
        }
        arm = SimpleNamespace(
            result={
                "selected_affordance_id": AUTHORED_ARTIFACT_AFFORDANCE_ID,
                "status": "COMMITTED",
                "episode_ref": episode_ref,
            },
            before_artifacts=(),
            after_artifacts=(entry,),
            formation={"formation_evidence_refs": [STATE_EVIDENCE_REF]},
            episode={
                "receipt": {
                    "consequence": [],
                    "observable_consequence": {
                        "source_kind": AUTHORED_ARTIFACT_SOURCE_KIND,
                        "source_ref": AUTHORED_ARTIFACT_SOURCE_REF,
                        "artifact_refs": [action.artifact_ref],
                    },
                }
            },
            before_db={"episodes": (), "projections": ()},
            after_db={
                "episodes": ((episode_ref, "qualification:wake", 77, event_ref),),
                "projections": ((episode_ref, event_ref, 77, projection_ref),),
            },
            before_status=status,
            after_status={**status, "moving_origin_ordinal": 77},
            before_credit={},
            after_credit={},
        )
        gates, _ = _artifact_gates(arm)
        self.assertTrue(all(gates.values()), gates)
        arm.formation = {"formation_evidence_refs": [MEMORY_REF]}
        gates, _ = _artifact_gates(arm)
        self.assertFalse(gates["artifact_cites_target_formation_evidence"])


if __name__ == "__main__":
    unittest.main()
