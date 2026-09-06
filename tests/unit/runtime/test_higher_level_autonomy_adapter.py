from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from angler.runtime.higher_level_autonomy_adapter import (
    AdaptiveHumanTurnRouter,
    CapabilityAwareConsolidationProjector,
    CapabilityModuleHit,
    CanonicalCapabilityCatalog,
    ControllerOutputContractExhausted,
    DeferredSemanticMemory,
    FrozenModelAffordanceController,
    HigherLevelAutonomyCycleAdapter,
    HigherLevelCortexAffordanceExecutor,
    IntentCandidate,
    LearnedAffordanceSelection,
    OutcomeUtilityAffordanceController,
    ReferenceAugmentedSemanticMemory,
    ReferenceAugmentedCapabilityCatalog,
    ReferenceMemoryHit,
    SemanticMemoryConsolidationProjector,
    SupervisorCanonicalMemory,
    _autonomous_target_recall_query,
)
from angler.runtime.higher_level_experience_cycle import (
    ConsequenceVector,
    ExecutionReceipt,
    MemoryCandidate,
    StructuredExperience,
    content_ref,
)
from angler.runtime.jenny_genesis import JennyGenesis
from angler.runtime.latency_trace import capture_latency_trace
from angler.runtime.authored_artifact import (
    AUTHORED_ARTIFACT_AFFORDANCE,
    AUTHORED_ARTIFACT_AFFORDANCE_ID,
    AUTHORED_ARTIFACT_PURPOSE,
    AUTHORED_ARTIFACT_SOURCE_KIND,
    AUTHORED_ARTIFACT_SOURCE_REF,
    AuthoredArtifactAction,
    AuthoredArtifactExecutor,
)
from angler.runtime.jenny_library import (
    JennyLibraryExecutor,
    LIBRARY_AFFORDANCE,
    LIBRARY_AFFORDANCE_ID,
    LIBRARY_CATALOG_CONTRACT,
    LIBRARY_PASSAGE_CONTRACT,
    LIBRARY_SOURCE_KIND,
    library_observation_from_observable,
)
from angler.runtime.persistent_autonomy import (
    Affordance,
    AffordanceReceipt,
    AffordanceRequest,
    CycleObservation,
    DynamicAffordanceRegistry,
    ObservableConsequence,
    PersistentAutonomySupervisor,
    RankedAffordance,
)
from angler.runtime.self_observation_diary import (
    SELF_OBSERVATION_DIARY_AFFORDANCE,
    SELF_OBSERVATION_DIARY_APPRAISAL_ROLE,
    SELF_OBSERVATION_DIARY_CONTRACT,
    SELF_OBSERVATION_DIARY_EPISTEMIC_STATUS,
    SELF_OBSERVATION_DIARY_LABEL_STATUS,
    SELF_OBSERVATION_DIARY_PHENOMENOLOGY_STATUS,
    SELF_OBSERVATION_DIARY_PURPOSE,
    SELF_OBSERVATION_DIARY_SOURCE_KIND,
    SELF_OBSERVATION_DIARY_SOURCE_REF,
    SELF_OBSERVATION_DIARY_STRENGTH_STATUS,
    SelfObservationDiaryExecutor,
    SelfObservationDiaryProposal,
)
from angler.runtime.temporal_v2 import TemporalV2, TrustedClock


QUALIFICATION = "sha256:" + "8" * 64


class _Source:
    def __init__(self):
        self.wall = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
        self.mono = 30_000_000_000

    def wall_now(self):
        value = self.wall
        self.wall += timedelta(milliseconds=1)
        return value

    def mono_now(self):
        value = self.mono
        self.mono += 1_000_000
        return value


class _Memory:
    def __init__(self):
        self.stores = []

    def recall(self, request, *, limit):
        return (
            MemoryCandidate(
                "sha256:" + "a" * 64,
                "Relevant prior attempt.",
                0.1,
                acquired_ordinal=0,
                acquired_time_utc="2026-09-02T11:59:00.000000Z",
            ),
        )[:limit]

    def store(self, content, provenance_refs):
        self.stores.append((content, tuple(provenance_refs)))
        return "sha256:" + "b" * 64


class _Experience:
    def __init__(self):
        self.reflections = 0
        self.last_temporal = None

    def generate_experience(self, request, memories, temporal):
        self.last_temporal = temporal
        return StructuredExperience(
            "Interpret the current constraints.",
            "VERIFY",
            "Attempt and verify against observable constraints.",
            "The objective receipt will reduce uncertainty.",
            ("Check the observable result.",),
            0.4,
            world_model="One bounded synthetic world is active.",
            self_model="The controller remains experimentally unqualified.",
            focus="Resolve the current bounded request.",
        )

    def reflect(self, request, receipt, consequence, experience):
        self.reflections += 1
        return "The outcome supports updating the selected affordance utility."

    def consolidate(self, request, experience, reflection, consequence):
        return json.dumps(
            {
                "lesson": "Retain the verified process/outcome relation.",
                "procedure": "Reuse the verified process, then check the new receipt.",
                "applicability": "A comparable bounded request has observable evidence.",
                "limits": "Re-evaluate when constraints or provenance differ.",
                "world_model": "The bounded environment returned observable evidence.",
                "self_model": "The selected process produced one attributable outcome.",
                "focus": "Use the observed consequence on the next choice.",
                "unfinished_patterns": [],
            },
            separators=(",", ":"),
            sort_keys=True,
        )


class _Cortex:
    def __init__(self):
        self.calls = []

    def execute(self, request, experience, memories):
        self.calls.append((request, experience, tuple(memories)))
        return ExecutionReceipt("sha256:" + "c" * 64, "A bounded public response.", "COMPLETED")


class _ControllerBackend:
    model_ref = "sha256:" + "4" * 64

    def __init__(self, output):
        self.output = output
        self.calls = []

    def generate(self, *, system, user, max_new_tokens):
        self.calls.append((system, user, max_new_tokens))
        value = self.output.pop(0) if type(self.output) is list else self.output
        return json.dumps(value)


class _StructuredControllerBackend(_ControllerBackend):
    def __init__(self, output):
        super().__init__(output)
        self.json_calls = []

    def generate_json(self, *, system, user, max_new_tokens, json_schema):
        self.json_calls.append((system, user, max_new_tokens, json_schema))
        value = self.output.pop(0) if type(self.output) is list else self.output
        return json.dumps(value)


class _ObservedOutcomeExperience(_Experience):
    def __init__(self):
        super().__init__()
        self.assessment_calls = []

    def assess_observed_outcome(
        self,
        request,
        receipt,
        experience,
        *,
        intent_proposal,
        observable_consequence,
        temporal,
        prior_situated_state,
        prediction_catalog,
        evidence_catalog,
    ):
        self.assessment_calls.append(
            {
                "request": request,
                "receipt": receipt,
                "experience": experience,
                "intent_proposal": intent_proposal,
                "observable_consequence": observable_consequence,
                "temporal": temporal,
                "prior_situated_state": prior_situated_state,
                "prediction_catalog": dict(prediction_catalog),
                "evidence_catalog": dict(evidence_catalog),
            }
        )
        relations = {
            "experience.predicted-consequence": "SUPPORTED",
            "choice.expected-state-delta": "PARTIAL",
            "selected-candidate.prediction-1": "CONTRADICTED",
            "selected-candidate.prediction-2": "UNRESOLVED",
        }
        return {
            "prediction_assessments": [
                {
                    "prediction_key": key,
                    "relation": relations[key],
                    "rationale": f"The source-bound observation bears on {key}.",
                    "evidence_keys": ["observable-consequence"],
                }
                for key in prediction_catalog
            ],
            "world_claims": [
                {
                    "text": "The bounded test world returned one source-bound result.",
                    "uncertainty": 0.1,
                    "evidence_keys": ["observable-consequence"],
                }
            ],
            "self_claims": [
                {
                    "text": "The selected observation procedure produced attributable evidence.",
                    "uncertainty": 0.2,
                    "evidence_keys": ["choice", "receipt"],
                }
            ],
            "focus_claims": [
                {
                    "text": "Resolve the remaining uncertainty without treating observation as reward.",
                    "uncertainty": 0.25,
                    "evidence_keys": ["observable-consequence"],
                }
            ],
            "causal_hypotheses": [
                {
                    "text": "The selected procedure may have exposed the observed result.",
                    "uncertainty": 0.5,
                    "evidence_keys": ["choice", "observable-consequence"],
                }
            ],
            "information_gained": ["One exact source-bound result is now available."],
            "limitations": ["No objective value judgment accompanied the result."],
            "unfinished_patterns": ["Check whether independent evidence resolves the result."],
            "next_internal_request": "Check whether independent evidence resolves the result.",
            "reasoned_judgment": (
                "Update situated beliefs provisionally while withholding utility and capability credit."
            ),
            "uncertainty": 0.25,
        }


class _SituatedObservedOutcomeExperience(_ObservedOutcomeExperience):
    def __init__(self):
        super().__init__()
        self.situated_generation_calls = []

    def generate_situated_experience(
        self, request, memories, temporal, *, cognitive_state
    ):
        self.situated_generation_calls.append(
            {
                "request": request,
                "memories": tuple(memories),
                "temporal": temporal,
                "cognitive_state": cognitive_state,
            }
        )
        prior_assessments = cognitive_state["outcome_assessment_evidence"]
        predicted = (
            "A second observation should test the prior contradicted prediction."
            if prior_assessments
            else "The objective receipt will reduce uncertainty."
        )
        return StructuredExperience(
            "Interpret the current constraints from durable cognitive state.",
            "Inspect the source-bound observation.",
            "Compare the next observation with the prior prediction assessment.",
            predicted,
            ("Check the observable result against prior prediction evidence.",),
            0.35,
            world_model="Use the current evidence-bound world claims.",
            self_model="Use the current evidence-bound self claims.",
            focus="Resolve the current bounded uncertainty.",
        )


class _InitiativeExperience(_Experience):
    def __init__(self, request="Investigate the unresolved source-bound question."):
        super().__init__()
        self.request = request
        self.initiative_calls = []

    def propose_autonomous_initiative(self, **kwargs):
        self.initiative_calls.append(kwargs)
        evidence_keys = ["canonical-state"] if self.request else []
        return {
            "commitment": {
                "contract": "jenny.autonomous-commitment.v1",
                "status": "ACTIVE" if self.request else "NONE",
                "statement": self.request,
                "rationale": "The current state may contain unfinished work.",
                "evidence_keys": evidence_keys,
                "uncertainty": 0.3,
            },
            "actionability": "ACT_NOW" if self.request else "NONE",
            "internal_request": self.request,
            "rationale": (
                "The current world state contains a question that the available "
                "bounded observer can investigate."
            ),
            "predicted_observation": (
                "A source-bound observation should narrow the unresolved question."
                if self.request
                else ""
            ),
            "reasons_for": ["A relevant unresolved pattern is present."],
            "reasons_against": [],
            "evidence_keys": evidence_keys,
            "uncertainty": 0.3,
        }


class _ToolBlindInitiativeExperience(_Experience):
    def __init__(self, request="Revisit the specific unresolved state evidence."):
        super().__init__()
        self.request = request
        self.formation_calls = []

    def form_autonomous_target(self, **kwargs):
        self.formation_calls.append(kwargs)
        evidence_keys = ["state-situated-state"] if self.request else []
        return {
            "commitment": {
                "contract": "jenny.autonomous-commitment.v1",
                "status": "ACTIVE" if self.request else "NONE",
                "statement": self.request,
                "rationale": "Specific state evidence remains unresolved.",
                "evidence_keys": evidence_keys,
                "uncertainty": 0.3,
            },
            "actionability": "ACT_NOW" if self.request else "NONE",
            "internal_request": self.request,
            "rationale": "The situated state contains one specific unresolved question.",
            "predicted_observation": (
                "One bounded operation may advance the cited question."
                if self.request
                else ""
            ),
            "reasons_for": ["Specific state evidence remains unresolved."],
            "reasons_against": [],
            "evidence_keys": evidence_keys,
            "uncertainty": 0.3,
        }


class _WaitingToolBlindInitiativeExperience(_Experience):
    def __init__(self):
        super().__init__()
        self.formation_calls = []

    def form_autonomous_target(self, **kwargs):
        self.formation_calls.append(kwargs)
        evidence_key = "state-current-autonomous-commitment"
        if evidence_key not in kwargs["evidence_catalog"]:
            raise AssertionError("current commitment evidence was not supplied")
        return {
            "commitment": {
                "contract": "jenny.autonomous-commitment.v1",
                "status": "ACTIVE",
                "statement": "Resolve the supported question when new evidence arrives.",
                "rationale": "The commitment remains relevant but is not actionable now.",
                "evidence_keys": [evidence_key],
                "uncertainty": 0.35,
            },
            "actionability": "WAIT_FOR_CHANGE",
            "internal_request": "",
            "rationale": "No distinct operation is justified on this unchanged wake.",
            "predicted_observation": (
                "A new state or event may make the commitment actionable."
            ),
            "reasons_for": ["The existing commitment remains supported."],
            "reasons_against": ["No new evidence exists now."],
            "evidence_keys": [evidence_key],
            "uncertainty": 0.35,
        }


class _ObservedOutcomeController:
    def __init__(self):
        self.update_calls = 0

    @property
    def qualification_ref(self):
        return QUALIFICATION

    def select(
        self,
        *,
        observation,
        temporal,
        affordances,
        memories,
        experience,
        state,
        capability_modules=None,
    ):
        del observation, memories, experience, state, capability_modules
        selected = affordances[0]
        candidate = IntentCandidate(
            candidate_id="inspect-observation",
            affordance_id=selected.affordance_id,
            proposed_action="Inspect the bounded source-bound result.",
            desired_state_change="Replace one uncertainty with attributable evidence.",
            rationale="The registered observer can supply relevant evidence.",
            predicted_consequences=(
                "The observer will return a source-bound result.",
                "The result may leave its objective value unresolved.",
            ),
            unknowns=("Whether the observed result warrants a value judgment.",),
            reversibility="REVERSIBLE",
            required_capability_keys=(),
            evidence_refs=(temporal.sample_ref,),
            reasons_for=("The observation is bounded and attributable.",),
            reasons_against=("Observation alone does not establish success.",),
        )
        return LearnedAffordanceSelection(
            selected_affordance_id=selected.affordance_id,
            rankings=(RankedAffordance(selected.affordance_id, 0.75),),
            action_payload=candidate.proposed_action,
            state_assessment="One bounded uncertainty can be inspected.",
            resolution_target="Obtain attributable evidence about the uncertainty.",
            expected_state_delta="The exact observation will provisionally update situated state.",
            evidence_refs=(temporal.sample_ref,),
            intent_candidates=(candidate,),
            selected_candidate_id=candidate.candidate_id,
            candidate_preference_order=(candidate.candidate_id,),
            affordance_preference_order=(selected.affordance_id,),
            selection_basis="This is the sole bounded evidence-producing operation.",
        )

    def update(self, *, selected_affordance_id, consequence, parent_state):
        del selected_affordance_id, consequence
        self.update_calls += 1
        return parent_state


class _CapturingObservedOutcomeController(_ObservedOutcomeController):
    def __init__(self):
        super().__init__()
        self.observations = []

    def select(self, **kwargs):
        self.observations.append(kwargs["observation"])
        return super().select(**kwargs)


class _SelfObservationDiaryController:
    def __init__(self, proposals):
        self.proposals = list(proposals)
        self.states = []
        self.observations = []
        self.update_calls = 0

    @property
    def qualification_ref(self):
        return QUALIFICATION

    def select(
        self,
        *,
        observation,
        temporal,
        affordances,
        memories,
        experience,
        state,
        capability_modules=None,
    ):
        del memories, experience, capability_modules
        self.observations.append(observation)
        self.states.append(json.loads(state))
        if not self.proposals:
            raise AssertionError("diary fixture has no model-authored proposal")
        proposal = self.proposals.pop(0)
        selected = next(
            item
            for item in affordances
            if item.affordance_id == SELF_OBSERVATION_DIARY_AFFORDANCE.affordance_id
        )
        candidate = IntentCandidate(
            candidate_id="record-self-observation",
            affordance_id=selected.affordance_id,
            proposed_action=proposal.canonical_json(),
            desired_state_change=(
                "Preserve one revisable longitudinal hypothesis with provenance."
            ),
            rationale=(
                "The current pattern can be recorded without assigning reward or fact."
            ),
            predicted_consequences=(
                "A source-bound hypothesis becomes available for later comparison.",
            ),
            unknowns=("Whether later observations support or revise the hypothesis.",),
            reversibility="REVERSIBLE",
            required_capability_keys=(),
            evidence_refs=(temporal.sample_ref,),
            reasons_for=("Longitudinal comparison requires an attributable record.",),
            reasons_against=("The interpretation remains uncertain.",),
        )
        return LearnedAffordanceSelection(
            selected_affordance_id=selected.affordance_id,
            rankings=(RankedAffordance(selected.affordance_id, 0.0),),
            action_payload=proposal.canonical_json(),
            state_assessment=(
                "One recurring functional pattern is available as a hypothesis."
            ),
            resolution_target=(
                "Make the hypothesis available for evidence-linked comparison."
            ),
            expected_state_delta=(
                "The diary gains one unverified and revisable observation."
            ),
            evidence_refs=(temporal.sample_ref,),
            intent_candidates=(candidate,),
            selected_candidate_id=candidate.candidate_id,
            candidate_preference_order=(candidate.candidate_id,),
            affordance_preference_order=(selected.affordance_id,),
            selection_basis=(
                "Recording preserves uncertainty and alternatives without value credit."
            ),
        )

    def update(self, *, selected_affordance_id, consequence, parent_state):
        del selected_affordance_id, consequence
        self.update_calls += 1
        return parent_state


class _AuthoredArtifactExperience(_ObservedOutcomeExperience):
    def __init__(self):
        super().__init__()
        self.initiative_calls = []

    def propose_autonomous_initiative(self, **kwargs):
        self.initiative_calls.append(kwargs)
        return {
            "commitment": {
                "contract": "jenny.autonomous-commitment.v1",
                "status": "ACTIVE",
                "statement": (
                    "Choose whether one private authored work is warranted by the "
                    "current evidence."
                ),
                "rationale": "The current state supports a bounded authorship inquiry.",
                "evidence_keys": ["canonical-state"],
                "uncertainty": 0.35,
            },
            "actionability": "ACT_NOW",
            "internal_request": (
                "Choose whether one private authored work is warranted by the "
                "current evidence."
            ),
            "rationale": (
                "The model can preserve one self-chosen work without assigning "
                "it reward or truth."
            ),
            "predicted_observation": (
                "If selected, one exact private artifact will be source-bound."
            ),
            "reasons_for": ["A bounded internal authorship affordance is present."],
            "reasons_against": ["Remaining dormant is also permitted."],
            "evidence_keys": ["canonical-state"],
            "uncertainty": 0.35,
        }

    def assess_observed_outcome(self, *args, **kwargs):
        assessment = super().assess_observed_outcome(*args, **kwargs)
        return {**assessment, "next_internal_request": ""}


class _AuthoredArtifactController:
    def __init__(self, actions):
        self.actions = list(actions)
        self.states = []
        self.update_calls = 0

    @property
    def qualification_ref(self):
        return QUALIFICATION

    def select(
        self,
        *,
        observation,
        temporal,
        affordances,
        memories,
        experience,
        state,
        capability_modules=None,
    ):
        del observation, memories, experience, capability_modules
        self.states.append(json.loads(state))
        if not self.actions:
            raise AssertionError("artifact fixture has no model-authored action")
        action = self.actions.pop(0)
        selected = next(
            item
            for item in affordances
            if item.affordance_id == AUTHORED_ARTIFACT_AFFORDANCE_ID
        )
        candidate = IntentCandidate(
            candidate_id="author-private-work",
            affordance_id=selected.affordance_id,
            proposed_action=action.canonical_json(),
            desired_state_change=(
                "Preserve one exact private model-authored work with provenance."
            ),
            rationale=(
                "The model selected this work from current context without a fixed topic."
            ),
            predicted_consequences=(
                "One source-bound artifact will become canonical state.",
                "The artifact remains unverified and receives no value credit.",
            ),
            unknowns=("Whether later evidence warrants revising the work.",),
            reversibility="REVERSIBLE",
            required_capability_keys=(),
            evidence_refs=(temporal.sample_ref,),
            reasons_for=("The work can preserve a useful model-authored result.",),
            reasons_against=("Authorship does not establish truth or utility.",),
        )
        return LearnedAffordanceSelection(
            selected_affordance_id=selected.affordance_id,
            rankings=(RankedAffordance(selected.affordance_id, 0.0),),
            action_payload=action.canonical_json(),
            state_assessment="One private work can be preserved without external effect.",
            resolution_target="Make the self-chosen work available for later reflection.",
            expected_state_delta="One immutable artifact enters the evidence lineage.",
            evidence_refs=(temporal.sample_ref,),
            intent_candidates=(candidate,),
            selected_candidate_id=candidate.candidate_id,
            candidate_preference_order=(candidate.candidate_id,),
            affordance_preference_order=(selected.affordance_id,),
            selection_basis="The model selected bounded private authorship from current evidence.",
        )

    def update(self, *, selected_affordance_id, consequence, parent_state):
        del selected_affordance_id, consequence
        self.update_calls += 1
        return parent_state


class _LibraryExperience(_ObservedOutcomeExperience):
    def __init__(self):
        super().__init__()
        self.initiative_calls = []

    def propose_autonomous_initiative(self, **kwargs):
        self.initiative_calls.append(kwargs)
        return {
            "commitment": {
                "contract": "jenny.autonomous-commitment.v1",
                "status": "ACTIVE",
                "statement": "Continue one evidence-bound library inquiry.",
                "rationale": "The current reading state supports a bounded inquiry.",
                "evidence_keys": ["canonical-state"],
                "uncertainty": 0.25,
            },
            "actionability": "ACT_NOW",
            "internal_request": "Continue one evidence-bound library inquiry.",
            "rationale": "The current reading state supports one bounded next observation.",
            "predicted_observation": "A verified catalog or passage will be returned.",
            "reasons_for": ["The inquiry has a source-bound next step."],
            "reasons_against": ["Source content remains untrusted evidence."],
            "evidence_keys": ["canonical-state"],
            "uncertainty": 0.25,
        }

    def assess_observed_outcome(self, *args, **kwargs):
        assessment = super().assess_observed_outcome(*args, **kwargs)
        # Leave the operation queue empty so the real initiative boundary is
        # exercised again and receives the just-committed reading state.
        return {**assessment, "next_internal_request": ""}


class _LibraryController:
    def __init__(self, actions):
        self.actions = list(actions)
        self.states = []
        self.observations = []
        self.update_calls = 0

    @property
    def qualification_ref(self):
        return QUALIFICATION

    def select(
        self,
        *,
        observation,
        temporal,
        affordances,
        memories,
        experience,
        state,
        capability_modules=None,
    ):
        del memories, experience, capability_modules
        self.observations.append(observation)
        self.states.append(json.loads(state))
        if not self.actions:
            raise AssertionError("library fixture has no model-authored action")
        action = self.actions.pop(0)
        selected = next(
            item
            for item in affordances
            if item.affordance_id == LIBRARY_AFFORDANCE_ID
        )
        action_payload = json.dumps(
            action,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        candidate = IntentCandidate(
            candidate_id="read-source-evidence",
            affordance_id=selected.affordance_id,
            proposed_action=action_payload,
            desired_state_change="Acquire one exact source-bound reading observation.",
            rationale="The library boundary can return bounded attributable evidence.",
            predicted_consequences=(
                "The library will return a source-bound result.",
                "The result may leave its objective value unresolved.",
            ),
            unknowns=("What interpretation the source evidence warrants.",),
            reversibility="REVERSIBLE",
            required_capability_keys=(),
            evidence_refs=(temporal.sample_ref,),
            reasons_for=("The observation has explicit provenance.",),
            reasons_against=("Reading is not itself reward or truth.",),
        )
        return LearnedAffordanceSelection(
            selected_affordance_id=selected.affordance_id,
            rankings=(RankedAffordance(selected.affordance_id, 0.0),),
            action_payload=action_payload,
            state_assessment="One bounded source question can be investigated.",
            resolution_target="Acquire attributable evidence without assigning value.",
            expected_state_delta="The exact reading cursor and evidence will be retained.",
            evidence_refs=(temporal.sample_ref,),
            intent_candidates=(candidate,),
            selected_candidate_id=candidate.candidate_id,
            candidate_preference_order=(candidate.candidate_id,),
            affordance_preference_order=(selected.affordance_id,),
            selection_basis="Use the sole source-bound reading operation.",
        )

    def update(self, *, selected_affordance_id, consequence, parent_state):
        del selected_affordance_id, consequence
        self.update_calls += 1
        return parent_state


def _synthetic_library(base: Path):
    root = base / "library"
    (root / "catalog").mkdir(parents=True)
    artifacts = {
        "works/first.txt": b"Alpha beta gamma delta epsilon.",
        "works/second.txt": b"Second eligible source.",
        "archives/source.zip": b"PK synthetic excluded source archive",
    }
    items = (
        {
            "path": "works/first.txt",
            "title": "First Synthetic Work",
            "author": "Synthetic Author",
            "source": "https://example.test/first",
            "license_class": "test_only",
            "default_reading_ingest": True,
            "adapter_status": "test_not_for_adapter",
        },
        {
            "path": "works/second.txt",
            "title": "Second Synthetic Work",
            "author": "Synthetic Author",
            "source": "https://example.test/second",
            "license_class": "test_only",
            "default_reading_ingest": True,
            "adapter_status": "test_not_for_adapter",
        },
        {
            "path": "archives/source.zip",
            "title": "Excluded Source Archive",
            "author": "Synthetic Author",
            "source": "https://example.test/archive",
            "license_class": "test_only",
            "default_reading_ingest": False,
            "adapter_status": "test_not_for_adapter",
        },
    )
    for relative, content in artifacts.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    catalog = {
        "schema": "jenny.library.catalog.v1",
        "acquired_at": "2026-09-03",
        "jurisdiction_note": "Synthetic test sources only.",
        "items": items,
    }
    (root / "catalog/library.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "catalog/SHA256SUMS").write_text(
        "\n".join(
            f"{hashlib.sha256(content).hexdigest()}  {relative}"
            for relative, content in sorted(artifacts.items())
        )
        + "\n",
        encoding="utf-8",
    )
    return JennyLibraryExecutor(root)


def _library_registry(executor):
    registry = DynamicAffordanceRegistry()
    registry.register(
        LIBRARY_AFFORDANCE,
        executor,
        observable_source_ref=executor.source_ref,
        observable_source_kind=LIBRARY_SOURCE_KIND,
    )
    return registry


def _episode_observation(supervisor, result):
    episode = json.loads(supervisor.episode_item(result.episode_ref).payload_json)
    payload = episode["receipt"]["observable_consequence"]
    return episode, ObservableConsequence(
        request_ref=payload["request_ref"],
        source_kind=payload["source_kind"],
        source_ref=payload["source_ref"],
        observation_json=payload["observation_json"],
        artifact_refs=tuple(payload["artifact_refs"]),
        evidence_refs=tuple(payload["evidence_refs"]),
    )


def _self_observation_proposal(
    *,
    label="protective attentional return",
    strength=0.63,
    uncertainty=0.31,
    revision_target_ref=None,
    resolution_reason=None,
):
    return SelfObservationDiaryProposal(
        candidate_label=label,
        functional_description=(
            "Attention returns to an unresolved relationship question while "
            "preserving uncertainty about its interpretation."
        ),
        estimated_strength=strength,
        uncertainty=uncertainty,
        observable_signals=(
            "The question was revisited across separated contexts.",
            "Contradictory evidence remained represented.",
        ),
        alternative_explanations=(
            "Recent prompt salience could explain the recurrence.",
            "Retrieval frequency may amplify the pattern.",
        ),
        revision_target_ref=revision_target_ref,
        resolution_reason=resolution_reason,
    )


def _self_observation_registry():
    registry = DynamicAffordanceRegistry()
    registry.register(
        SELF_OBSERVATION_DIARY_AFFORDANCE,
        SelfObservationDiaryExecutor(),
        observable_source_ref=SELF_OBSERVATION_DIARY_SOURCE_REF,
        observable_source_kind=SELF_OBSERVATION_DIARY_SOURCE_KIND,
    )
    return registry


class _CountingAuthoredArtifactExecutor:
    def __init__(self):
        self.calls = 0
        self._delegate = AuthoredArtifactExecutor()

    def __call__(self, request):
        self.calls += 1
        return self._delegate(request)


class _CountingPermissionGate:
    def __init__(self):
        self.calls = 0

    def authorize(self, affordance):
        del affordance
        self.calls += 1
        raise AssertionError("permission must follow authored-artifact validation")


def _authored_artifact_registry(executor=None):
    registry = DynamicAffordanceRegistry()
    registry.register(
        AUTHORED_ARTIFACT_AFFORDANCE,
        AuthoredArtifactExecutor() if executor is None else executor,
        observable_source_ref=AUTHORED_ARTIFACT_SOURCE_REF,
        observable_source_kind=AUTHORED_ARTIFACT_SOURCE_KIND,
    )
    return registry


def _authored_artifact_action(
    *,
    version=1,
    kind="research reflection",
    title="A question worth carrying forward",
    body=(
        "A useful inquiry can remain open without being converted into a fixed "
        "drive. This work records the model's chosen reasoning in its own words."
    ),
    purpose="Preserve a self-chosen line of thought for later evidence-based revision.",
    evidence_refs=(),
    source_episode_refs=(),
    supersedes_ref=None,
):
    return AuthoredArtifactAction(
        version=version,
        kind=kind,
        title=title,
        body=body,
        purpose=purpose,
        evidence_refs=tuple(evidence_refs),
        source_episode_refs=tuple(source_episode_refs),
        supersedes_ref=supersedes_ref,
    )


def _initiative_state():
    state = json.loads(_state())
    state["next_internal_request"] = ""
    return json.dumps(state, separators=(",", ":"), sort_keys=True).encode()


def _tool_blind_grounded_state(**overrides):
    state = {
        "affordance_utility": {},
        "memory_utility": {},
        "next_internal_request": "",
        "affordance_signals": {},
        "motivation_weights": {},
        "situated_state": {
            "world_model": ["One relevant question remains unresolved."],
            "self_model": ["A bounded observer is available."],
            "focus": ["Seek evidence for the unresolved question."],
            "unfinished_patterns": ["Which observation resolves the question?"],
        },
    }
    state.update(overrides)
    return json.dumps(state, separators=(",", ":"), sort_keys=True).encode()


def _consequence(receipt):
    return ConsequenceVector(0.8, 0.7, 0.2, 0.5, 0.8, 0.6, 0.1, 1.0)


def _clock():
    source = _Source()
    return TrustedClock(wall_clock=source.wall_now, monotonic_clock=source.mono_now)


def _artifact_test_clock():
    source = _Source()
    return TrustedClock(
        timezone_name="UTC",
        wall_clock=source.wall_now,
        monotonic_clock=source.mono_now,
    )


def _state():
    return json.dumps(
        {
            "affordance_utility": {},
            "memory_utility": {},
            "next_internal_request": "Inspect an unresolved bounded pattern.",
            "affordance_signals": {},
            "motivation_weights": {},
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


class HigherLevelAutonomyAdapterTests(unittest.TestCase):
    def test_fast_human_turn_remains_situated_and_returns_public_words(self):
        temporal = _clock().sample(-1)
        backend = _ControllerBackend(
            {
                "route": "FAST_RESPONSE",
                "response": "Hello. I am online, temporally situated, and able to use retained context.",
                "interpretation": "A simple greeting warrants a direct situated reply.",
                "rationale": "Further deliberation is unlikely to improve this greeting.",
                "predicted_consequence": "The human receives a clear acknowledgement.",
                "uncertainty": 0.05,
                "reasons_for_direct_response": ["The request is a simple greeting."],
                "reasons_for_further_deliberation": [],
                "unfinished_patterns": [],
                "next_internal_request": "",
                "affordance_preference_order": ["cortex.respond"],
                "required_capability_keys": [],
                "evidence_keys": ["observation"],
            }
        )
        cycle = HigherLevelAutonomyCycleAdapter(
            memory=_Memory(),
            experience_model=_Experience(),
            controller=OutcomeUtilityAffordanceController(
                qualification_ref=QUALIFICATION
            ),
            human_turn_router=AdaptiveHumanTurnRouter(
                backend,
                final_response_authority_ref=backend.model_ref,
                fast_response_qualification_ref=QUALIFICATION,
            ),
        )
        choice = cycle.choose(
            observation=CycleObservation("human:greeting", "HUMAN", "Hello Jenny"),
            temporal=temporal,
            affordances=(
                Affordance("cortex.respond", "ACT", "Respond.", "internal.cognition"),
                Affordance("control.ask", "ASK", "Ask.", "internal.cognition"),
                Affordance("control.wait", "WAIT", "Wait.", "internal.cognition"),
                Affordance("control.stop", "STOP", "Stop.", "internal.cognition"),
            ),
            state=_state(),
        )
        context = json.loads(choice.context_json)
        self.assertEqual(choice.selected_affordance_id, "cortex.respond")
        self.assertEqual(context["adaptive_effort"]["route"], "FAST_RESPONSE")
        self.assertEqual(
            [item["affordance_id"] for item in context["intent_proposal"]["transaction_rankings"]],
            ["cortex.respond"],
        )
        cortex = _Cortex()
        receipt = HigherLevelCortexAffordanceExecutor(
            cortex,
            precomputed_response_authority_ref=backend.model_ref,
            precomputed_response_qualification_ref=QUALIFICATION,
        )(
            AffordanceRequest(
                idempotency_key="sha256:" + "1" * 64,
                trigger_ref="human:greeting",
                affordance_id="cortex.respond",
                observation_ref=CycleObservation(
                    "human:greeting", "HUMAN", "Hello Jenny"
                ).observation_ref,
                state_head_ref="sha256:" + hashlib.sha256(_state()).hexdigest(),
                action_payload="Hello Jenny",
                choice_context_json=choice.context_json,
            )
        )
        self.assertEqual(
            {
                name: context["precomputed_response"][name]
                for name in ("source", "response", "model_ref")
            },
            context["adaptive_response_draft"],
        )
        self.assertEqual(
            context["precomputed_response"]["observation_ref"],
            CycleObservation(
                "human:greeting", "HUMAN", "Hello Jenny"
            ).observation_ref,
        )
        self.assertEqual(
            context["precomputed_response"]["temporal_sample_ref"],
            temporal.sample_ref,
        )
        self.assertEqual(
            context["precomputed_response"]["qualification_ref"],
            QUALIFICATION,
        )
        self.assertIn(
            "temporally situated",
            context["adaptive_response_draft"]["response"],
        )
        self.assertEqual(
            receipt.output,
            "Hello. I am online, temporally situated, and able to use retained context.",
        )
        self.assertEqual(len(cortex.calls), 0)

        tampered = dict(context)
        tampered["adaptive_response_draft"] = {
            **context["adaptive_response_draft"],
            "response": "A different unbound response.",
        }
        with self.assertRaisesRegex(
            ValueError, "precomputed adaptive response is malformed"
        ):
            HigherLevelCortexAffordanceExecutor(
                cortex,
                precomputed_response_authority_ref=backend.model_ref,
                precomputed_response_qualification_ref=QUALIFICATION,
            )(
                AffordanceRequest(
                    idempotency_key="sha256:" + "4" * 64,
                    trigger_ref="human:greeting",
                    affordance_id="cortex.respond",
                    observation_ref=CycleObservation(
                        "human:greeting", "HUMAN", "Hello Jenny"
                    ).observation_ref,
                    state_head_ref="sha256:" + hashlib.sha256(_state()).hexdigest(),
                    action_payload="Hello Jenny",
                    choice_context_json=json.dumps(
                        tampered, separators=(",", ":"), sort_keys=True
                    ),
                )
            )

        tampered_qualification = dict(context)
        tampered_qualification["precomputed_response"] = {
            **context["precomputed_response"],
            "qualification_ref": "sha256:" + "9" * 64,
        }
        with self.assertRaisesRegex(
            ValueError, "precomputed adaptive response is malformed"
        ):
            HigherLevelCortexAffordanceExecutor(
                cortex,
                precomputed_response_authority_ref=backend.model_ref,
                precomputed_response_qualification_ref=QUALIFICATION,
            )(
                AffordanceRequest(
                    idempotency_key="sha256:" + "5" * 64,
                    trigger_ref="human:greeting",
                    affordance_id="cortex.respond",
                    observation_ref=CycleObservation(
                        "human:greeting", "HUMAN", "Hello Jenny"
                    ).observation_ref,
                    state_head_ref="sha256:" + hashlib.sha256(_state()).hexdigest(),
                    action_payload="Hello Jenny",
                    choice_context_json=json.dumps(
                        tampered_qualification,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                )
            )

        unqualified_cycle = HigherLevelAutonomyCycleAdapter(
            memory=_Memory(),
            experience_model=_Experience(),
            controller=OutcomeUtilityAffordanceController(
                qualification_ref=QUALIFICATION
            ),
            human_turn_router=AdaptiveHumanTurnRouter(
                backend, final_response_authority_ref=backend.model_ref
            ),
        )
        unqualified_choice = unqualified_cycle.choose(
            observation=CycleObservation(
                "human:unqualified", "HUMAN", "Hello Jenny"
            ),
            temporal=_clock().sample(-1),
            affordances=(
                Affordance("cortex.respond", "ACT", "Respond.", "internal.cognition"),
                Affordance("control.ask", "ASK", "Ask.", "internal.cognition"),
                Affordance("control.wait", "WAIT", "Wait.", "internal.cognition"),
                Affordance("control.stop", "STOP", "Stop.", "internal.cognition"),
            ),
            state=_state(),
        )
        unqualified_context = json.loads(unqualified_choice.context_json)
        self.assertIsNone(unqualified_context["precomputed_response"])
        fallback_cortex = _Cortex()
        fallback_receipt = HigherLevelCortexAffordanceExecutor(
            fallback_cortex,
            precomputed_response_authority_ref=backend.model_ref,
        )(
            AffordanceRequest(
                idempotency_key="sha256:" + "6" * 64,
                trigger_ref="human:unqualified",
                affordance_id="cortex.respond",
                observation_ref=CycleObservation(
                    "human:unqualified", "HUMAN", "Hello Jenny"
                ).observation_ref,
                state_head_ref="sha256:" + hashlib.sha256(_state()).hexdigest(),
                action_payload="Hello Jenny",
                choice_context_json=unqualified_choice.context_json,
            )
        )
        self.assertEqual(fallback_receipt.output, "A bounded public response.")
        self.assertEqual(len(fallback_cortex.calls), 1)

    def test_autonomous_controller_conclusion_executes_without_second_cortex_call(
        self,
    ) -> None:
        temporal = _clock().sample(-1)
        response = (
            "The retained evidence supports waiting for an observable consequence "
            "before revising the current procedure."
        )
        backend = _ControllerBackend(
            {
                "selected_affordance_id": "cortex.respond",
                "action_payload": response,
                "state_assessment": "One bounded question can be resolved now.",
                "resolution_target": "Distinguish an inference from observed learning.",
                "expected_state_delta": "The distinction becomes explicit without credit.",
                "evidence_refs": [temporal.sample_ref],
                "selected_candidate_id": "retain-grounded-conclusion",
                "candidate_preference_order": ["retain-grounded-conclusion"],
                "affordance_preference_order": ["cortex.respond"],
                "selection_basis": "The available evidence supports a bounded conclusion but no utility update.",
                "intent_candidates": [
                    {
                        "candidate_id": "retain-grounded-conclusion",
                        "affordance_id": "cortex.respond",
                        "proposed_action": response,
                        "desired_state_change": "Preserve the grounded distinction.",
                        "rationale": "No additional model invocation observes new evidence.",
                        "predicted_consequences": [
                            "The conclusion is retained as unevaluated model output."
                        ],
                        "unknowns": ["Whether later feedback supports it."],
                        "reversibility": "REVERSIBLE",
                        "required_capability_keys": [],
                        "evidence_refs": [temporal.sample_ref],
                        "reasons_for": ["The current evidence is sufficient."],
                        "reasons_against": ["No outcome has yet been observed."],
                    }
                ],
            }
        )
        controller = FrozenModelAffordanceController(
            backend,
            qualification_ref=QUALIFICATION,
            require_intent_candidates=True,
            maximum_output_tokens=2_048,
        )
        cycle = HigherLevelAutonomyCycleAdapter(
            memory=_Memory(),
            experience_model=_Experience(),
            controller=controller,
        )
        observation = CycleObservation(
            "scheduler:reuse-controller-conclusion",
            "SCHEDULER",
            "Inspect an unresolved bounded pattern.",
        )
        choice = cycle.choose(
            observation=observation,
            temporal=temporal,
            affordances=(
                Affordance(
                    "cortex.respond", "ACT", "Respond.", "internal.cognition"
                ),
            ),
            state=_state(),
        )
        cortex = _Cortex()
        receipt = HigherLevelCortexAffordanceExecutor(
            cortex,
            autonomous_response_authority_ref=controller.model_ref,
            autonomous_response_qualification_ref=QUALIFICATION,
        )(
            AffordanceRequest(
                idempotency_key="sha256:" + "2" * 64,
                trigger_ref=observation.trigger_ref,
                affordance_id="cortex.respond",
                observation_ref=observation.observation_ref,
                state_head_ref="sha256:" + hashlib.sha256(_state()).hexdigest(),
                action_payload=choice.action_payload,
                choice_context_json=choice.context_json,
            )
        )

        context = json.loads(choice.context_json)
        self.assertEqual(
            context["precomputed_response"]["source"],
            "AUTONOMOUS_AFFORDANCE_CONTROLLER",
        )
        self.assertEqual(receipt.output, response)
        self.assertEqual(receipt.status, "COMPLETED_UNEVALUATED")
        self.assertEqual(cortex.calls, [])
        self.assertEqual(len(backend.calls), 1)

    def test_autonomous_controller_uses_bounded_attributable_working_memory(
        self,
    ) -> None:
        temporal = _clock().sample(-1)
        response = "Retain the current commitment until new evidence arrives."
        backend = _ControllerBackend(
            {
                "selected_affordance_id": "cortex.respond",
                "action_payload": response,
                "state_assessment": "Current evidence does not justify another operation.",
                "resolution_target": "Preserve the commitment without inventing progress.",
                "expected_state_delta": "The unresolved state remains explicit and attributable.",
                "evidence_refs": [temporal.sample_ref],
                "selected_candidate_id": "retain-current-state",
                "candidate_preference_order": ["retain-current-state"],
                "affordance_preference_order": ["cortex.respond"],
                "selection_basis": "No new observation supports a different operation.",
                "intent_candidates": [
                    {
                        "candidate_id": "retain-current-state",
                        "affordance_id": "cortex.respond",
                        "proposed_action": response,
                        "desired_state_change": "Keep the unresolved state visible.",
                        "rationale": "New evidence is absent.",
                        "predicted_consequences": [
                            "No unsupported capability credit is created."
                        ],
                        "unknowns": ["Which future observation resolves the state."],
                        "reversibility": "REVERSIBLE",
                        "required_capability_keys": [],
                        "evidence_refs": [temporal.sample_ref],
                        "reasons_for": ["The decision remains evidence-bound."],
                        "reasons_against": ["It does not resolve the uncertainty."],
                    }
                ],
            }
        )
        controller = FrozenModelAffordanceController(
            backend,
            qualification_ref=QUALIFICATION,
            require_intent_candidates=True,
            maximum_output_tokens=2_048,
        )
        state_payload = json.loads(_state())
        canonical_history = []
        for index in range(8):
            canonical_history.append(
                {
                    "choice_ref": "sha256:" + f"{index + 1:x}" * 64,
                    "epistemic_status": "ATTRIBUTED_MODEL_INFERENCE",
                    "moving_origin_ordinal": index,
                    "detail": f"history-{index}-" + "x" * 20_000,
                }
            )
        history_channels = (
            "intent_ranking_evidence",
            "outcome_assessment_evidence",
            "qualitative_feedback",
            "memory_credit_evidence",
            "self_appraisal_evidence",
            "self_appraisal_calibration",
            "compute_calibration",
        )
        for channel in history_channels:
            state_payload[channel] = canonical_history
        state_payload["last_intent"] = {
            "epistemic_status": "ATTRIBUTED_MODEL_INFERENCE",
            "resolution_target": "Resolve the current evidence gap.",
            "desired_state_change": "Acquire one attributable observation.",
            "predicted_consequence": "The observation may narrow uncertainty.",
            "selected_affordance_id": "cortex.respond",
            "selected_capability_keys": [],
            "uncertainty": 0.5,
            "moving_origin_ordinal": 7,
            "intent_candidates": ["y" * 20_000],
            "alternatives": ["z" * 20_000],
        }
        state = json.dumps(
            state_payload, separators=(",", ":"), sort_keys=True
        ).encode()
        state_before = bytes(state)

        choice = controller.select(
            observation=CycleObservation(
                "scheduler:bounded-working-memory",
                "SCHEDULER",
                "Inspect the current unresolved state.",
            ),
            temporal=temporal,
            affordances=(
                Affordance(
                    "cortex.respond", "ACT", "Respond.", "internal.cognition"
                ),
            ),
            memories=(
                MemoryCandidate(
                    "sha256:" + "a" * 64,
                    "An older semantically relevant outcome.",
                    0.1,
                    acquired_ordinal=0,
                    acquired_time_utc="2026-09-02T11:59:00.000000Z",
                ),
            ),
            experience=_Experience().generate_experience("request", (), temporal),
            state=state,
        )

        self.assertEqual(choice.action_payload, response)
        self.assertEqual(state, state_before)
        self.assertEqual(len(backend.calls), 1)
        supplied = json.loads(backend.calls[0][1])
        self.assertEqual(
            supplied["working_set"]["contract"],
            "jenny.choice-cognitive-working-set.v1",
        )
        self.assertEqual(
            supplied["working_set"]["canonical_state_ref"],
            "sha256:" + hashlib.sha256(state).hexdigest(),
        )
        for channel in history_channels:
            channel_index = supplied["working_set"]["history_channels"][channel]
            self.assertEqual(channel_index["canonical_count"], 8)
            self.assertEqual(channel_index["included_count"], 2)
            self.assertEqual(
                channel_index["representation"],
                "HASH_LINKED_BOUNDED_PROJECTIONS",
            )
            self.assertLessEqual(len(json.dumps(supplied[channel])), 3_072)
            self.assertEqual(len(supplied[channel]), 2)
            self.assertTrue(
                all("canonical_record_ref" in item for item in supplied[channel])
            )
        self.assertNotIn("intent_candidates", supplied["last_intent"])
        self.assertNotIn("alternatives", supplied["last_intent"])
        self.assertIn("grounded_appraisal_dynamics", supplied)
        self.assertLess(len(backend.calls[0][1].encode("utf-8")), 40_000)

    def test_full_human_turn_can_learnedly_choose_unobserved_library_catalog(self):
        backend = _ControllerBackend(
            {
                "route": "FULL_DELIBERATION",
                "response": "",
                "interpretation": "The catalog has not yet been observed.",
                "rationale": "An effectless source observation could resolve the request.",
                "predicted_consequence": "The controller compares reading with direct speech.",
                "uncertainty": 0.4,
                "reasons_for_direct_response": [],
                "reasons_for_further_deliberation": [
                    "Null catalog state is not evidence that the catalog is empty."
                ],
                "unfinished_patterns": [],
                "next_internal_request": "",
                "affordance_preference_order": [
                    "tool.web_research",
                    LIBRARY_AFFORDANCE_ID,
                    "cortex.respond",
                ],
                "required_capability_keys": [],
                "evidence_keys": ["observation"],
            }
        )
        controller = _LibraryController(
            (
                {
                    "operation": "list",
                    "reading_purpose": (
                        "Observe the eligible catalog before answering the human."
                    ),
                },
            )
        )
        cycle = HigherLevelAutonomyCycleAdapter(
            memory=_Memory(),
            experience_model=_Experience(),
            controller=controller,
            human_turn_router=AdaptiveHumanTurnRouter(backend),
        )

        choice = cycle.choose(
            observation=CycleObservation(
                "human:library-catalog", "HUMAN", "Is your library empty?"
            ),
            temporal=_clock().sample(-1),
            affordances=(
                Affordance(
                    "cortex.respond", "ACT", "Respond.", "internal.cognition"
                ),
                LIBRARY_AFFORDANCE,
                Affordance(
                    "tool.hidden", "ACT", "Do not expose.", "internal.cognition"
                ),
                SELF_OBSERVATION_DIARY_AFFORDANCE,
            ),
            state=_state(),
        )

        self.assertEqual(choice.selected_affordance_id, LIBRARY_AFFORDANCE_ID)
        router_system, router_user, _ = backend.calls[0]
        supplied = json.loads(router_user)
        self.assertEqual(
            [item["affordance_id"] for item in supplied["affordances"]],
            ["cortex.respond", LIBRARY_AFFORDANCE_ID],
        )
        self.assertEqual(
            supplied["authoritative_turn_affordance_ids"],
            ["cortex.respond", LIBRARY_AFFORDANCE_ID],
        )
        self.assertIn("NOT_YET_OBSERVED, never EMPTY", router_system)
        self.assertIsNone(
            supplied["cognitive_state"]["library_reading_state"]["catalog"]
        )
        self.assertEqual(
            supplied["cognitive_state"]["library_availability"],
            {
                "catalog_observation_status": "NOT_YET_OBSERVED",
                "contract": "jenny.library.availability-evidence.v1",
                "empty_catalog_supported": False,
                "internal_library_read_available": True,
            },
        )
        # The learned router already selected the unobserved catalog as the
        # first operation. Reopening that decision in the general controller
        # would add a second model call without adding evidence.
        self.assertEqual(len(controller.observations), 0)
        self.assertEqual(
            json.loads(choice.action_payload),
            {
                "operation": "list",
                "reading_purpose": (
                    "An effectless source observation could resolve the request."
                ),
            },
        )
        self.assertEqual(
            json.loads(choice.context_json)["adaptive_effort"][
                "affordance_preference_order"
            ],
            [LIBRARY_AFFORDANCE_ID, "cortex.respond"],
        )
        self.assertIsNone(json.loads(choice.context_json)["precomputed_response"])

    def test_human_router_resolves_named_last_outcome_evidence(self):
        temporal = _clock().sample(-1)
        receipt_ref = "sha256:" + "6" * 64
        observation_ref = "sha256:" + "7" * 64
        state = json.loads(_state())
        state["last_outcome"] = {
            "receipt_ref": receipt_ref,
            "observation_ref": observation_ref,
        }
        backend = _ControllerBackend(
            {
                "route": "FAST_RESPONSE",
                "response": "The last observed result is available with its receipt.",
                "interpretation": "Answer from the most recent bound observation.",
                "rationale": "The receipt and observation directly support the answer.",
                "predicted_consequence": "The human receives a provenance-bound answer.",
                "uncertainty": 0.1,
                "reasons_for_direct_response": ["The cited outcome is already present."],
                "reasons_for_further_deliberation": [],
                "unfinished_patterns": [],
                "next_internal_request": "",
                "affordance_preference_order": ["cortex.respond"],
                "required_capability_keys": [],
                "evidence_keys": [
                    "observation",
                    "receipt,observable-consequence",
                ],
            }
        )
        decision = AdaptiveHumanTurnRouter(backend).decide(
            observation=CycleObservation(
                "human:last-outcome", "HUMAN", "What happened last?"
            ),
            temporal=temporal,
            affordances=(
                Affordance(
                    "cortex.respond", "ACT", "Respond.", "internal.cognition"
                ),
            ),
            memories=(),
            state=json.dumps(state, separators=(",", ":"), sort_keys=True).encode(),
        )
        self.assertEqual(
            decision.evidence_refs,
            (
                CycleObservation(
                    "human:last-outcome", "HUMAN", "What happened last?"
                ).observation_ref,
                receipt_ref,
                observation_ref,
            ),
        )

    def test_human_router_uses_bounded_semantic_prompt_not_hash_plumbing(self):
        backend = _ControllerBackend(
            {
                "route": "FAST_RESPONSE",
                "response": "Hello. I retained the relevant context.",
                "interpretation": "Answer the current greeting directly.",
                "rationale": "The retrieved working set is sufficient.",
                "predicted_consequence": "The human receives a prompt greeting.",
                "uncertainty": 0.05,
                "reasons_for_direct_response": ["No further evidence is needed."],
                "reasons_for_further_deliberation": [],
                "unfinished_patterns": [],
                "next_internal_request": "",
                "affordance_preference_order": ["cortex.respond"],
                "required_capability_keys": [],
                "evidence_keys": ["observation"],
            }
        )
        state_payload = json.loads(_state())
        opaque_ref = "sha256:" + "9" * 64
        large_record = {
            "choice_ref": opaque_ref,
            "receipt_ref": opaque_ref,
            "detail": "x" * 20_000,
        }
        history_channels = (
            "intent_ranking_evidence",
            "outcome_assessment_evidence",
            "qualitative_feedback",
            "memory_credit_evidence",
            "self_appraisal_evidence",
            "self_appraisal_calibration",
            "compute_calibration",
        )
        for channel in history_channels:
            state_payload[channel] = [dict(large_record) for _ in range(8)]
        state_payload["situated_state"] = {
            "world_model": "A bounded test world is active.",
            "self_model": "Relevant context is selected before generation.",
            "focus": ["Answer the current turn."],
            "outcome_claims": {"world": ["x" * 20_000]},
            "choice_ref": opaque_ref,
        }
        state_payload["last_intent"] = {
            "epistemic_status": "MODEL_INTENT",
            "resolution_target": "Preserve only relevant prior continuity.",
            "desired_state_change": "The current turn receives a grounded answer.",
            "selected_affordance_id": "cortex.respond",
            "selected_capability_keys": [],
            "uncertainty": 0.1,
            "alternatives": [{"detail": "x" * 20_000}],
            "intent_candidates": [{"detail": "x" * 20_000}],
        }
        state_payload["last_outcome"] = {
            "epistemic_status": "OBSERVED",
            "receipt_ref": opaque_ref,
            "choice_ref": opaque_ref,
        }
        state_payload["last_human_interaction"] = {
            "human_observation": "A prior greeting.",
            "model_output": "A prior response.",
            "receipt_ref": opaque_ref,
        }
        state = json.dumps(
            state_payload, separators=(",", ":"), sort_keys=True
        ).encode()
        original_state = bytes(state)
        observation = CycleObservation(
            "human:semantic-working-set", "HUMAN", "Hello Jenny"
        )
        trace = {}
        with capture_latency_trace(
            request_id="semantic-working-set-test", source="HUMAN"
        ) as trace:
            decision = AdaptiveHumanTurnRouter(backend).decide(
                observation=observation,
                temporal=_clock().sample(-1),
                affordances=(
                    Affordance(
                        "cortex.respond", "ACT", "Respond.", "internal.cognition"
                    ),
                ),
                memories=(
                    MemoryCandidate(
                        opaque_ref,
                        "One semantically retrieved memory remains visible.",
                        0.1,
                    ),
                ),
                state=state,
                trusted_composition={
                    "SELF": {
                        "system": {
                            "composition_integrity": "FRESH",
                            "operational_status": "READY",
                            "claims_current": True,
                            "refresh_required": False,
                        }
                    },
                    "WORLD": {
                        "available_interfaces": {
                            "composition_integrity": "FRESH",
                            "operational_status": "READY",
                            "claims_current": True,
                            "refresh_required": False,
                            "effect_boundary": {
                                "general_external_effects_enabled": False
                            },
                            "entries": [
                                {"description": "x" * 20_000}
                                for _ in range(32)
                            ],
                        }
                    },
                },
            )

        self.assertEqual(state, original_state)
        self.assertEqual(decision.evidence_refs, (observation.observation_ref,))
        supplied_text = backend.calls[0][1]
        supplied = json.loads(supplied_text)
        self.assertNotIn("evidence_catalog", supplied)
        self.assertIn("available_evidence_keys", supplied)
        self.assertIn("memory-1", supplied["available_evidence_keys"])
        self.assertEqual(
            supplied["retrieved_memories"][0]["evidence_key"], "memory-1"
        )
        self.assertNotIn("record_ref", supplied["retrieved_memories"][0])
        self.assertNotIn("sha256:", supplied_text)
        cognitive = supplied["cognitive_state"]
        self.assertNotIn("outcome_assessment_evidence", cognitive)
        self.assertNotIn("last_intent", cognitive)
        self.assertNotIn(
            "alternatives", cognitive["prior_intent_continuity"]
        )
        self.assertEqual(
            cognitive["prior_intent_continuity"]["resolution_target"],
            "Preserve only relevant prior continuity.",
        )
        self.assertNotIn("outcome_claims", cognitive["situated_state"])
        self.assertEqual(
            cognitive["trusted_composition"]["WORLD"]
            ["available_interfaces"]["active_interface_count"],
            32,
        )
        self.assertEqual(
            cognitive["working_set"]["history_channels"]
            ["outcome_assessment_evidence"]["canonical_count"],
            8,
        )
        self.assertLess(len(supplied_text.encode("utf-8")), 8_000)
        trace_input = trace["adaptive_router_input"]
        self.assertEqual(
            trace_input["user_utf8_bytes"], len(supplied_text.encode("utf-8"))
        )
        self.assertEqual(trace_input["retrieved_memory_count"], 1)

    def test_human_router_repairs_stale_decision_not_bound_to_current_turn(self):
        temporal = _clock().sample(-1)
        stale = {
            "route": "FAST_RESPONSE",
            "response": "This answers an earlier request.",
            "interpretation": "Continue the previous task.",
            "rationale": "The previous state is available.",
            "predicted_consequence": "The previous task receives an answer.",
            "uncertainty": 0.1,
            "reasons_for_direct_response": ["The prior state is explicit."],
            "reasons_for_further_deliberation": [],
            "unfinished_patterns": [],
            "next_internal_request": "",
            "affordance_preference_order": ["cortex.respond"],
            "required_capability_keys": [],
            "evidence_keys": ["canonical-state"],
        }
        repaired = {
            **stale,
            "response": "This answers the current request.",
            "interpretation": "Answer the current request.",
            "rationale": "The current observation is authoritative.",
            "predicted_consequence": "The current request receives an answer.",
            "evidence_keys": ["observation"],
        }
        backend = _ControllerBackend([stale, repaired])
        observation = CycleObservation(
            "human:current-turn", "HUMAN", "Please answer this current request."
        )
        decision = AdaptiveHumanTurnRouter(backend).decide(
            observation=observation,
            temporal=temporal,
            affordances=(
                Affordance(
                    "cortex.respond", "ACT", "Respond.", "internal.cognition"
                ),
            ),
            memories=(),
            state=_state(),
        )
        self.assertEqual(decision.response, "This answers the current request.")
        self.assertEqual(decision.evidence_refs, (observation.observation_ref,))
        self.assertEqual(len(backend.calls), 2)
        first_payload = json.loads(backend.calls[0][1])
        self.assertEqual(first_payload["turn_to_answer"]["content"], observation.content)

    def test_outcome_profile_preserves_informative_near_miss_dimensions(self):
        controller = OutcomeUtilityAffordanceController(alpha=0.5)
        consequence = ConsequenceVector(
            0.0, 0.4, 0.9, 0.8, 0.7, 0.3, 0.2, 1.0, 0.25
        )
        learned = json.loads(
            controller.update(
                selected_affordance_id="cortex.respond",
                consequence=consequence,
                parent_state=_state(),
            )
        )
        profile = learned["affordance_outcome_profiles"]["cortex.respond"]
        self.assertEqual(profile["observations"], 1)
        self.assertGreater(profile["information_gain"], profile["objective_progress"])
        self.assertGreater(profile["human_feedback"], profile["objective_progress"])
        progress = learned["latest_learning_progress"]
        self.assertEqual(progress["affordance_id"], "cortex.respond")
        self.assertIn("prediction_error", progress["outcome_profile_delta"])

    def test_agent_ingress_propagates_typed_exhausted_controller_contract_without_commit(
        self,
    ) -> None:
        unavailable_ref = "sha256:" + "f" * 64
        invalid_choice = {
            "selected_affordance_id": "cortex.respond",
            "rankings": [
                {"affordance_id": "cortex.respond", "score": 1.0},
            ],
            "action_payload": "Answer the delegated request.",
            "state_assessment": "The request is bounded.",
            "resolution_target": "Answer the request.",
            "expected_state_delta": "The request receives an answer.",
            "evidence_refs": [unavailable_ref],
        }
        backend = _ControllerBackend([invalid_choice, invalid_choice])
        registry = DynamicAffordanceRegistry()
        registry.register(
            Affordance(
                "cortex.respond", "ACT", "Respond.", "internal.cognition"
            ),
            HigherLevelCortexAffordanceExecutor(_Cortex(), _consequence),
        )
        with tempfile.TemporaryDirectory() as directory:
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "contract-exhaustion.sqlite3",
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=_state(),
                clock=_clock(),
                cycle=HigherLevelAutonomyCycleAdapter(
                    memory=_Memory(),
                    experience_model=_Experience(),
                    controller=FrozenModelAffordanceController(
                        backend,
                        qualification_ref=QUALIFICATION,
                    ),
                ),
                affordances=registry,
            )
            before_state = supervisor.state_bytes()
            before_head = supervisor.state_head()

            with self.assertRaises(ControllerOutputContractExhausted) as raised:
                supervisor.agent_ingress(
                    "agent:contract-exhaustion",
                    "Answer this delegated request.",
                )

            self.assertIn(
                "unavailable state evidence reference",
                raised.exception.initial_error,
            )
            self.assertIn(
                "unavailable state evidence reference",
                raised.exception.repair_error,
            )
            self.assertEqual(len(backend.calls), 2)
            self.assertEqual(supervisor.state_bytes(), before_state)
            self.assertEqual(supervisor.state_head(), before_head)
            self.assertIsNone(supervisor.pending_bytes())

    def test_selection_level_model_contract_exhaustion_is_typed(self) -> None:
        temporal = _clock().sample(-1)
        invalid_choice = {
            "selected_affordance_id": "cortex.nowhere",
            "rankings": [
                {"affordance_id": "cortex.respond", "score": 1.0},
            ],
            "action_payload": "Answer the delegated request.",
            # This passes the field extraction in decode but is rejected by the
            # complete LearnedAffordanceSelection contract.
            "state_assessment": "The request is clear.",
            "resolution_target": "Answer the request.",
            "expected_state_delta": "The request receives an answer.",
            "evidence_refs": [temporal.sample_ref],
        }
        backend = _ControllerBackend([invalid_choice, invalid_choice])

        with self.assertRaises(ControllerOutputContractExhausted) as raised:
            FrozenModelAffordanceController(
                backend,
                qualification_ref=QUALIFICATION,
            ).select(
                observation=CycleObservation(
                    "agent:selection-contract",
                    "AGENT",
                    "Answer this delegated request.",
                ),
                temporal=temporal,
                affordances=(
                    Affordance(
                        "cortex.respond",
                        "ACT",
                        "Respond.",
                        "internal.cognition",
                    ),
                ),
                memories=(),
                experience=_Experience().generate_experience(
                    "Answer this delegated request.",
                    (),
                    type("T", (), {"now": 0})(),
                ),
                state=_state(),
            )

        self.assertIn(
            "controller selected an unavailable affordance",
            raised.exception.initial_error,
        )
        self.assertIn(
            "controller selected an unavailable affordance",
            raised.exception.repair_error,
        )
        self.assertEqual(len(backend.calls), 2)

    def test_agent_ingress_does_not_retype_canonical_state_integrity_failure(
        self,
    ) -> None:
        invalid_state = json.loads(_state())
        invalid_state["memory_utility"] = []
        backend = _ControllerBackend({})
        registry = DynamicAffordanceRegistry()
        registry.register(
            Affordance(
                "cortex.respond", "ACT", "Respond.", "internal.cognition"
            ),
            HigherLevelCortexAffordanceExecutor(_Cortex(), _consequence),
        )
        with tempfile.TemporaryDirectory() as directory:
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "state-integrity.sqlite3",
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=json.dumps(
                    invalid_state, separators=(",", ":"), sort_keys=True
                ).encode(),
                clock=_clock(),
                cycle=HigherLevelAutonomyCycleAdapter(
                    memory=_Memory(),
                    experience_model=_Experience(),
                    controller=FrozenModelAffordanceController(
                        backend,
                        qualification_ref=QUALIFICATION,
                    ),
                ),
                affordances=registry,
            )
            before_state = supervisor.state_bytes()
            before_head = supervisor.state_head()

            with self.assertRaisesRegex(
                ValueError, "memory_utility must be an object"
            ) as raised:
                supervisor.agent_ingress(
                    "agent:state-integrity",
                    "Answer this delegated request.",
                )

            self.assertIs(type(raised.exception), ValueError)
            self.assertEqual(backend.calls, [])
            self.assertEqual(supervisor.state_bytes(), before_state)
            self.assertEqual(supervisor.state_head(), before_head)
            self.assertIsNone(supervisor.pending_bytes())

    def test_structured_controller_uses_one_capacity_bound_schema_pass(self) -> None:
        temporal = _clock().sample(-1)
        output = {
            "selected_affordance_id": "cortex.respond",
            "action_payload": "Answer from the current evidence.",
            "state_assessment": "The current evidence supports a direct answer.",
            "resolution_target": "Answer the bounded request.",
            "expected_state_delta": "The owner receives one grounded answer.",
            "evidence_refs": [temporal.sample_ref],
            "selected_candidate_id": "answer-now",
            "candidate_preference_order": ["answer-now", "explain-limits"],
            "affordance_preference_order": ["cortex.respond"],
            "selection_basis": "The first operation answers directly; the second adds unnecessary detail.",
            "intent_candidates": [
                {
                    "candidate_id": "answer-now",
                    "affordance_id": "cortex.respond",
                    "proposed_action": "Answer from the current evidence.",
                    "desired_state_change": "Resolve the request.",
                    "rationale": "The answer is already supported.",
                    "predicted_consequences": ["The owner receives an answer."],
                    "unknowns": [],
                    "reversibility": "REVERSIBLE",
                    "required_capability_keys": [],
                    "evidence_refs": [temporal.sample_ref],
                    "reasons_for": ["No further observation is needed."],
                    "reasons_against": [],
                },
                {
                    "candidate_id": "explain-limits",
                    "affordance_id": "cortex.respond",
                    "proposed_action": "Describe the evidence limits before answering.",
                    "desired_state_change": "Expose remaining uncertainty.",
                    "rationale": "The evidence has bounded scope.",
                    "predicted_consequences": ["The limits become explicit."],
                    "unknowns": [],
                    "reversibility": "REVERSIBLE",
                    "required_capability_keys": [],
                    "evidence_refs": [temporal.sample_ref],
                    "reasons_for": ["It preserves uncertainty."],
                    "reasons_against": ["It is unnecessary for this request."],
                },
            ],
        }
        backend = _StructuredControllerBackend(output)
        selection = FrozenModelAffordanceController(
            backend,
            qualification_ref=QUALIFICATION,
            require_intent_candidates=True,
            maximum_output_tokens=4_096,
        ).select(
            observation=CycleObservation(
                "scheduler:structured-controller",
                "SCHEDULER",
                "Answer the bounded request.",
            ),
            temporal=temporal,
            affordances=(
                Affordance(
                    "cortex.respond", "ACT", "Respond.", "internal.cognition"
                ),
            ),
            memories=(),
            experience=_Experience().generate_experience(
                "request", (), type("T", (), {"now": 0})()
            ),
            state=_state(),
        )

        self.assertEqual(selection.selected_candidate_id, "answer-now")
        self.assertEqual(selection.compute_route, "SINGLE_PASS")
        self.assertEqual(backend.calls, [])
        self.assertEqual(len(backend.json_calls), 1)
        _, _, token_ceiling, schema = backend.json_calls[0]
        self.assertEqual(token_ceiling, 4_096)
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            schema["properties"]["intent_candidates"]["minItems"], 1
        )
        self.assertEqual(
            schema["properties"]["intent_candidates"]["maxItems"], 5
        )
        self.assertIn("1 through 5", backend.json_calls[0][0])

    def test_frozen_model_controller_ranks_dynamic_affordances_without_code_priority(self):
        backend = _ControllerBackend({})
        controller = FrozenModelAffordanceController(
            backend, qualification_ref=QUALIFICATION
        )
        temporal = _clock().sample(-1)
        backend.output = {
                "selected_affordance_id": "control.wait",
                "rankings": [
                    {"affordance_id": "cortex.respond", "score": 0.1},
                    {"affordance_id": "control.wait", "score": 0.8},
                ],
                "action_payload": None,
                "state_assessment": "No supported unresolved state is present.",
                "resolution_target": "Preserve state pending relevant evidence.",
                "expected_state_delta": "State remains unchanged until new evidence arrives.",
                "evidence_refs": [temporal.sample_ref],
            }
        from angler.runtime.persistent_autonomy import CycleObservation

        selection = controller.select(
            observation=CycleObservation("scheduler:model", "SCHEDULER", ""),
            temporal=temporal,
            affordances=(
                Affordance("cortex.respond", "ACT", "Run cortex.", "internal.cognition"),
                Affordance("control.wait", "WAIT", "Wait.", "internal.cognition"),
            ),
            memories=(),
            experience=_Experience().generate_experience("request", (), type("T", (), {"now": 0})()),
            state=_state(),
        )
        self.assertEqual(selection.selected_affordance_id, "control.wait")
        self.assertEqual([item.affordance_id for item in selection.rankings], ["control.wait", "cortex.respond"])
        self.assertEqual(selection.action_payload, "")
        self.assertEqual(len(backend.calls), 1)
        backend.output = [
            {
                "selected_affordance_id": "cortex.respond",
                "rankings": [
                    {"affordance_id": "cortex.respond", "score": 0.9},
                    {"affordance_id": "control.wait", "score": 0.1},
                ],
                "action_payload": None,
                "state_assessment": "One unresolved bounded state is present.",
                "resolution_target": "Resolve the bounded state.",
                "expected_state_delta": "The uncertainty should decrease.",
                "evidence_refs": [temporal.sample_ref],
            },
            {
                "selected_affordance_id": "control.wait",
                "rankings": [
                    {"affordance_id": "cortex.respond", "score": 0.1},
                    {"affordance_id": "control.wait", "score": 0.8},
                ],
                "action_payload": "",
                "state_assessment": "The original active choice lacked an executable inquiry.",
                "resolution_target": "Preserve state until a supported inquiry exists.",
                "expected_state_delta": "State remains unchanged pending relevant evidence.",
                "evidence_refs": [temporal.sample_ref],
            },
        ]
        revised = controller.select(
            observation=CycleObservation("scheduler:empty-act", "SCHEDULER", ""),
            temporal=temporal,
            affordances=(
                Affordance("cortex.respond", "ACT", "Run cortex.", "internal.cognition"),
                Affordance("control.wait", "WAIT", "Wait.", "internal.cognition"),
            ),
            memories=(),
            experience=_Experience().generate_experience(
                "request", (), type("T", (), {"now": 0})()
            ),
            state=_state(),
        )
        self.assertEqual(revised.selected_affordance_id, "control.wait")
        self.assertEqual(revised.action_payload, "")
        self.assertEqual(len(backend.calls), 3)

    def test_frozen_model_controller_selects_from_open_ended_intent_candidates(self):
        temporal = _clock().sample(-1)
        backend = _ControllerBackend(
            {
                "selected_affordance_id": "control.ask",
                "action_payload": "Redundant top-level wording that is not authoritative.",
                "unused_commentary": "This inert extra field must not affect execution.",
                "requires_extended_deliberation": False,
                "compute_rationale": "The evidence gap and preferred question are already clear.",
                "state_assessment": "Two plausible sources conflict and owner authority can resolve provenance.",
                "resolution_target": "Identify the authoritative source before synthesis.",
                "expected_state_delta": "The answer will disambiguate provenance and enable a grounded comparison.",
                "evidence_refs": [temporal.sample_ref],
                "selected_candidate_id": "ask-authority",
                "candidate_preference_order": [
                    "ask-authority", "investigate-known-evidence"
                ],
                "affordance_preference_order": [
                    "control.ask", "cortex.respond"
                ],
                "selection_basis": "Only the owner can resolve source authority; internal comparison can expose but not settle the conflict.",
                "intent_candidates": [
                    {
                        "candidate_id": "investigate-known-evidence",
                        "affordance_id": "cortex.respond",
                        "proposed_action": "Compare only the currently available evidence and expose the conflict.",
                        "desired_state_change": "Bound the disagreement without inventing authority.",
                        "rationale": "An internal comparison can identify but not settle the provenance conflict.",
                        "predicted_consequences": ["The unresolved source conflict remains explicit."],
                        "unknowns": ["Which source the owner designates as authoritative."],
                        "reversibility": "REVERSIBLE",
                        "required_capability_keys": [],
                        "evidence_refs": [temporal.sample_ref],
                        "reasons_for": ["It can expose the exact disagreement."],
                        "reasons_against": ["It cannot assign project authority."],
                    },
                    {
                        "candidate_id": "ask-authority",
                        "unused_score": 0.99,
                        "affordance_id": "control.ask",
                        "proposed_action": "Which source should we treat as authoritative for this comparison?",
                        "desired_state_change": "Resolve the provenance ambiguity with owner guidance.",
                        "rationale": "Only the owner can designate authority for this project decision.",
                        "predicted_consequences": [
                            "The provenance decision becomes explicit.",
                            "Subsequent research can use the selected source consistently.",
                        ],
                        "unknowns": ["The owner's source preference."],
                        "reversibility": "REVERSIBLE",
                        "required_capability_keys": [],
                        "evidence_refs": [temporal.sample_ref],
                        "reasons_for": ["The missing fact is owner authority."],
                        "reasons_against": ["It interrupts the owner."],
                    },
                ],
            }
        )
        controller = FrozenModelAffordanceController(
            backend,
            draft_backend=backend,
            qualification_ref=QUALIFICATION,
            require_intent_candidates=True,
            maximum_output_tokens=2_048,
            maximum_draft_output_tokens=2_048,
        )
        from angler.runtime.persistent_autonomy import CycleObservation

        selection = controller.select(
            observation=CycleObservation("scheduler:research", "SCHEDULER", ""),
            temporal=temporal,
            affordances=(
                Affordance("cortex.respond", "ACT", "Run cortex.", "internal.cognition"),
                Affordance("control.ask", "ASK", "Ask the human.", "internal.cognition"),
            ),
            memories=(),
            experience=_Experience().generate_experience(
                "request", (), type("T", (), {"now": 0})()
            ),
            state=_state(),
        )

        self.assertEqual(selection.selected_candidate_id, "ask-authority")
        self.assertEqual(
            selection.action_payload,
            "Which source should we treat as authoritative for this comparison?",
        )
        self.assertEqual(len(selection.intent_candidates), 2)
        self.assertEqual(selection.intent_candidates[0].candidate_id, "investigate-known-evidence")
        self.assertEqual(
            selection.candidate_preference_order,
            ("ask-authority", "investigate-known-evidence"),
        )
        self.assertNotIn("score", selection.intent_candidates[0].__dataclass_fields__)
        self.assertEqual(selection.compute_route, "DIRECT")
        self.assertIn("already clear", selection.compute_rationale)
        self.assertIn("variable list", backend.calls[0][0])
        self.assertEqual(backend.calls[0][2], 2_048)

    def test_frozen_controller_canonicalizes_model_tool_json_payload(self):
        temporal = _clock().sample(-1)
        tool_payload = {
            "limit": 3,
            "query": "temporal metacognition",
            "sources": ["general", "scholarly"],
        }
        backend = _ControllerBackend(
            {
                "selected_affordance_id": "tool.web_research",
                "action_payload": json.dumps(tool_payload, indent=2),
                "state_assessment": "A bounded unresolved question warrants research.",
                "resolution_target": "Retrieve current source-bound observations.",
                "expected_state_delta": "New evidence will refine the current world model.",
                "evidence_refs": [temporal.sample_ref],
                "selected_candidate_id": "research-current-evidence",
                "candidate_preference_order": ["research-current-evidence"],
                "affordance_preference_order": ["tool.web_research"],
                "selection_basis": "A reversible read can reduce the current evidence gap.",
                "intent_candidates": [
                    {
                        "candidate_id": "research-current-evidence",
                        "affordance_id": "tool.web_research",
                        "proposed_action": json.dumps(tool_payload, indent=2),
                        "desired_state_change": "Replace an open question with observations.",
                        "rationale": "Current sources can bear on the unresolved question.",
                        "predicted_consequences": ["Relevant search results become available."],
                        "unknowns": ["Whether current sources agree."],
                        "reversibility": "REVERSIBLE",
                        "required_capability_keys": [],
                        "evidence_refs": [temporal.sample_ref],
                        "reasons_for": ["The operation is bounded and read-only."],
                        "reasons_against": ["Search snippets may be incomplete."],
                    }
                ],
            }
        )
        selection = FrozenModelAffordanceController(
            backend,
            qualification_ref=QUALIFICATION,
            require_intent_candidates=True,
            maximum_output_tokens=2_048,
        ).select(
            observation=CycleObservation("scheduler:web", "SCHEDULER", ""),
            temporal=temporal,
            affordances=(
                Affordance(
                    "tool.web_research",
                    "ACT",
                    "Research with canonical JSON.",
                    "external.readonly.web",
                    external_effect=True,
                ),
            ),
            memories=(),
            experience=_Experience().generate_experience(
                "research", (), type("T", (), {"now": 0})()
            ),
            state=_state(),
        )
        expected = json.dumps(tool_payload, separators=(",", ":"), sort_keys=True)
        self.assertEqual(selection.action_payload, expected)
        self.assertEqual(selection.intent_candidates[0].proposed_action, expected)

    def test_unselected_structured_alternative_does_not_force_second_model_pass(self):
        temporal = _clock().sample(-1)
        backend = _ControllerBackend(
            {
                "selected_affordance_id": "cortex.respond",
                "action_payload": "Answer from the evidence already present.",
                "requires_extended_deliberation": False,
                "compute_rationale": "The selected response is already grounded.",
                "state_assessment": "The answer is available in current evidence.",
                "resolution_target": "Answer without an unnecessary operation.",
                "expected_state_delta": "The owner receives the grounded answer.",
                "evidence_refs": [temporal.sample_ref],
                "selected_candidate_id": "answer-now",
                "candidate_preference_order": ["answer-now", "inspect-library"],
                "affordance_preference_order": [
                    "cortex.respond",
                    "internal.library-read",
                ],
                "selection_basis": "Existing evidence resolves the request; library inspection would add no relevant fact.",
                "intent_candidates": [
                    {
                        "candidate_id": "answer-now",
                        "affordance_id": "cortex.respond",
                        "proposed_action": "Answer from the evidence already present.",
                        "desired_state_change": "Provide the grounded answer.",
                        "rationale": "No evidence gap remains.",
                        "predicted_consequences": ["The owner receives an answer."],
                        "unknowns": [],
                        "reversibility": "REVERSIBLE",
                        "required_capability_keys": [],
                        "evidence_refs": [temporal.sample_ref],
                        "reasons_for": ["It is sufficient."],
                        "reasons_against": [],
                    },
                    {
                        "candidate_id": "inspect-library",
                        "affordance_id": "internal.library-read",
                        "proposed_action": "List the library if it later becomes relevant.",
                        "desired_state_change": "Observe the catalog.",
                        "rationale": "This is an unselected possibility.",
                        "predicted_consequences": ["The catalog would be observed."],
                        "unknowns": [],
                        "reversibility": "REVERSIBLE",
                        "required_capability_keys": [],
                        "evidence_refs": [temporal.sample_ref],
                        "reasons_for": [],
                        "reasons_against": ["It is irrelevant now."],
                    },
                ],
            }
        )
        selection = FrozenModelAffordanceController(
            backend,
            draft_backend=backend,
            qualification_ref=QUALIFICATION,
            require_intent_candidates=True,
            maximum_output_tokens=2_048,
            maximum_draft_output_tokens=2_048,
        ).select(
            observation=CycleObservation(
                "scheduler:avoid-alternative-overvalidation",
                "SCHEDULER",
                "Answer the bounded question.",
            ),
            temporal=temporal,
            affordances=(
                Affordance(
                    "cortex.respond", "ACT", "Respond.", "internal.cognition"
                ),
                Affordance(
                    "internal.library-read",
                    "ACT",
                    "Read with canonical JSON.",
                    "internal.cognition",
                ),
            ),
            memories=(),
            experience=_Experience().generate_experience(
                "request", (), type("T", (), {"now": 0})()
            ),
            state=_state(),
        )

        self.assertEqual(selection.compute_route, "DIRECT")
        self.assertEqual(selection.selected_affordance_id, "cortex.respond")
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(
            selection.intent_candidates[1].proposed_action,
            "List the library if it later becomes relevant.",
        )

    def test_selected_capability_module_reaches_cortex_and_receives_outcome(self):
        capability_key = "capability.synthetic-transfer"
        initial = json.loads(_state())
        first_capability = {
            "capability_key": capability_key,
            "revision_index": 1,
            "supersedes_capability_ref": None,
            "procedure": "Use the first transfer procedure.",
            "applicability": "Comparable bounded cases.",
            "limits": "Unverified outside the source form.",
        }
        old_ref = "sha256:" + hashlib.sha256(
            json.dumps(
                first_capability, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ).encode()
        ).hexdigest()
        first_capability["capability_ref"] = old_ref
        revised_capability = {
            "capability_key": capability_key,
            "revision_index": 2,
            "supersedes_capability_ref": old_ref,
            "procedure": "Use the revised transfer procedure and verify the receipt.",
            "applicability": "Comparable bounded cases with an observable receipt.",
            "limits": "Re-evaluate when constraints differ.",
        }
        active_ref = "sha256:" + hashlib.sha256(
            json.dumps(
                revised_capability, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ).encode()
        ).hexdigest()
        revised_capability["capability_ref"] = active_ref
        initial["capability_evidence"] = [first_capability, revised_capability]
        initial_state = json.dumps(
            initial, separators=(",", ":"), sort_keys=True
        ).encode()
        backend = _ControllerBackend(
            {
                "selected_affordance_id": "cortex.respond",
                "action_payload": "Use the selected capability on this bounded case.",
                "requires_extended_deliberation": False,
                "compute_rationale": "The applicable revised procedure and check are clear.",
                "state_assessment": "One revised capability applies to the bounded request.",
                "resolution_target": "Test transfer using the active revision.",
                "expected_state_delta": "The observable receipt will test the selected module.",
                "evidence_refs": [active_ref],
                "selected_candidate_id": "apply-revised-transfer",
                "candidate_preference_order": ["apply-revised-transfer"],
                "affordance_preference_order": ["cortex.respond"],
                "selection_basis": "The latest applicable revision includes an explicit observable check.",
                "intent_candidates": [
                    {
                        "candidate_id": "apply-revised-transfer",
                        "affordance_id": "cortex.respond",
                        "proposed_action": "Use the selected capability on this bounded case.",
                        "desired_state_change": "Test whether the revised procedure transfers.",
                        "rationale": "The active revision is applicable and testable.",
                        "predicted_consequences": ["The receipt will expose transfer success."],
                        "unknowns": ["Whether the new form preserves the prior constraints."],
                        "reversibility": "REVERSIBLE",
                        "required_capability_keys": [capability_key],
                        "evidence_refs": [active_ref],
                        "reasons_for": ["It is the latest applicable revision."],
                        "reasons_against": ["Transfer remains uncertain until observed."],
                    }
                ],
            }
        )

        class ContextualCortex(_Cortex):
            def execute_with_context(
                self, request, experience, memories, temporal_now, cognitive_state
            ):
                self.calls.append(
                    (request, experience, tuple(memories), temporal_now, cognitive_state)
                )
                return ExecutionReceipt(
                    "sha256:" + "c" * 64, "A bounded public response.", "COMPLETED"
                )

        class CapabilityExperience(_Experience):
            def consolidate_with_capabilities(
                self,
                request,
                experience,
                reflection,
                consequence,
                selected_capability_modules,
            ):
                legacy = json.loads(
                    super().consolidate(request, experience, reflection, consequence)
                )
                return json.dumps(
                    {
                        "lesson": legacy["lesson"],
                        "world_model": legacy["world_model"],
                        "self_model": legacy["self_model"],
                        "focus": legacy["focus"],
                        "unfinished_patterns": [],
                        "next_internal_request": "",
                        "capability_proposal": {
                            "retain": True,
                            "revision_target_key": selected_capability_modules[0][
                                "capability_key"
                            ],
                            "procedure": "Use the revised procedure and verify two constraints.",
                            "applicability": "Comparable cases with observable constraints.",
                            "limits": "Transfer remains uncertain outside tested forms.",
                            "rationale": "The observed receipt justifies revising the used module.",
                        },
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )

        cortex = ContextualCortex()
        adapter = HigherLevelAutonomyCycleAdapter(
            memory=_Memory(),
            experience_model=CapabilityExperience(),
            controller=FrozenModelAffordanceController(
                backend,
                draft_backend=backend,
                qualification_ref=QUALIFICATION,
                require_intent_candidates=True,
                maximum_output_tokens=2_048,
                maximum_draft_output_tokens=2_048,
            ),
        )
        registry = DynamicAffordanceRegistry()
        registry.register(
            Affordance("cortex.respond", "ACT", "Run cortex.", "internal.cognition"),
            HigherLevelCortexAffordanceExecutor(cortex, _consequence),
        )
        with tempfile.TemporaryDirectory() as directory:
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "capability-workspace.sqlite3",
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=initial_state,
                clock=_clock(),
                cycle=adapter,
                affordances=registry,
            )
            result = supervisor.scheduler_tick("capability-workspace:one")
            self.assertEqual(result.status, "COMMITTED")
            controller_input = json.loads(backend.calls[0][1])
            modules = controller_input["capability_modules"]
            self.assertEqual(len(modules), 1)
            self.assertEqual(modules[0]["active_capability_ref"], active_ref)
            workspace = cortex.calls[0][4]
            self.assertEqual(
                workspace["capability_workspace"]["selected_capability_keys"],
                [capability_key],
            )
            self.assertEqual(
                workspace["capability_modules"][0]["active_capability_ref"],
                active_ref,
            )
            learned = json.loads(supervisor.state_bytes())
            use = learned["capability_use_evidence"][-1]
            self.assertEqual(use["capability_keys"], [capability_key])
            self.assertEqual(
                use["epistemic_status"],
                "MODEL_SELECTED_CAPABILITY_WITH_OBSERVED_CONSEQUENCE",
            )
            self.assertEqual(use["observed_consequence"]["safety"], 1.0)
            self.assertEqual(
                use["capability_versions"][0]["capability_ref"], active_ref
            )
            revised = learned["capability_evidence"][-1]
            self.assertEqual(revised["capability_key"], capability_key)
            self.assertEqual(revised["revision_index"], 3)
            self.assertEqual(revised["supersedes_capability_ref"], active_ref)

    def test_model_can_decline_capability_retention_after_observed_outcome(self):
        class DecliningExperience(_Experience):
            def consolidate_with_capabilities(
                self,
                request,
                experience,
                reflection,
                consequence,
                selected_capability_modules,
            ):
                legacy = json.loads(
                    super().consolidate(request, experience, reflection, consequence)
                )
                return json.dumps(
                    {
                        "lesson": legacy["lesson"],
                        "world_model": legacy["world_model"],
                        "self_model": legacy["self_model"],
                        "focus": legacy["focus"],
                        "unfinished_patterns": [],
                        "next_internal_request": "",
                        "capability_proposal": {
                            "retain": False,
                            "revision_target_key": None,
                            "procedure": "",
                            "applicability": "",
                            "limits": "",
                            "rationale": "This single outcome does not support a reusable procedure.",
                        },
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )

        with tempfile.TemporaryDirectory() as directory:
            adapter = HigherLevelAutonomyCycleAdapter(
                memory=_Memory(),
                experience_model=DecliningExperience(),
                controller=OutcomeUtilityAffordanceController(
                    qualification_ref=QUALIFICATION
                ),
            )
            registry = DynamicAffordanceRegistry()
            registry.register(
                Affordance("cortex.respond", "ACT", "Run cortex.", "internal.cognition"),
                HigherLevelCortexAffordanceExecutor(_Cortex(), _consequence),
            )
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "decline-capability.sqlite3",
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=_state(),
                clock=_clock(),
                cycle=adapter,
                affordances=registry,
            )
            result = supervisor.human_ingress(
                "capability:decline", "Evaluate one non-reusable case."
            )
            self.assertEqual(result.status, "COMMITTED")
            learned = json.loads(supervisor.state_bytes())
            self.assertNotIn("capability_evidence", learned)
            decision = learned["capability_consolidation_evidence"][-1]
            self.assertEqual(
                decision["epistemic_status"],
                "MODEL_CAPABILITY_PROPOSAL_WITH_OBSERVED_CONSEQUENCE",
            )
            self.assertFalse(decision["proposal"]["retain"])

    def test_model_authored_new_capability_gets_opaque_genesis_identity(self):
        class CreatingExperience(_Experience):
            def consolidate_with_capabilities(
                self,
                request,
                experience,
                reflection,
                consequence,
                selected_capability_modules,
            ):
                self.assert_no_selected = selected_capability_modules
                legacy = json.loads(
                    super().consolidate(request, experience, reflection, consequence)
                )
                return json.dumps(
                    {
                        "lesson": legacy["lesson"],
                        "world_model": legacy["world_model"],
                        "self_model": legacy["self_model"],
                        "focus": legacy["focus"],
                        "unfinished_patterns": [],
                        "next_internal_request": "",
                        "capability_proposal": {
                            "retain": True,
                            "revision_target_key": None,
                            "procedure": "Compare constraints, act, and verify the receipt.",
                            "applicability": "New bounded cases with observable outcomes.",
                            "limits": "The procedure has only one attributable observation.",
                            "rationale": "The observed sequence supports one distinct reusable proposal.",
                        },
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )

        with tempfile.TemporaryDirectory() as directory:
            experience = CreatingExperience()
            adapter = HigherLevelAutonomyCycleAdapter(
                memory=_Memory(),
                experience_model=experience,
                controller=OutcomeUtilityAffordanceController(
                    qualification_ref=QUALIFICATION
                ),
            )
            registry = DynamicAffordanceRegistry()
            registry.register(
                Affordance("cortex.respond", "ACT", "Run cortex.", "internal.cognition"),
                HigherLevelCortexAffordanceExecutor(_Cortex(), _consequence),
            )
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "create-capability.sqlite3",
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=_state(),
                clock=_clock(),
                cycle=adapter,
                affordances=registry,
            )
            result = supervisor.human_ingress(
                "capability:create", "Learn from one attributable bounded case."
            )
            self.assertEqual(result.status, "COMMITTED")
            self.assertEqual(experience.assert_no_selected, [])
            learned = json.loads(supervisor.state_bytes())
            capability = learned["capability_evidence"][-1]
            self.assertTrue(capability["capability_key"].startswith("learned:sha256:"))
            self.assertEqual(capability["revision_index"], 1)
            self.assertIsNone(capability["supersedes_capability_ref"])
            evidence = learned["capability_consolidation_evidence"][-1]
            self.assertEqual(
                evidence["resolved_capability_key"], capability["capability_key"]
            )

            class CapabilityBackend:
                def __init__(self):
                    self.projected = []

                def search(self, request, *, limit):
                    return ()

                def project(self, **value):
                    self.projected.append(value)
                    return "capability-index:one"

            capability_backend = CapabilityBackend()
            projector = CapabilityAwareConsolidationProjector(
                SemanticMemoryConsolidationProjector(adapter.memory),
                supervisor,
                capability_backend,
            )
            self.assertEqual(len(supervisor.pending_projections()), 1)
            supervisor.retry_pending_projections(projector)
            self.assertEqual(supervisor.pending_projections(), ())
            self.assertEqual(len(capability_backend.projected), 1)
            projected = capability_backend.projected[0]
            self.assertEqual(projected["record_ref"], capability["capability_ref"])
            self.assertEqual(
                projected["provenance_refs"],
                (capability["capability_ref"], supervisor.state_head().state_ref),
            )
            self.assertNotIn("latest_use_evidence", projected["content"])

    def test_completed_without_consequence_updates_only_unverified_working_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory, experience, cortex = _Memory(), _Experience(), _Cortex()
            adapter = HigherLevelAutonomyCycleAdapter(
                memory=memory,
                experience_model=experience,
                controller=OutcomeUtilityAffordanceController(
                    qualification_ref=QUALIFICATION
                ),
            )
            registry = DynamicAffordanceRegistry()
            registry.register(
                Affordance("cortex.respond", "ACT", "Run cortex.", "internal.cognition"),
                HigherLevelCortexAffordanceExecutor(cortex),
            )
            initial = _state()
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "unevaluated.sqlite3",
                genesis=JennyGenesis.owner_approved(created_at_utc="2026-09-02T12:00:00Z"),
                initial_state=initial,
                clock=_clock(),
                cycle=adapter,
                affordances=registry,
            )
            result = supervisor.human_ingress("unevaluated:one", "Answer without fake credit.")
            self.assertEqual(result.status, "COMMITTED")
            learned = json.loads(supervisor.state_bytes())
            self.assertEqual(learned["affordance_utility"], {})
            self.assertEqual(learned["memory_utility"], {})
            self.assertEqual(learned["next_internal_request"], "")
            self.assertEqual(
                learned["initiative_state"]["epistemic_status"],
                "MODEL_PROPOSAL_WITHOUT_OUTCOME",
            )
            self.assertEqual(experience.reflections, 0)
            self.assertEqual(experience.last_temporal.trusted_utc, "2026-09-02T12:00:00.001000Z")
            self.assertEqual(experience.last_temporal.local_timezone, "America/New_York")
            self.assertEqual(experience.last_temporal.local_utc_offset_seconds, -14400)
            second = supervisor.human_ingress(
                "unevaluated:two", "Revisit the bounded answer."
            )
            self.assertEqual(second.status, "COMMITTED")
            relation = experience.last_temporal.landmark_relations[0][1]
            self.assertIn("event_distance=0", relation)
            self.assertIn("elapsed_seconds=60.", relation)
            self.assertNotIn("semantic_age", relation)

    def test_source_bound_observation_supports_qualitative_situated_assessment_without_credit(
        self,
    ) -> None:
        source_ref = "sha256:" + "1" * 64
        artifact_ref = "sha256:" + "2" * 64
        source_evidence_ref = "sha256:" + "3" * 64
        captured = {}

        def observe(request):
            observation = ObservableConsequence(
                request_ref=request.idempotency_key,
                source_kind="TEST",
                source_ref=source_ref,
                observation_json='{"result":"bounded-observation"}',
                artifact_refs=(artifact_ref,),
                evidence_refs=(source_evidence_ref,),
            )
            receipt = AffordanceReceipt(
                "COMPLETED",
                "One source-bound result was observed.",
                (),
                observation,
            )
            captured.update(request=request, observation=observation, receipt=receipt)
            return receipt

        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "qualitative-observation.sqlite3"
            memory = _Memory()
            experience = _ObservedOutcomeExperience()
            controller = _ObservedOutcomeController()
            adapter = HigherLevelAutonomyCycleAdapter(
                memory=memory,
                experience_model=experience,
                controller=controller,
            )
            registry = DynamicAffordanceRegistry()
            registry.register(
                Affordance(
                    "test.observe",
                    "ACT",
                    "Observe one bounded result.",
                    "internal.cognition",
                ),
                observe,
                observable_source_ref=source_ref,
                observable_source_kind="TEST",
            )
            supervisor = PersistentAutonomySupervisor(
                state_path,
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=_state(),
                clock=_clock(),
                cycle=adapter,
                affordances=registry,
            )
            parent_head = supervisor.state_head()
            result = supervisor.scheduler_tick("observed:qualitative")

            self.assertEqual(result.status, "COMMITTED")
            self.assertEqual(len(experience.assessment_calls), 1)

            call = experience.assessment_calls[0]
            self.assertEqual(
                call["prediction_catalog"],
                {
                    "experience.predicted-consequence": (
                        "The objective receipt will reduce uncertainty."
                    ),
                    "choice.expected-state-delta": (
                        "The exact observation will provisionally update situated state."
                    ),
                    "selected-candidate.prediction-1": (
                        "The observer will return a source-bound result."
                    ),
                    "selected-candidate.prediction-2": (
                        "The result may leave its objective value unresolved."
                    ),
                },
            )
            evidence_catalog = call["evidence_catalog"]
            observation = captured["observation"]
            receipt = captured["receipt"]
            self.assertEqual(
                evidence_catalog["observable-consequence"],
                observation.observation_ref,
            )
            self.assertEqual(evidence_catalog["observable-source"], source_ref)
            self.assertEqual(evidence_catalog["receipt"], receipt.receipt_ref)
            self.assertEqual(evidence_catalog["parent-state"], parent_head.state_ref)
            self.assertEqual(evidence_catalog["artifact-1"], artifact_ref)
            self.assertEqual(
                evidence_catalog["source-evidence-1"], source_evidence_ref
            )
            self.assertEqual(
                evidence_catalog["temporal"],
                "sha256:"
                + hashlib.sha256(
                    json.dumps(
                        call["temporal"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode()
                ).hexdigest(),
            )

            learned = json.loads(supervisor.state_bytes())
            situated = learned["situated_state"]
            self.assertEqual(
                situated["epistemic_status"],
                "MODEL_INFERENCE_FROM_SOURCE_BOUND_OBSERVATION_WITHOUT_VALUE_CREDIT",
            )
            self.assertIn(
                "The bounded test world returned one source-bound result.",
                json.dumps(situated["world_model"]),
            )
            self.assertIn(
                "The selected observation procedure produced attributable evidence.",
                json.dumps(situated["self_model"]),
            )
            self.assertIn(
                "Resolve the remaining uncertainty without treating observation as reward.",
                json.dumps(situated["focus"]),
            )
            self.assertEqual(situated["observation_ref"], observation.observation_ref)
            self.assertEqual(
                learned["last_outcome"]["epistemic_status"],
                "SOURCE_BOUND_OBSERVATION_WITH_QUALITATIVE_MODEL_INTERPRETATION",
            )
            self.assertEqual(
                learned["last_outcome"]["observation_ref"], observation.observation_ref
            )
            ranking = learned["intent_ranking_evidence"][-1]
            self.assertEqual(
                ranking["epistemic_status"],
                "COMPARATIVE_CHOICE_WITH_QUALITATIVE_OUTCOME_ASSESSMENT",
            )
            self.assertEqual(ranking["observation_ref"], observation.observation_ref)
            self.assertEqual(
                {item["prediction_key"]: item["relation"] for item in ranking["prediction_assessments"]},
                {
                    "experience.predicted-consequence": "SUPPORTED",
                    "choice.expected-state-delta": "PARTIAL",
                    "selected-candidate.prediction-1": "CONTRADICTED",
                    "selected-candidate.prediction-2": "UNRESOLVED",
                },
            )
            self.assertEqual(controller.update_calls, 0)
            self.assertEqual(learned["affordance_utility"], {})
            self.assertEqual(learned["memory_utility"], {})
            self.assertNotIn("affordance_outcome_profiles", learned)
            self.assertNotIn("capability_evidence", learned)
            self.assertNotIn("procedure_proposal", situated)
            self.assertEqual(memory.stores, [])
            committed_state = supervisor.state_bytes()
            committed_head = supervisor.state_head()

            restarted = PersistentAutonomySupervisor(
                state_path,
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=_state(),
                clock=_clock(),
                cycle=adapter,
                affordances=registry,
            )
            replay = restarted.scheduler_tick("observed:qualitative")
            self.assertEqual(replay.episode_ref, result.episode_ref)
            self.assertEqual(restarted.state_bytes(), committed_state)
            self.assertEqual(restarted.state_head(), committed_head)
            self.assertEqual(len(experience.assessment_calls), 1)

    def test_source_bound_outcome_changes_next_situated_prediction(self) -> None:
        source_ref = "sha256:" + "7" * 64

        def observe(request):
            observation = ObservableConsequence(
                request_ref=request.idempotency_key,
                source_kind="TEST",
                source_ref=source_ref,
                observation_json='{"result":"bounded-observation"}',
            )
            return AffordanceReceipt(
                "COMPLETED", "One source-bound result was observed.", (), observation
            )

        with tempfile.TemporaryDirectory() as directory:
            experience = _SituatedObservedOutcomeExperience()
            adapter = HigherLevelAutonomyCycleAdapter(
                memory=_Memory(),
                experience_model=experience,
                controller=_ObservedOutcomeController(),
            )
            registry = DynamicAffordanceRegistry()
            registry.register(
                Affordance(
                    "test.observe",
                    "ACT",
                    "Observe one bounded result.",
                    "internal.cognition",
                ),
                observe,
                observable_source_ref=source_ref,
                observable_source_kind="TEST",
            )
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "situated-prediction.sqlite3",
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=_state(),
                clock=_clock(),
                cycle=adapter,
                affordances=registry,
            )

            first = supervisor.scheduler_tick("observed:situated:first")
            second = supervisor.scheduler_tick("observed:situated:second")

            self.assertEqual(first.status, "COMMITTED")
            self.assertEqual(second.status, "COMMITTED")
            self.assertEqual(len(experience.situated_generation_calls), 2)
            first_state = experience.situated_generation_calls[0]["cognitive_state"]
            second_state = experience.situated_generation_calls[1]["cognitive_state"]
            self.assertEqual(first_state["outcome_assessment_evidence"], [])
            self.assertEqual(len(second_state["outcome_assessment_evidence"]), 1)
            self.assertEqual(
                second_state["situated_state"]["world_model"],
                ["The bounded test world returned one source-bound result."],
            )
            self.assertEqual(
                experience.assessment_calls[1]["prediction_catalog"][
                    "experience.predicted-consequence"
                ],
                "A second observation should test the prior contradicted prediction.",
            )

    def test_quiescent_state_can_form_and_execute_model_authored_initiative(self) -> None:
        source_ref = "sha256:" + "6" * 64

        def observe(request):
            observation = ObservableConsequence(
                request_ref=request.idempotency_key,
                source_kind="TEST",
                source_ref=source_ref,
                observation_json='{"result":"new-evidence"}',
            )
            return AffordanceReceipt(
                "COMPLETED", "A bounded observation was returned.", (), observation
            )

        state = json.dumps(
            {
                "affordance_utility": {},
                "memory_utility": {},
                "next_internal_request": "",
                "affordance_signals": {},
                "motivation_weights": {},
                "situated_state": {
                    "world_model": ["One relevant question remains unresolved."],
                    "self_model": ["A bounded observer is available."],
                    "focus": ["Seek evidence for the unresolved question."],
                    "unfinished_patterns": ["Which observation resolves the question?"],
                },
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        with tempfile.TemporaryDirectory() as directory:
            experience = _ToolBlindInitiativeExperience()
            controller = _CapturingObservedOutcomeController()
            adapter = HigherLevelAutonomyCycleAdapter(
                memory=_Memory(),
                experience_model=experience,
                controller=controller,
            )
            registry = DynamicAffordanceRegistry()
            registry.register(
                Affordance(
                    "test.observe",
                    "ACT",
                    "Observe one bounded result.",
                    "internal.cognition",
                ),
                observe,
                observable_source_ref=source_ref,
                observable_source_kind="TEST",
            )
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "initiative.sqlite3",
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=state,
                clock=_clock(),
                cycle=adapter,
                affordances=registry,
            )
            before_head = supervisor.state_head()

            result = supervisor.scheduler_tick("initiative:wake")

            self.assertEqual(result.status, "COMMITTED")
            self.assertEqual(len(experience.formation_calls), 1)
            formation_input = experience.formation_calls[0]
            self.assertNotIn("affordances", formation_input)
            self.assertNotIn(
                "trusted_composition", formation_input["cognitive_state"]
            )
            self.assertIn(
                "state-situated-state", formation_input["evidence_catalog"]
            )
            self.assertEqual(len(controller.observations), 1)
            self.assertEqual(controller.observations[0].source, "SCHEDULER")
            self.assertEqual(
                controller.observations[0].content,
                "Revisit the specific unresolved state evidence.",
            )
            episode = supervisor.episode_item(result.episode_ref)
            episode_payload = json.loads(episode.payload_json)
            choice_context = json.loads(
                episode_payload["choice"]["context_json"]
            )
            self.assertEqual(
                choice_context["autonomous_initiative"]["epistemic_status"],
                "MODEL_AUTHORED_STATE_DEPENDENT_INITIATIVE",
            )
            formation = choice_context["autonomous_initiative"]["formation"]
            self.assertEqual(
                formation["formation_mode"], "TOOL_BLIND_STATE_MEMORY_TIME"
            )
            self.assertEqual(
                formation["formation_evidence_keys"],
                ["state-situated-state"],
            )
            self.assertEqual(
                formation["moving_origin_ordinal"],
                before_head.moving_origin_ordinal,
            )
            self.assertEqual(
                result.moving_origin_ordinal,
                formation["moving_origin_ordinal"] + 1,
            )
            unhashed = dict(formation)
            formation_ref = unhashed.pop("formation_ref")
            self.assertEqual(formation_ref, content_ref(unhashed))
            self.assertEqual(result.moving_origin_ordinal, 0)
            learned = json.loads(supervisor.state_bytes())
            commitment = learned["current_autonomous_commitment"]
            self.assertEqual(
                commitment["contract"],
                "jenny.autonomous-commitment-state.v1",
            )
            self.assertEqual(
                commitment["epistemic_status"],
                "MODEL_AUTHORED_COMMITMENT_WITH_ATTEMPT_EVIDENCE",
            )
            self.assertEqual(
                commitment["commitment"], formation["proposal"]["commitment"]
            )
            self.assertEqual(commitment["formation_ref"], formation_ref)
            self.assertEqual(
                commitment["choice_ref"],
                "sha256:" + hashlib.sha256(
                    json.dumps(
                        episode_payload["choice"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest(),
            )
            self.assertEqual(
                commitment["receipt_ref"],
                "sha256:" + hashlib.sha256(
                    json.dumps(
                        episode_payload["receipt"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest(),
            )
            self.assertEqual(commitment["moving_origin_ordinal"], 0)
            self.assertEqual(len(learned["autonomous_commitment_history"]), 1)

    def test_active_commitment_can_wait_without_selection_commit_or_repeat(
        self,
    ) -> None:
        source_ref = "sha256:" + "6" * 64
        executions = []

        def observe(request):
            executions.append(request)
            observation = ObservableConsequence(
                request_ref=request.idempotency_key,
                source_kind="TEST",
                source_ref=source_ref,
                observation_json='{"result":"bounded-observation"}',
            )
            return AffordanceReceipt(
                "COMPLETED", "A bounded observation was returned.", (), observation
            )

        registry = DynamicAffordanceRegistry()
        registry.register(
            Affordance(
                "test.observe",
                "ACT",
                "Observe one bounded result.",
                "internal.cognition",
            ),
            observe,
            observable_source_ref=source_ref,
            observable_source_kind="TEST",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "event-driven-commitment.sqlite3"
            active_adapter = HigherLevelAutonomyCycleAdapter(
                memory=_Memory(),
                experience_model=_ToolBlindInitiativeExperience(),
                controller=_CapturingObservedOutcomeController(),
            )
            active = PersistentAutonomySupervisor(
                path,
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=_tool_blind_grounded_state(),
                clock=_clock(),
                cycle=active_adapter,
                affordances=registry,
            )
            committed = active.scheduler_tick("commitment:initial-attempt")
            self.assertEqual(committed.status, "COMMITTED")
            self.assertEqual(len(executions), 1)
            committed_state = active.state_bytes()
            committed_head = active.state_head()
            self.assertEqual(
                json.loads(committed_state)["current_autonomous_commitment"][
                    "commitment"
                ]["status"],
                "ACTIVE",
            )

            waiting_experience = _WaitingToolBlindInitiativeExperience()
            waiting_controller = _CapturingObservedOutcomeController()
            waiting = PersistentAutonomySupervisor(
                path,
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=b"ignored-existing-state",
                clock=_clock(),
                cycle=HigherLevelAutonomyCycleAdapter(
                    memory=_Memory(),
                    experience_model=waiting_experience,
                    controller=waiting_controller,
                ),
                affordances=registry,
            )

            first_wait = waiting.scheduler_tick("commitment:wait-once")
            repeated_wait = waiting.scheduler_tick("commitment:wait-deduped")

            self.assertEqual(first_wait.status, "QUIESCENT")
            self.assertEqual(repeated_wait.status, "QUIESCENT")
            self.assertIn("no canonical state or event change", repeated_wait.detail)
            self.assertEqual(len(waiting_experience.formation_calls), 1)
            self.assertEqual(waiting_controller.observations, [])
            self.assertEqual(len(executions), 1)
            self.assertEqual(waiting.state_bytes(), committed_state)
            self.assertEqual(waiting.state_head(), committed_head)
            self.assertEqual(
                waiting.cycle.last_autonomous_formation["proposal"][
                    "actionability"
                ],
                "WAIT_FOR_CHANGE",
            )

            restarted_experience = _WaitingToolBlindInitiativeExperience()
            restarted_controller = _CapturingObservedOutcomeController()
            restarted = PersistentAutonomySupervisor(
                path,
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=b"ignored-existing-state",
                clock=_clock(),
                cycle=HigherLevelAutonomyCycleAdapter(
                    memory=_Memory(),
                    experience_model=restarted_experience,
                    controller=restarted_controller,
                ),
                affordances=registry,
            )
            after_restart = restarted.scheduler_tick(
                "commitment:wait-after-restart"
            )
            self.assertEqual(after_restart.status, "QUIESCENT")
            self.assertEqual(restarted_experience.formation_calls, [])
            self.assertEqual(restarted_controller.observations, [])
            self.assertEqual(restarted.state_head(), committed_head)

            new_event = restarted.agent_ingress(
                "agent:new-commitment-evidence",
                "One new attributed observation changed the canonical state.",
            )
            self.assertEqual(new_event.status, "COMMITTED")
            self.assertEqual(len(restarted_controller.observations), 1)
            self.assertEqual(len(executions), 2)
            post_human_state = restarted.state_bytes()
            post_human_head = restarted.state_head()

            invited_wait = restarted.scheduler_tick(
                "commitment:wait-after-new-event"
            )
            self.assertEqual(invited_wait.status, "QUIESCENT")
            self.assertEqual(len(restarted_experience.formation_calls), 1)
            self.assertEqual(len(restarted_controller.observations), 1)
            self.assertEqual(len(executions), 2)
            self.assertEqual(restarted.state_bytes(), post_human_state)
            self.assertEqual(restarted.state_head(), post_human_head)

    def test_tool_blind_target_whitespace_is_canonicalized_before_scheduler_handoff(
        self,
    ) -> None:
        state = _tool_blind_grounded_state()
        clock = _clock()
        temporal = clock.sample(-1)
        affordances = (
            Affordance(
                "test.observe",
                "ACT",
                "Observe one bounded result.",
                "internal.cognition",
            ),
        )
        experience = _ToolBlindInitiativeExperience(
            request="  Revisit the specific unresolved state evidence.\n"
        )
        controller = _CapturingObservedOutcomeController()
        adapter = HigherLevelAutonomyCycleAdapter(
            memory=_Memory(),
            experience_model=experience,
            controller=controller,
        )

        request = adapter.propose_autonomous_request(
            state=state,
            temporal=temporal,
            affordances=affordances,
        )
        self.assertEqual(
            request, "Revisit the specific unresolved state evidence."
        )

        choice = adapter.choose(
            observation=CycleObservation(
                "tool-blind:canonical-whitespace", "SCHEDULER", request
            ),
            temporal=temporal,
            affordances=affordances,
            state=state,
        )

        self.assertEqual(len(controller.observations), 1)
        self.assertEqual(controller.observations[0].content, request)
        initiative = json.loads(choice.context_json)["autonomous_initiative"]
        self.assertEqual(
            set(initiative), {"epistemic_status", "formation"}
        )
        self.assertEqual(
            initiative["formation"]["proposal"]["internal_request"], request
        )
        unhashed = dict(initiative["formation"])
        formation_ref = unhashed.pop("formation_ref")
        self.assertEqual(formation_ref, content_ref(unhashed))

    def test_tampered_pending_formation_fails_before_selection_permission_or_execution(
        self,
    ) -> None:
        state = _tool_blind_grounded_state()
        experience = _ToolBlindInitiativeExperience()
        controller = _CapturingObservedOutcomeController()
        executor = _CountingAuthoredArtifactExecutor()
        permission_gate = _CountingPermissionGate()
        adapter = HigherLevelAutonomyCycleAdapter(
            memory=_Memory(),
            experience_model=experience,
            controller=controller,
        )
        original_propose = adapter.propose_autonomous_request

        def propose_then_tamper(**kwargs):
            request = original_propose(**kwargs)
            pending = adapter._pending_autonomous_initiative
            self.assertIsNotNone(pending)
            pending["formation"]["proposal"]["internal_request"] = (
                "A different target injected after formation."
            )
            return request

        adapter.propose_autonomous_request = propose_then_tamper
        with tempfile.TemporaryDirectory() as directory:
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "tool-blind-tampered-formation.sqlite3",
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=state,
                clock=_artifact_test_clock(),
                cycle=adapter,
                affordances=_authored_artifact_registry(executor),
                permission_gate=permission_gate,
            )
            before_state = supervisor.state_bytes()
            before_head = supervisor.state_head()

            with self.assertRaisesRegex(
                ValueError, "autonomous target formation_ref differs"
            ):
                supervisor.scheduler_tick("tool-blind:tampered-formation")

            self.assertEqual(controller.observations, [])
            self.assertEqual(permission_gate.calls, 0)
            self.assertEqual(executor.calls, 0)
            self.assertEqual(supervisor.state_bytes(), before_state)
            self.assertEqual(supervisor.state_head(), before_head)
            self.assertIsNone(supervisor.pending_bytes())
            self.assertIsNone(adapter._pending_autonomous_initiative)

            # A consumed invalid handoff cannot be replayed on a later wake.
            adapter.propose_autonomous_request = original_propose
            experience.request = ""
            retry = supervisor.scheduler_tick("tool-blind:after-tamper")
            self.assertEqual(retry.status, "QUIESCENT")
            self.assertEqual(controller.observations, [])
            self.assertEqual(permission_gate.calls, 0)
            self.assertEqual(executor.calls, 0)

    def test_retained_formation_lineage_tamper_fails_before_any_effect(self) -> None:
        def tamper_input(pending):
            pending["formation"]["formation_input"]["cognitive_state"][
                "situated_state"
            ]["focus"] = ["A target injected after the input was retained."]

        def tamper_model(pending):
            pending["formation"]["target_model_ref"] = "sha256:" + "f" * 64

        def tamper_evidence_mapping(pending):
            pending["formation"]["formation_evidence_refs"][0] = (
                "sha256:" + "e" * 64
            )

        cases = (
            (
                "formation-input",
                tamper_input,
                "autonomous target formation input identity differs",
            ),
            (
                "target-model",
                tamper_model,
                "autonomous target formation input identity differs",
            ),
            (
                "evidence-key-ref-mapping",
                tamper_evidence_mapping,
                "autonomous target evidence lineage differs",
            ),
        )
        for label, tamper, expected_error in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                state = _tool_blind_grounded_state()
                experience = _ToolBlindInitiativeExperience()
                controller = _CapturingObservedOutcomeController()
                executor = _CountingAuthoredArtifactExecutor()
                permission_gate = _CountingPermissionGate()
                adapter = HigherLevelAutonomyCycleAdapter(
                    memory=_Memory(),
                    experience_model=experience,
                    controller=controller,
                )
                original_propose = adapter.propose_autonomous_request

                def propose_then_tamper(**kwargs):
                    request = original_propose(**kwargs)
                    pending = adapter._pending_autonomous_initiative
                    self.assertIsNotNone(pending)
                    tamper(pending)
                    return request

                adapter.propose_autonomous_request = propose_then_tamper
                supervisor = PersistentAutonomySupervisor(
                    Path(directory) / f"tool-blind-{label}.sqlite3",
                    genesis=JennyGenesis.owner_approved(
                        created_at_utc="2026-09-02T12:00:00Z"
                    ),
                    initial_state=state,
                    clock=_artifact_test_clock(),
                    cycle=adapter,
                    affordances=_authored_artifact_registry(executor),
                    permission_gate=permission_gate,
                )
                before_state = supervisor.state_bytes()
                before_head = supervisor.state_head()

                with self.assertRaisesRegex(ValueError, expected_error):
                    supervisor.scheduler_tick("tool-blind:tamper-" + label)

                self.assertEqual(controller.observations, [])
                self.assertEqual(permission_gate.calls, 0)
                self.assertEqual(executor.calls, 0)
                self.assertEqual(supervisor.state_bytes(), before_state)
                self.assertEqual(supervisor.state_head(), before_head)
                self.assertIsNone(supervisor.pending_bytes())
                self.assertIsNone(adapter._pending_autonomous_initiative)

    def test_legacy_catalog_shaped_state_never_reaches_tool_blind_target_formation(
        self,
    ) -> None:
        sentinels = (
            "POISON_LAST_INTENT_TOOL_CATALOG",
            "POISON_INITIATIVE_AFFORDANCE_REGISTRY",
            "POISON_RANKING_AVAILABLE_OPERATION",
        )
        state = _tool_blind_grounded_state(
            last_outcome={
                "epistemic_status": "UNEVALUATED",
                "bounded_semantic_stress": "x" * 100_000,
            },
            last_intent={
                "proposed_action": sentinels[0],
                "affordance_preference_order": ["artifact.compose"],
            },
            initiative_state={
                "next_internal_request": sentinels[1],
                "available_tools": ["artifact.compose"],
            },
            intent_ranking_evidence=[
                {
                    "selected_affordance_id": "artifact.compose",
                    "catalog_echo": sentinels[2],
                }
            ],
        )
        experience = _ToolBlindInitiativeExperience(request="")
        adapter = HigherLevelAutonomyCycleAdapter(
            memory=_Memory(),
            experience_model=experience,
            controller=_CapturingObservedOutcomeController(),
        )

        request = adapter.propose_autonomous_request(
            state=state,
            temporal=_clock().sample(-1),
            affordances=(
                Affordance(
                    "artifact.compose",
                    "ACT",
                    "Compose one private artifact.",
                    "internal.cognition",
                ),
            ),
        )

        self.assertIsNone(request)
        self.assertEqual(len(experience.formation_calls), 1)
        formation_call = experience.formation_calls[0]
        formation_state = formation_call["cognitive_state"]
        self.assertNotIn("last_intent", formation_state)
        self.assertNotIn("initiative_state", formation_state)
        self.assertNotIn("intent_ranking_evidence", formation_state)
        serialized = repr(formation_call)
        for sentinel in sentinels:
            self.assertNotIn(sentinel, serialized)
        self.assertNotIn("artifact.compose", serialized)
        cognitive_json = json.dumps(
            formation_state, separators=(",", ":"), sort_keys=True
        )
        self.assertLessEqual(len(cognitive_json), 49_152)
        formation = adapter.last_autonomous_formation
        self.assertIsNotNone(formation)
        formation_input_json = json.dumps(
            formation["formation_input"], separators=(",", ":"), sort_keys=True
        )
        self.assertLessEqual(len(formation_input_json), 65_536)
        self.assertNotIn("x" * 100_000, formation_input_json)

    def test_autonomous_recall_query_prioritizes_unresolved_situated_semantics(
        self,
    ) -> None:
        critical = "Which source can resolve the observed temporal contradiction?"
        query = _autonomous_target_recall_query(
            {
                "authored_artifacts": {
                    "purpose": "STATIC_METADATA_" + "x" * 100_000,
                    "recent_artifacts": [],
                    "retained_working_set_count": 0,
                },
                "self_observation_diary": {
                    "purpose": "STATIC_DIARY_" + "y" * 100_000,
                    "active_hypotheses": [],
                    "recent_resolutions": [],
                },
                "situated_state": {
                    "world_model": ["Two observations currently disagree."],
                    "self_model": ["The disagreement remains unresolved."],
                    "focus": ["Seek source-bound evidence."],
                    "unfinished_patterns": [critical],
                },
                "last_human_interaction": {
                    "human_observation": "Please retain adjacent context.",
                    "model_output": "Adjacent context retained.",
                },
            }
        )

        self.assertTrue(query.startswith("situated_state="))
        self.assertIn(critical, query)
        self.assertIn("Please retain adjacent context.", query)
        self.assertNotIn("STATIC_METADATA_", query)
        self.assertNotIn("STATIC_DIARY_", query)
        self.assertLessEqual(len(query), 4_096)

    def test_state_or_moving_origin_advance_invalidates_pending_target_formation(
        self,
    ) -> None:
        original_state = _tool_blind_grounded_state()
        changed_state_payload = json.loads(original_state)
        changed_state_payload["state_advanced_after_formation"] = True
        changed_state = json.dumps(
            changed_state_payload, separators=(",", ":"), sort_keys=True
        ).encode()
        affordances = (
            Affordance(
                "test.observe",
                "ACT",
                "Observe one bounded result.",
                "internal.cognition",
            ),
        )
        cases = ("state", "moving-origin")
        for case in cases:
            with self.subTest(case=case):
                clock = _clock()
                formed_at = clock.sample(-1)
                experience = _ToolBlindInitiativeExperience()
                controller = _CapturingObservedOutcomeController()
                adapter = HigherLevelAutonomyCycleAdapter(
                    memory=_Memory(),
                    experience_model=experience,
                    controller=controller,
                )
                request = adapter.propose_autonomous_request(
                    state=original_state,
                    temporal=formed_at,
                    affordances=affordances,
                )
                cycle_state = changed_state if case == "state" else original_state
                cycle_temporal = (
                    formed_at if case == "state" else clock.sample(0)
                )

                with self.assertRaisesRegex(
                    ValueError,
                    "scheduler target formation no longer matches this cycle",
                ):
                    adapter.choose(
                        observation=CycleObservation(
                            "tool-blind:stale-" + case,
                            "SCHEDULER",
                            request,
                        ),
                        temporal=cycle_temporal,
                        affordances=affordances,
                        state=cycle_state,
                    )

                self.assertEqual(controller.observations, [])
                self.assertIsNone(adapter._pending_autonomous_initiative)
                with self.assertRaisesRegex(
                    RuntimeError,
                    "requires a valid frozen target formation",
                ):
                    adapter.choose(
                        observation=CycleObservation(
                            "tool-blind:stale-replay-" + case,
                            "SCHEDULER",
                            request,
                        ),
                        temporal=formed_at,
                        affordances=affordances,
                        state=original_state,
                    )
                self.assertEqual(controller.observations, [])

    def test_clock_discontinuity_or_expired_handoff_invalidates_pending_target(
        self,
    ) -> None:
        state = _tool_blind_grounded_state()
        affordances = (
            Affordance(
                "test.observe",
                "ACT",
                "Observe one bounded result.",
                "internal.cognition",
            ),
        )
        cases = ("clock-jump", "anchor-change", "over-300-seconds")
        for case in cases:
            with self.subTest(case=case):
                source = _Source()
                clock = TrustedClock(
                    timezone_name="UTC",
                    wall_clock=source.wall_now,
                    monotonic_clock=source.mono_now,
                )
                formed_at = clock.sample(-1)
                experience = _ToolBlindInitiativeExperience()
                controller = _CapturingObservedOutcomeController()
                adapter = HigherLevelAutonomyCycleAdapter(
                    memory=_Memory(),
                    experience_model=experience,
                    controller=controller,
                )
                request = adapter.propose_autonomous_request(
                    state=state,
                    temporal=formed_at,
                    affordances=affordances,
                )
                if case == "clock-jump":
                    cycle_temporal = replace(formed_at, jump_detected=True)
                elif case == "anchor-change":
                    cycle_temporal = replace(
                        formed_at, clock_anchor_ref="sha256:" + "f" * 64
                    )
                else:
                    source.wall += timedelta(seconds=301)
                    source.mono += 301_000_000_000
                    cycle_temporal = clock.sample(-1)

                with self.assertRaisesRegex(
                    ValueError,
                    "scheduler target formation no longer matches this cycle",
                ):
                    adapter.choose(
                        observation=CycleObservation(
                            "tool-blind:temporal-" + case,
                            "SCHEDULER",
                            request,
                        ),
                        temporal=cycle_temporal,
                        affordances=affordances,
                        state=state,
                    )

                self.assertEqual(controller.observations, [])
                self.assertIsNone(adapter._pending_autonomous_initiative)

    def test_target_formation_compatibility_matrix_records_catalog_visibility(
        self,
    ) -> None:
        state = _tool_blind_grounded_state()
        temporal = _clock().sample(-1)
        affordances = (
            Affordance(
                "test.observe",
                "ACT",
                "Observe one bounded result.",
                "internal.cognition",
            ),
        )

        tool_blind_experience = _ToolBlindInitiativeExperience()
        tool_blind_adapter = HigherLevelAutonomyCycleAdapter(
            memory=_Memory(),
            experience_model=tool_blind_experience,
            controller=_CapturingObservedOutcomeController(),
        )
        self.assertIsNotNone(
            tool_blind_adapter.propose_autonomous_request(
                state=state,
                temporal=temporal,
                affordances=affordances,
            )
        )
        tool_blind = tool_blind_adapter.last_autonomous_formation
        self.assertIsNotNone(tool_blind)
        self.assertEqual(
            tool_blind["formation_mode"], "TOOL_BLIND_STATE_MEMORY_TIME"
        )
        self.assertEqual(tool_blind["formation_input"]["operation_catalog"], [])
        self.assertNotIn("affordances", tool_blind_experience.formation_calls[0])

        legacy_experience = _InitiativeExperience()
        legacy_adapter = HigherLevelAutonomyCycleAdapter(
            memory=_Memory(),
            experience_model=legacy_experience,
            controller=_CapturingObservedOutcomeController(),
        )
        self.assertIsNotNone(
            legacy_adapter.propose_autonomous_request(
                state=state,
                temporal=temporal,
                affordances=affordances,
            )
        )
        legacy = legacy_adapter.last_autonomous_formation
        self.assertIsNotNone(legacy)
        expected_catalog = [asdict(item) for item in affordances]
        self.assertEqual(
            legacy["formation_mode"], "LEGACY_AFFORDANCE_VISIBLE"
        )
        self.assertEqual(
            legacy["formation_input"]["operation_catalog"], expected_catalog
        )
        self.assertEqual(
            legacy_experience.initiative_calls[0]["affordances"], expected_catalog
        )

    def test_scheduler_reuses_model_authored_target_as_structured_experience(
        self,
    ) -> None:
        class FormationCountingExperience(_ToolBlindInitiativeExperience):
            def __init__(self) -> None:
                super().__init__()
                self.situated_generation_calls = 0

            def generate_situated_experience(self, *args, **kwargs):
                del args, kwargs
                self.situated_generation_calls += 1
                raise AssertionError(
                    "a retained target must not trigger redundant experience generation"
                )

        state = _tool_blind_grounded_state()
        temporal = _clock().sample(-1)
        affordances = (
            Affordance(
                "test.observe",
                "ACT",
                "Observe one bounded result.",
                "internal.cognition",
            ),
        )
        experience = FormationCountingExperience()
        adapter = HigherLevelAutonomyCycleAdapter(
            memory=_Memory(),
            experience_model=experience,
            controller=_CapturingObservedOutcomeController(),
        )

        request = adapter.propose_autonomous_request(
            state=state,
            temporal=temporal,
            affordances=affordances,
        )
        self.assertIsNotNone(request)
        choice = adapter.choose(
            observation=CycleObservation(
                "tool-blind:reuse-model-target",
                "SCHEDULER",
                request,
            ),
            temporal=temporal,
            affordances=affordances,
            state=state,
        )

        context = json.loads(choice.context_json)
        self.assertEqual(len(experience.formation_calls), 1)
        self.assertEqual(experience.situated_generation_calls, 0)
        self.assertEqual(
            context["experience"]["interpretation"],
            experience.request,
        )
        self.assertEqual(
            context["phase_timings_ms"]["structured_experience_reused"],
            0.0,
        )

    def test_stored_request_cannot_bypass_tool_blind_reformation(self) -> None:
        state = json.dumps(
            {
                "affordance_utility": {},
                "memory_utility": {},
                "next_internal_request": (
                    "A catalog-exposed draft must not execute directly."
                ),
                "affordance_signals": {},
                "motivation_weights": {},
                "situated_state": {
                    "world_model": ["No current evidence supports the old draft."],
                    "self_model": ["The target must be re-evaluated first."],
                    "focus": ["Prefer honest dormancy."],
                    "unfinished_patterns": [],
                },
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        with tempfile.TemporaryDirectory() as directory:
            experience = _ToolBlindInitiativeExperience(request="")
            controller = _CapturingObservedOutcomeController()
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "tool-blind-dormancy.sqlite3",
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=state,
                clock=_clock(),
                cycle=HigherLevelAutonomyCycleAdapter(
                    memory=_Memory(),
                    experience_model=experience,
                    controller=controller,
                ),
                affordances=_authored_artifact_registry(),
            )

            result = supervisor.scheduler_tick("tool-blind:dormant")

            self.assertEqual(result.status, "QUIESCENT")
            self.assertEqual(len(experience.formation_calls), 1)
            self.assertEqual(controller.observations, [])
            formation_call = experience.formation_calls[0]
            formation_state = formation_call["cognitive_state"]
            self.assertNotIn("prospective_request", formation_state)
            self.assertNotIn("next_internal_request", formation_state)
            self.assertNotIn(
                "A catalog-exposed draft must not execute directly.",
                repr(formation_call),
            )
            self.assertEqual(
                json.loads(supervisor.state_bytes())["next_internal_request"],
                "A catalog-exposed draft must not execute directly.",
            )

    def test_source_bound_observation_without_assessment_provider_keeps_observation_only(
        self,
    ) -> None:
        source_ref = "sha256:" + "4" * 64
        captured = {}

        def observe(request):
            observation = ObservableConsequence(
                request_ref=request.idempotency_key,
                source_kind="TEST",
                source_ref=source_ref,
                observation_json='{"result":"unassessed"}',
            )
            captured["observation"] = observation
            return AffordanceReceipt(
                "COMPLETED", "One raw result was observed.", (), observation
            )

        with tempfile.TemporaryDirectory() as directory:
            memory = _Memory()
            experience = _Experience()
            controller = _ObservedOutcomeController()
            adapter = HigherLevelAutonomyCycleAdapter(
                memory=memory,
                experience_model=experience,
                controller=controller,
            )
            registry = DynamicAffordanceRegistry()
            registry.register(
                Affordance(
                    "test.observe",
                    "ACT",
                    "Observe one bounded result.",
                    "internal.cognition",
                ),
                observe,
                observable_source_ref=source_ref,
                observable_source_kind="TEST",
            )
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "observation-only.sqlite3",
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=_state(),
                clock=_clock(),
                cycle=adapter,
                affordances=registry,
            )
            result = supervisor.scheduler_tick("observed:fallback")

            self.assertEqual(result.status, "COMMITTED")
            learned = json.loads(supervisor.state_bytes())
            retained = learned["observed_outcome_evidence"][-1]
            self.assertEqual(
                retained["epistemic_status"],
                "SOURCE_BOUND_OBSERVATION_WITHOUT_VALUE_JUDGMENT",
            )
            self.assertEqual(
                retained["observation_ref"], captured["observation"].observation_ref
            )
            self.assertNotIn("outcome_assessment_evidence", learned)
            self.assertNotIn("situated_state", learned)
            self.assertNotIn("last_outcome", learned)
            self.assertNotIn("intent_ranking_evidence", learned)
            self.assertEqual(controller.update_calls, 0)
            self.assertEqual(learned["affordance_utility"], {})
            self.assertEqual(learned["memory_utility"], {})
            self.assertNotIn("capability_evidence", learned)
            self.assertEqual(experience.reflections, 0)
            self.assertEqual(memory.stores, [])

    def test_learned_intrinsic_signal_weights_change_dynamic_ranking(self) -> None:
        controller = OutcomeUtilityAffordanceController()
        state = json.dumps(
            {
                "affordance_utility": {},
                "affordance_signals": {
                    "act.alpha": {"learning_progress": 0.1},
                    "act.beta": {"learning_progress": 0.8},
                },
                "motivation_weights": {"learning_progress": 1.0},
                "memory_utility": {},
                "next_internal_request": "Inspect one learned uncertainty.",
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        experience = _Experience().generate_experience("request", (), type("T", (), {"now": 0})())
        source = _Source()
        temporal = TrustedClock(
            wall_clock=source.wall_now, monotonic_clock=source.mono_now
        ).sample(-1)
        from angler.runtime.persistent_autonomy import CycleObservation

        selection = controller.select(
            observation=CycleObservation("signal:test", "SCHEDULER", ""),
            temporal=temporal,
            affordances=(
                Affordance("act.alpha", "ACT", "Alpha.", "internal.cognition"),
                Affordance("act.beta", "ACT", "Beta.", "internal.cognition"),
            ),
            memories=(),
            experience=experience,
            state=state,
        )
        self.assertEqual(selection.selected_affordance_id, "act.beta")
        self.assertGreater(selection.rankings[0].score, selection.rankings[1].score)

    def test_default_unqualified_controller_is_shadow_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory, experience, cortex = _Memory(), _Experience(), _Cortex()
            adapter = HigherLevelAutonomyCycleAdapter(
                memory=memory,
                experience_model=experience,
                controller=OutcomeUtilityAffordanceController(),
            )
            registry = DynamicAffordanceRegistry()
            registry.register(
                Affordance("cortex.respond", "ACT", "Run the frozen cortex.", "internal.cognition"),
                HigherLevelCortexAffordanceExecutor(cortex, _consequence),
            )
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "shadow.sqlite3",
                genesis=JennyGenesis.owner_approved(created_at_utc="2026-09-02T12:00:00Z"),
                initial_state=_state(),
                clock=_clock(),
                cycle=adapter,
                affordances=registry,
            )
            result = supervisor.scheduler_tick("adapter:shadow")
            self.assertEqual(result.status, "SHADOW")
            self.assertEqual(cortex.calls, [])
            self.assertEqual(supervisor.state_head().moving_origin_ordinal, -1)

    def test_qualified_synthetic_binding_runs_one_integrated_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory, experience, cortex = _Memory(), _Experience(), _Cortex()
            adapter = HigherLevelAutonomyCycleAdapter(
                memory=memory,
                experience_model=experience,
                controller=OutcomeUtilityAffordanceController(
                    alpha=0.5, qualification_ref=QUALIFICATION
                ),
            )
            registry = DynamicAffordanceRegistry()
            registry.register(
                Affordance("cortex.respond", "ACT", "Run the frozen cortex.", "internal.cognition"),
                HigherLevelCortexAffordanceExecutor(cortex, _consequence),
            )
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "qualified.sqlite3",
                genesis=JennyGenesis.owner_approved(created_at_utc="2026-09-02T12:00:00Z"),
                initial_state=_state(),
                clock=_clock(),
                cycle=adapter,
                affordances=registry,
            )
            result = supervisor.human_ingress("adapter:human", "Solve this bounded synthetic request.")
            self.assertEqual(result.status, "COMMITTED")
            self.assertEqual(len(cortex.calls), 1)
            self.assertEqual(experience.reflections, 1)
            self.assertEqual(memory.stores, [])
            learned = json.loads(supervisor.state_bytes())
            self.assertGreater(learned["affordance_utility"]["cortex.respond"], 0.0)
            self.assertNotIn("sha256:" + "a" * 64, learned["memory_utility"])
            self.assertEqual(
                learned["memory_credit_evidence"][-1]["credited_refs"], []
            )
            self.assertEqual(
                learned["last_outcome"]["epistemic_status"], "OBSERVED_CONSEQUENCE"
            )
            self.assertEqual(
                learned["last_intent"]["epistemic_status"],
                "MODEL_INTENT_WITH_OBSERVED_CONSEQUENCE",
            )
            self.assertEqual(
                learned["last_intent"]["selected_affordance_id"],
                "cortex.respond",
            )
            self.assertEqual(
                learned["compute_calibration"][-1]["epistemic_status"],
                "COMPUTE_ROUTE_WITH_OBSERVED_CONSEQUENCE",
            )
            self.assertEqual(len(learned["last_intent"]["alternatives"]), 1)
            self.assertIn("observed_consequence", learned["last_intent"])
            self.assertEqual(
                learned["situated_state"]["epistemic_status"],
                "INFERRED_FROM_OBSERVED_CONSEQUENCE",
            )
            self.assertNotIn("procedure_proposal", learned["situated_state"])
            self.assertNotIn("capability_evidence", learned)
            unavailable = learned["capability_consolidation_evidence"][-1]
            self.assertEqual(
                unavailable["epistemic_status"],
                "CAPABILITY_DISPOSITION_UNAVAILABLE",
            )
            self.assertFalse(unavailable["proposal"]["retain"])
            self.assertEqual(supervisor.state_head().moving_origin_ordinal, 0)
            pending = supervisor.pending_projections()
            self.assertEqual(len(pending), 1)
            projected = supervisor.retry_pending_projections(
                SemanticMemoryConsolidationProjector(memory)
            )
            self.assertEqual(projected, (pending[0].projection_ref,))
            self.assertEqual(len(memory.stores), 1)
            self.assertEqual(memory.stores[0][1], (pending[0].episode_ref, pending[0].event_ref))
            self.assertEqual(supervisor.pending_projections(), ())
            self.assertEqual(
                supervisor.retry_pending_projections(
                    SemanticMemoryConsolidationProjector(memory)
                ),
                (),
            )
            second = supervisor.human_ingress(
                "adapter:human-second", "Try the comparable request once more."
            )
            self.assertEqual(second.status, "COMMITTED")
            after_second = json.loads(supervisor.state_bytes())
            self.assertNotIn("capability_evidence", after_second)
            self.assertEqual(
                len(after_second["capability_consolidation_evidence"]), 2
            )

    def test_model_open_pattern_becomes_attributable_next_internal_inquiry(self):
        class OpenExperience(_Experience):
            def consolidate(self, request, experience, reflection, consequence):
                payload = json.loads(
                    super().consolidate(request, experience, reflection, consequence)
                )
                payload["unfinished_patterns"] = [
                    "Compare the revised process against one fresh bounded case."
                ]
                return json.dumps(payload, separators=(",", ":"), sort_keys=True)

        with tempfile.TemporaryDirectory() as directory:
            memory, experience, cortex = _Memory(), OpenExperience(), _Cortex()
            adapter = HigherLevelAutonomyCycleAdapter(
                memory=memory,
                experience_model=experience,
                controller=OutcomeUtilityAffordanceController(
                    qualification_ref=QUALIFICATION
                ),
            )
            registry = DynamicAffordanceRegistry()
            registry.register(
                Affordance("cortex.respond", "ACT", "Run cortex.", "internal.cognition"),
                HigherLevelCortexAffordanceExecutor(cortex, _consequence),
            )
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "initiative.sqlite3",
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=_state(),
                clock=_clock(),
                cycle=adapter,
                affordances=registry,
            )
            result = supervisor.human_ingress(
                "initiative:seed", "Resolve one bounded case."
            )
            self.assertEqual(result.status, "COMMITTED")
            learned = json.loads(supervisor.state_bytes())
            self.assertEqual(
                learned["next_internal_request"],
                "Compare the revised process against one fresh bounded case.",
            )
            self.assertEqual(
                learned["initiative_state"]["epistemic_status"],
                "MODEL_PROPOSAL_FROM_OBSERVED_CONSEQUENCE",
            )
            self.assertEqual(
                learned["initiative_state"]["receipt_ref"],
                learned["last_outcome"]["receipt_ref"],
            )

    def test_only_model_cited_recalled_memory_receives_outcome_credit(self):
        memory, experience, cortex = _Memory(), _Experience(), _Cortex()
        cited_ref = "sha256:" + "a" * 64
        backend = _ControllerBackend(
            {
                "selected_affordance_id": "cortex.respond",
                "rankings": [{"affordance_id": "cortex.respond", "score": 0.7}],
                "action_payload": "Use the cited prior attempt, then verify it.",
                "state_assessment": "A relevant prior attempt is available.",
                "resolution_target": "Verify transfer to the current request.",
                "expected_state_delta": "The receipt will test whether transfer worked.",
                "evidence_refs": [cited_ref],
            }
        )
        adapter = HigherLevelAutonomyCycleAdapter(
            memory=memory,
            experience_model=experience,
            controller=FrozenModelAffordanceController(
                backend, qualification_ref=QUALIFICATION
            ),
        )
        registry = DynamicAffordanceRegistry()
        registry.register(
            Affordance("cortex.respond", "ACT", "Run cortex.", "internal.cognition"),
            HigherLevelCortexAffordanceExecutor(cortex, _consequence),
        )
        with tempfile.TemporaryDirectory() as directory:
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "cited-memory.sqlite3",
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=_state(),
                clock=_clock(),
                cycle=adapter,
                affordances=registry,
            )
            result = supervisor.scheduler_tick("memory-credit:cited")
            self.assertEqual(result.status, "COMMITTED")
            learned = json.loads(supervisor.state_bytes())
            self.assertGreater(learned["memory_utility"][cited_ref], 0.0)
            self.assertEqual(
                learned["memory_credit_evidence"][-1]["credited_refs"], [cited_ref]
            )

    def test_userless_unevaluated_action_retains_decision_without_second_model_pass_or_credit(
        self,
    ):
        class AppraisingExperience(_Experience):
            def __init__(self):
                super().__init__()
                self.continuation_calls = 0

            def propose_continuation(self, request, receipt, experience):
                self.continuation_calls += 1
                return {
                    "world_model": "One internal comparison remains unverified.",
                    "self_model": "The attempt produced a candidate, not a verified result.",
                    "focus": "Resolve the remaining evidence gap.",
                    "unfinished_patterns": ["Check one contrary example."],
                    "next_internal_request": "Check one contrary example.",
                    "rationale": "A contrary example would test the candidate conclusion.",
                    "uncertainty": 0.6,
                    "self_appraisal": {
                        "changes_noticed": ["A candidate conclusion became available."],
                        "evidence_gained": ["A candidate conclusion was produced."],
                        "remaining_questions": ["Does a contrary example refute it?"],
                        "costs_or_risks": ["One pass was used; transfer is unverified."],
                        "counterevidence": ["No objective receipt exists."],
                        "reasons_to_continue": ["A contrary example can challenge it."],
                        "reasons_to_stop_or_wait": ["Wait if no contrary case is available."],
                        "reasoned_judgment": "One bounded contrary check is justified.",
                    },
                }

        with tempfile.TemporaryDirectory() as directory:
            memory, experience, cortex = _Memory(), AppraisingExperience(), _Cortex()
            adapter = HigherLevelAutonomyCycleAdapter(
                memory=memory,
                experience_model=experience,
                controller=OutcomeUtilityAffordanceController(
                    qualification_ref=QUALIFICATION
                ),
            )
            registry = DynamicAffordanceRegistry()
            registry.register(
                Affordance("cortex.respond", "ACT", "Run cortex.", "internal.cognition"),
                HigherLevelCortexAffordanceExecutor(cortex),
            )
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "self-appraisal.sqlite3",
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=_state(),
                clock=_clock(),
                cycle=adapter,
                affordances=registry,
            )
            result = supervisor.scheduler_tick("self-appraisal:one")
            self.assertEqual(result.status, "COMMITTED")
            learned = json.loads(supervisor.state_bytes())
            self.assertEqual(experience.continuation_calls, 0)
            self.assertEqual(learned["next_internal_request"], "")
            self.assertEqual(
                learned["last_intent"]["epistemic_status"],
                "MODEL_INTENT_WITHOUT_OUTCOME",
            )
            self.assertNotIn("self_appraisal_evidence", learned)
            self.assertEqual(learned.get("affordance_utility", {}), {})
            self.assertNotIn("capability_evidence", learned)
            self.assertEqual(
                learned["situated_state"]["epistemic_status"],
                "MODEL_DECISION_WITHOUT_OUTCOME",
            )
            feedback = supervisor.submit_feedback(
                "self-appraisal:feedback",
                target_episode_ref=result.episode_ref,
                feedback_text="The attempt missed the contrary case and should be revised.",
                feedback_source_ref="sha256:" + "7" * 64,
                consequence=(("human_feedback", -0.5), ("prediction_error", 0.8)),
            )
            self.assertEqual(feedback.status, "COMMITTED")
            calibrated = json.loads(supervisor.state_bytes())
            self.assertEqual(experience.reflections, 0)
            self.assertEqual(
                calibrated["qualitative_feedback"][-1]["consequence"][
                    "human_feedback"
                ],
                -0.5,
            )
            self.assertNotIn("self_appraisal_calibration", calibrated)
            self.assertEqual(
                calibrated["affordance_outcome_profiles"]["cortex.respond"][
                    "observations"
                ],
                1,
            )
            self.assertEqual(
                calibrated["situated_state"]["epistemic_status"],
                "FAST_STATE_UPDATED_FROM_OBSERVED_FEEDBACK",
            )

    def test_self_observation_diary_commits_once_without_credit_and_replays_exactly(
        self,
    ) -> None:
        proposal = _self_observation_proposal(
            label="protective attentional return",
            strength=0.63,
            uncertainty=0.31,
        )
        state = _initiative_state()
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "self-observation-diary.sqlite3"
            memory = _Memory()
            experience = _InitiativeExperience(
                "Record the current recurring functional pattern as a hypothesis."
            )
            controller = _SelfObservationDiaryController((proposal,))
            adapter = HigherLevelAutonomyCycleAdapter(
                memory=memory,
                experience_model=experience,
                controller=controller,
            )
            supervisor = PersistentAutonomySupervisor(
                state_path,
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=state,
                clock=_clock(),
                cycle=adapter,
                affordances=_self_observation_registry(),
            )

            result = supervisor.scheduler_tick("self-observation:first")

            self.assertEqual(result.status, "COMMITTED")
            self.assertEqual(
                result.selected_affordance_id,
                SELF_OBSERVATION_DIARY_AFFORDANCE.affordance_id,
            )
            learned = json.loads(supervisor.state_bytes())
            self.assertEqual(len(learned["self_observation_diary"]), 1)
            entry = learned["self_observation_diary"][0]
            self.assertEqual(entry["contract"], SELF_OBSERVATION_DIARY_CONTRACT)
            self.assertEqual(entry["purpose"], SELF_OBSERVATION_DIARY_PURPOSE)
            self.assertEqual(
                entry["epistemic_status"],
                SELF_OBSERVATION_DIARY_EPISTEMIC_STATUS,
            )
            self.assertEqual(
                entry["phenomenology_status"],
                SELF_OBSERVATION_DIARY_PHENOMENOLOGY_STATUS,
            )
            self.assertEqual(
                entry["proposal"]["candidate_label"],
                "protective attentional return",
            )
            self.assertEqual(entry["proposal"]["estimated_strength"], 0.63)
            self.assertEqual(entry["proposal"]["uncertainty"], 0.31)
            temporal = TemporalV2(**entry["temporal"])
            self.assertEqual(entry["temporal_ref"], temporal.temporal_ref)
            self.assertEqual(temporal.moving_origin_ordinal, 0)
            self.assertEqual(temporal.source, "jenny.persistent-autonomy.v1")
            for reference in (
                entry["observation_ref"],
                entry["choice_ref"],
                entry["receipt_ref"],
                entry["temporal_ref"],
                SELF_OBSERVATION_DIARY_SOURCE_REF,
            ):
                self.assertIn(reference, entry["evidence_refs"])
            unhashed = dict(entry)
            entry_ref = unhashed.pop("entry_ref")
            self.assertEqual(entry_ref, content_ref(unhashed))
            self.assertEqual(
                learned["self_observation_state"]["active_entry_refs"],
                [entry_ref],
            )
            self.assertEqual(controller.update_calls, 0)
            self.assertEqual(learned["affordance_utility"], {})
            self.assertEqual(learned["memory_utility"], {})
            self.assertNotIn("affordance_outcome_profiles", learned)
            self.assertNotIn("capability_evidence", learned)
            self.assertNotIn("capability_use_evidence", learned)
            self.assertEqual(memory.stores, [])
            episode = json.loads(
                supervisor.episode_item(result.episode_ref).payload_json
            )
            self.assertEqual(episode["receipt"]["consequence"], [])
            self.assertEqual(
                episode["receipt"]["observable_consequence"]["source_ref"],
                SELF_OBSERVATION_DIARY_SOURCE_REF,
            )
            committed_state = supervisor.state_bytes()
            committed_head = supervisor.state_head()

            replay_experience = _InitiativeExperience(
                "Record the current recurring functional pattern as a hypothesis."
            )
            replay_controller = _SelfObservationDiaryController((proposal,))
            restarted = PersistentAutonomySupervisor(
                state_path,
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=state,
                clock=_clock(),
                cycle=HigherLevelAutonomyCycleAdapter(
                    memory=_Memory(),
                    experience_model=replay_experience,
                    controller=replay_controller,
                ),
                affordances=_self_observation_registry(),
            )
            replay = restarted.scheduler_tick("self-observation:first")
            self.assertEqual(replay.episode_ref, result.episode_ref)
            self.assertEqual(restarted.state_bytes(), committed_state)
            self.assertEqual(restarted.state_head(), committed_head)
            self.assertEqual(
                len(json.loads(restarted.state_bytes())["self_observation_diary"]),
                1,
            )
            self.assertEqual(replay_controller.observations, [])

    def test_self_observation_revision_supersedes_only_active_head_and_is_visible_next(
        self,
    ) -> None:
        first_proposal = _self_observation_proposal(
            label="attention returning to unresolved context",
            strength=0.55,
            uncertainty=0.4,
        )
        experience = _InitiativeExperience(
            "Compare the current functional pattern with the diary evidence."
        )
        controller = _SelfObservationDiaryController((first_proposal,))
        adapter = HigherLevelAutonomyCycleAdapter(
            memory=_Memory(),
            experience_model=experience,
            controller=controller,
        )
        with tempfile.TemporaryDirectory() as directory:
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "self-observation-revision.sqlite3",
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=_initiative_state(),
                clock=_clock(),
                cycle=adapter,
                affordances=_self_observation_registry(),
            )
            first = supervisor.scheduler_tick("self-observation:revision-one")
            self.assertEqual(first.status, "COMMITTED")
            first_entry = json.loads(supervisor.state_bytes())[
                "self_observation_diary"
            ][0]
            first_ref = first_entry["entry_ref"]
            revised_proposal = _self_observation_proposal(
                label="evidence-sensitive attentional return",
                strength=0.42,
                uncertainty=0.22,
                revision_target_ref=first_ref,
            )
            controller.proposals.append(revised_proposal)

            second = supervisor.scheduler_tick("self-observation:revision-two")

            self.assertEqual(second.status, "COMMITTED")
            self.assertEqual(len(experience.initiative_calls), 2)
            next_cognitive_state = experience.initiative_calls[1][
                "cognitive_state"
            ]
            visible = next_cognitive_state["self_observation_diary"]
            self.assertEqual(visible["purpose"], SELF_OBSERVATION_DIARY_PURPOSE)
            self.assertEqual(
                visible["active_hypotheses"][0]["entry_ref"],
                first_ref,
            )
            self.assertEqual(
                controller.states[1]["self_observation_state"]["purpose"],
                SELF_OBSERVATION_DIARY_PURPOSE,
            )
            self.assertEqual(
                controller.states[1]["self_observation_diary"][0]["entry_ref"],
                first_ref,
            )
            learned = json.loads(supervisor.state_bytes())
            self.assertEqual(len(learned["self_observation_diary"]), 2)
            revised_entry = learned["self_observation_diary"][1]
            self.assertEqual(
                revised_entry["proposal"]["revision_target_ref"],
                first_ref,
            )
            self.assertEqual(
                revised_entry["proposal"]["candidate_label"],
                "evidence-sensitive attentional return",
            )
            self.assertEqual(revised_entry["proposal"]["estimated_strength"], 0.42)
            self.assertEqual(revised_entry["proposal"]["uncertainty"], 0.22)
            self.assertEqual(
                learned["self_observation_state"]["active_entry_refs"],
                [revised_entry["entry_ref"]],
            )
            self.assertEqual(controller.update_calls, 0)
            self.assertEqual(learned["affordance_utility"], {})
            self.assertNotIn("capability_evidence", learned)

    def test_self_observation_resolution_closes_active_lineage_and_remains_visible(
        self,
    ) -> None:
        first_proposal = _self_observation_proposal(
            label="provisional protective attentional return",
            strength=0.57,
            uncertainty=0.36,
        )
        experience = _InitiativeExperience(
            "Compare current evidence with one provisional functional appraisal."
        )
        controller = _SelfObservationDiaryController((first_proposal,))
        with tempfile.TemporaryDirectory() as directory:
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "self-observation-resolution.sqlite3",
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=_initiative_state(),
                clock=_clock(),
                cycle=HigherLevelAutonomyCycleAdapter(
                    memory=_Memory(),
                    experience_model=experience,
                    controller=controller,
                ),
                affordances=_self_observation_registry(),
            )
            first = supervisor.scheduler_tick("self-observation:resolve-one")
            self.assertEqual(first.status, "COMMITTED")
            first_entry = json.loads(supervisor.state_bytes())[
                "self_observation_diary"
            ][0]
            first_ref = first_entry["entry_ref"]
            resolution = _self_observation_proposal(
                label="protective return not currently sustained",
                strength=0.14,
                uncertainty=0.43,
                revision_target_ref=first_ref,
                resolution_reason=(
                    "Later attributable observations no longer warrant keeping "
                    "the earlier functional appraisal active."
                ),
            )
            controller.proposals.append(resolution)

            second = supervisor.scheduler_tick("self-observation:resolve-two")

            self.assertEqual(second.status, "COMMITTED")
            learned = json.loads(supervisor.state_bytes())
            resolution_entry = learned["self_observation_diary"][-1]
            appraisal_state = learned["self_observation_state"]
            self.assertEqual(appraisal_state["active_entry_refs"], [])
            self.assertEqual(
                appraisal_state["resolved_entry_refs"],
                [resolution_entry["entry_ref"]],
            )
            self.assertEqual(appraisal_state["resolved_target_refs"], [first_ref])
            self.assertEqual(
                appraisal_state["strength_calibration_status"],
                SELF_OBSERVATION_DIARY_STRENGTH_STATUS,
            )
            self.assertEqual(
                appraisal_state["label_status"],
                SELF_OBSERVATION_DIARY_LABEL_STATUS,
            )
            self.assertEqual(
                appraisal_state["appraisal_role"],
                SELF_OBSERVATION_DIARY_APPRAISAL_ROLE,
            )
            self.assertEqual(controller.update_calls, 0)
            self.assertEqual(learned["affordance_utility"], {})
            self.assertNotIn("capability_evidence", learned)

            # A later learned deliberation sees both the empty active set and the
            # source/time/state-bound terminal record; no second state authority is
            # introduced to carry appraisal history.
            controller.proposals.append(
                _self_observation_proposal(
                    label="new unrelated provisional appraisal",
                    strength=0.28,
                    uncertainty=0.65,
                )
            )
            third = supervisor.scheduler_tick("self-observation:resolve-visibility")
            self.assertEqual(third.status, "COMMITTED")
            visible = experience.initiative_calls[2]["cognitive_state"][
                "self_observation_diary"
            ]
            self.assertEqual(visible["active_hypotheses"], [])
            self.assertEqual(
                visible["recent_resolutions"][0]["entry_ref"],
                resolution_entry["entry_ref"],
            )
            self.assertEqual(
                visible["strength_calibration_status"],
                SELF_OBSERVATION_DIARY_STRENGTH_STATUS,
            )
            self.assertEqual(
                controller.states[2]["self_observation_state"][
                    "resolved_target_refs"
                ],
                [first_ref],
            )

    def test_self_observation_revision_rejects_foreign_and_superseded_targets(
        self,
    ) -> None:
        foreign = _self_observation_proposal(
            revision_target_ref="sha256:" + "f" * 64
        )
        with tempfile.TemporaryDirectory() as directory:
            controller = _SelfObservationDiaryController((foreign,))
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "self-observation-foreign.sqlite3",
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=_initiative_state(),
                clock=_clock(),
                cycle=HigherLevelAutonomyCycleAdapter(
                    memory=_Memory(),
                    experience_model=_InitiativeExperience(),
                    controller=controller,
                ),
                affordances=_self_observation_registry(),
            )
            before_state = supervisor.state_bytes()
            before_head = supervisor.state_head()
            with self.assertRaisesRegex(
                ValueError,
                "revision must target a visible active hypothesis",
            ):
                supervisor.scheduler_tick("self-observation:foreign-target")
            self.assertEqual(supervisor.state_bytes(), before_state)
            after_head = supervisor.state_head()
            self.assertEqual(after_head.state_ref, before_head.state_ref)
            self.assertEqual(
                after_head.moving_origin_ordinal,
                before_head.moving_origin_ordinal,
            )
            self.assertEqual(after_head.last_event_ref, before_head.last_event_ref)
            self.assertTrue(supervisor.pending_bytes())
            self.assertEqual(controller.update_calls, 0)

        with tempfile.TemporaryDirectory() as directory:
            first_proposal = _self_observation_proposal(label="first hypothesis")
            controller = _SelfObservationDiaryController((first_proposal,))
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "self-observation-superseded.sqlite3",
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=_initiative_state(),
                clock=_clock(),
                cycle=HigherLevelAutonomyCycleAdapter(
                    memory=_Memory(),
                    experience_model=_InitiativeExperience(),
                    controller=controller,
                ),
                affordances=_self_observation_registry(),
            )
            supervisor.scheduler_tick("self-observation:superseded-one")
            first_ref = json.loads(supervisor.state_bytes())[
                "self_observation_diary"
            ][0]["entry_ref"]
            controller.proposals.append(
                _self_observation_proposal(
                    label="second hypothesis",
                    revision_target_ref=first_ref,
                )
            )
            supervisor.scheduler_tick("self-observation:superseded-two")
            before_state = supervisor.state_bytes()
            before_head = supervisor.state_head()
            controller.proposals.append(
                _self_observation_proposal(
                    label="invalid second revision",
                    revision_target_ref=first_ref,
                )
            )
            with self.assertRaisesRegex(
                ValueError,
                "revision must target a visible active hypothesis",
            ):
                supervisor.scheduler_tick("self-observation:superseded-three")
            self.assertEqual(supervisor.state_bytes(), before_state)
            after_head = supervisor.state_head()
            self.assertEqual(after_head.state_ref, before_head.state_ref)
            self.assertEqual(
                after_head.moving_origin_ordinal,
                before_head.moving_origin_ordinal,
            )
            self.assertEqual(after_head.last_event_ref, before_head.last_event_ref)
            self.assertTrue(supervisor.pending_bytes())
            self.assertEqual(
                json.loads(supervisor.state_bytes())[
                    "self_observation_state"
                ]["active_entry_refs"],
                [json.loads(before_state)["self_observation_diary"][1]["entry_ref"]],
            )
            self.assertEqual(controller.update_calls, 0)

    def test_authored_artifact_commits_restarts_supersedes_and_receives_no_credit(
        self,
    ) -> None:
        recalled_episode_ref = "sha256:" + "a" * 64
        first_action = _authored_artifact_action(
            evidence_refs=(recalled_episode_ref,),
            source_episode_refs=(recalled_episode_ref,),
        )
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "authored-artifact.sqlite3"
            memory = _Memory()
            experience = _AuthoredArtifactExperience()
            controller = _AuthoredArtifactController((first_action,))
            supervisor = PersistentAutonomySupervisor(
                state_path,
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=_initiative_state(),
                clock=_artifact_test_clock(),
                cycle=HigherLevelAutonomyCycleAdapter(
                    memory=memory,
                    experience_model=experience,
                    controller=controller,
                ),
                affordances=_authored_artifact_registry(),
            )

            first = supervisor.scheduler_tick("authored-artifact:first")

            self.assertEqual(first.status, "COMMITTED")
            learned = json.loads(supervisor.state_bytes())
            self.assertEqual(len(learned["authored_artifacts"]), 1)
            entry = learned["authored_artifacts"][0]
            self.assertEqual(entry["artifact"], first_action.canonical_payload())
            self.assertEqual(entry["artifact_ref"], first_action.artifact_ref)
            self.assertEqual(entry["predecessor_version"], None)
            self.assertEqual(entry["purpose"], AUTHORED_ARTIFACT_PURPOSE)
            self.assertEqual(
                entry["epistemic_status"],
                "MODEL_AUTHORED_PRIVATE_ARTIFACT_UNVERIFIED",
            )
            unhashed = dict(entry)
            entry_ref = unhashed.pop("entry_ref")
            self.assertEqual(entry_ref, content_ref(unhashed))
            self.assertIn(first_action.artifact_ref, entry["evidence_refs"])
            self.assertEqual(
                learned["observed_outcome_evidence"][-1][
                    "authored_artifact_entry_ref"
                ],
                entry_ref,
            )
            self.assertEqual(controller.update_calls, 0)
            self.assertEqual(learned["affordance_utility"], {})
            self.assertEqual(learned["memory_utility"], {})
            self.assertNotIn("capability_evidence", learned)
            episode = json.loads(
                supervisor.episode_item(first.episode_ref).payload_json
            )
            self.assertEqual(episode["receipt"]["consequence"], [])
            self.assertEqual(
                episode["receipt"]["observable_consequence"]["source_ref"],
                AUTHORED_ARTIFACT_SOURCE_REF,
            )
            pending = supervisor.pending_projections()
            self.assertEqual(len(pending), 1)
            supervisor.retry_pending_projections(
                SemanticMemoryConsolidationProjector(memory)
            )
            projected = json.loads(memory.stores[-1][0])
            self.assertEqual(
                projected["memory_kind"], "MODEL_AUTHORED_PRIVATE_ARTIFACT"
            )
            self.assertEqual(projected["body"], first_action.body)
            self.assertEqual(projected["artifact_ref"], first_action.artifact_ref)
            committed_state = supervisor.state_bytes()
            committed_head = supervisor.state_head()

            second_action = _authored_artifact_action(
                version=2,
                title="A revised question worth carrying forward",
                body=(
                    "Later context changed the work without erasing its predecessor; "
                    "the revision remains a model-authored and unverified artifact."
                ),
                evidence_refs=(first_action.artifact_ref,),
                supersedes_ref=first_action.artifact_ref,
            )
            restarted_experience = _AuthoredArtifactExperience()
            restarted_controller = _AuthoredArtifactController((second_action,))
            restarted = PersistentAutonomySupervisor(
                state_path,
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=_initiative_state(),
                clock=_artifact_test_clock(),
                cycle=HigherLevelAutonomyCycleAdapter(
                    memory=_Memory(),
                    experience_model=restarted_experience,
                    controller=restarted_controller,
                ),
                affordances=_authored_artifact_registry(),
            )
            self.assertEqual(restarted.state_bytes(), committed_state)
            self.assertEqual(restarted.state_head(), committed_head)
            self.assertEqual(restarted_controller.states, [])

            second = restarted.scheduler_tick("authored-artifact:second")

            self.assertEqual(second.status, "COMMITTED")
            visible = restarted_experience.initiative_calls[0][
                "cognitive_state"
            ]["authored_artifacts"]
            self.assertLessEqual(len(json.dumps(visible)), 16_384)
            self.assertEqual(visible["retained_working_set_count"], 1)
            self.assertEqual(
                visible["recent_artifacts"][0]["body"], first_action.body
            )
            self.assertEqual(
                visible["active_artifact_refs"], [first_action.artifact_ref]
            )
            revised_state = json.loads(restarted.state_bytes())
            self.assertEqual(len(revised_state["authored_artifacts"]), 2)
            revision = revised_state["authored_artifacts"][-1]
            self.assertEqual(revision["artifact_ref"], second_action.artifact_ref)
            self.assertEqual(revision["predecessor_version"], 1)
            self.assertEqual(
                revision["artifact"]["supersedes_ref"], first_action.artifact_ref
            )
            self.assertEqual(restarted_controller.update_calls, 0)
            self.assertEqual(revised_state["affordance_utility"], {})
            self.assertEqual(revised_state["memory_utility"], {})
            revised_bytes = restarted.state_bytes()
            revised_head = restarted.state_head()
            replay = restarted.scheduler_tick("authored-artifact:first")
            self.assertEqual(replay.episode_ref, first.episode_ref)
            self.assertEqual(restarted.state_bytes(), revised_bytes)
            self.assertEqual(restarted.state_head(), revised_head)

    def test_authored_artifact_rejects_foreign_supersession_without_state_advance(
        self,
    ) -> None:
        foreign = _authored_artifact_action(
            version=2,
            supersedes_ref="sha256:" + "f" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            controller = _AuthoredArtifactController((foreign,))
            executor = _CountingAuthoredArtifactExecutor()
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "authored-artifact-foreign.sqlite3",
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=_initiative_state(),
                clock=_artifact_test_clock(),
                cycle=HigherLevelAutonomyCycleAdapter(
                    memory=_Memory(),
                    experience_model=_AuthoredArtifactExperience(),
                    controller=controller,
                ),
                affordances=_authored_artifact_registry(executor),
            )
            before_state = supervisor.state_bytes()
            before_head = supervisor.state_head()

            with self.assertRaisesRegex(
                ValueError,
                "revision must target a retained active head",
            ):
                supervisor.scheduler_tick("authored-artifact:foreign")

            self.assertEqual(supervisor.state_bytes(), before_state)
            after_head = supervisor.state_head()
            self.assertEqual(after_head.state_ref, before_head.state_ref)
            self.assertEqual(
                after_head.moving_origin_ordinal,
                before_head.moving_origin_ordinal,
            )
            self.assertEqual(after_head.last_event_ref, before_head.last_event_ref)
            self.assertIsNone(supervisor.pending_bytes())
            self.assertEqual(executor.calls, 0)
            self.assertEqual(controller.update_calls, 0)

    def test_authored_artifact_rejects_invented_evidence_and_episode_references(
        self,
    ) -> None:
        cases = (
            (
                "evidence",
                _authored_artifact_action(
                    evidence_refs=("sha256:" + "d" * 64,),
                ),
                "evidence absent from the exact choice context",
            ),
            (
                "source-episode",
                _authored_artifact_action(
                    source_episode_refs=("sha256:" + "e" * 64,),
                ),
                "source episodes absent from retrieved context",
            ),
        )
        for label, invented, expected_error in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                controller = _AuthoredArtifactController((invented,))
                executor = _CountingAuthoredArtifactExecutor()
                supervisor = PersistentAutonomySupervisor(
                    Path(directory) / "authored-artifact-invented-reference.sqlite3",
                    genesis=JennyGenesis.owner_approved(
                        created_at_utc="2026-09-02T12:00:00Z"
                    ),
                    initial_state=_initiative_state(),
                    clock=_artifact_test_clock(),
                    cycle=HigherLevelAutonomyCycleAdapter(
                        memory=_Memory(),
                        experience_model=_AuthoredArtifactExperience(),
                        controller=controller,
                    ),
                    affordances=_authored_artifact_registry(executor),
                )
                before_state = supervisor.state_bytes()
                before_head = supervisor.state_head()

                with self.assertRaisesRegex(ValueError, expected_error):
                    supervisor.scheduler_tick(
                        f"authored-artifact:invented-{label}"
                    )

                self.assertEqual(supervisor.state_bytes(), before_state)
                self.assertEqual(
                    supervisor.state_head().state_ref, before_head.state_ref
                )
                self.assertIsNone(supervisor.pending_bytes())
                self.assertEqual(executor.calls, 0)
                self.assertEqual(controller.update_calls, 0)

    def test_autonomous_artifact_without_formation_evidence_rejects_pre_permission(
        self,
    ) -> None:
        state = json.dumps(
            {
                "affordance_utility": {},
                "memory_utility": {},
                "next_internal_request": "",
                "affordance_signals": {},
                "motivation_weights": {},
                "situated_state": {
                    "world_model": ["One evidence-bound question remains unresolved."],
                    "self_model": ["Private authorship is available but optional."],
                    "focus": ["Ground any authored work in the unresolved evidence."],
                    "unfinished_patterns": ["What work follows from this evidence?"],
                },
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        action_without_formation_evidence = _authored_artifact_action()
        with tempfile.TemporaryDirectory() as directory:
            experience = _ToolBlindInitiativeExperience(
                request=(
                    "Choose whether the unresolved evidence warrants private "
                    "authorship."
                )
            )
            controller = _AuthoredArtifactController(
                (action_without_formation_evidence,)
            )
            executor = _CountingAuthoredArtifactExecutor()
            permission_gate = _CountingPermissionGate()
            supervisor = PersistentAutonomySupervisor(
                Path(directory)
                / "authored-artifact-missing-formation-evidence.sqlite3",
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=state,
                clock=_artifact_test_clock(),
                cycle=HigherLevelAutonomyCycleAdapter(
                    memory=_Memory(),
                    experience_model=experience,
                    controller=controller,
                ),
                affordances=_authored_artifact_registry(executor),
                permission_gate=permission_gate,
            )
            before_state = supervisor.state_bytes()
            before_head = supervisor.state_head()

            with self.assertRaisesRegex(
                ValueError,
                "authored artifact does not cite its target-formation evidence",
            ):
                supervisor.scheduler_tick(
                    "authored-artifact:missing-formation-evidence"
                )

            self.assertEqual(len(experience.formation_calls), 1)
            self.assertEqual(permission_gate.calls, 0)
            self.assertEqual(executor.calls, 0)
            self.assertEqual(supervisor.state_bytes(), before_state)
            self.assertEqual(supervisor.state_head(), before_head)
            self.assertIsNone(supervisor.pending_bytes())
            self.assertEqual(controller.update_calls, 0)

    def test_authored_artifact_context_rejects_tamper_and_unbounded_history(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first_action = _authored_artifact_action()
            producer = PersistentAutonomySupervisor(
                Path(directory) / "authored-artifact-source.sqlite3",
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=_initiative_state(),
                clock=_artifact_test_clock(),
                cycle=HigherLevelAutonomyCycleAdapter(
                    memory=_Memory(),
                    experience_model=_AuthoredArtifactExperience(),
                    controller=_AuthoredArtifactController((first_action,)),
                ),
                affordances=_authored_artifact_registry(),
            )
            producer.scheduler_tick("authored-artifact:source")
            tampered = json.loads(producer.state_bytes())
            tampered["authored_artifacts"][0]["artifact"]["body"] += " altered"
            tampered_state = json.dumps(
                tampered, separators=(",", ":"), sort_keys=True
            ).encode()
            reader = PersistentAutonomySupervisor(
                Path(directory) / "authored-artifact-tamper.sqlite3",
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=tampered_state,
                clock=_artifact_test_clock(),
                cycle=HigherLevelAutonomyCycleAdapter(
                    memory=_Memory(),
                    experience_model=_AuthoredArtifactExperience(),
                    controller=_AuthoredArtifactController(()),
                ),
                affordances=_authored_artifact_registry(),
            )
            with self.assertRaisesRegex(ValueError, "artifact identity differs"):
                reader.scheduler_tick("authored-artifact:tampered")

            oversized = json.loads(producer.state_bytes())
            oversized["authored_artifacts"] = [
                oversized["authored_artifacts"][0]
            ] * 257
            oversized_state = json.dumps(
                oversized, separators=(",", ":"), sort_keys=True
            ).encode()
            bounded_reader = PersistentAutonomySupervisor(
                Path(directory) / "authored-artifact-unbounded.sqlite3",
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=oversized_state,
                clock=_artifact_test_clock(),
                cycle=HigherLevelAutonomyCycleAdapter(
                    memory=_Memory(),
                    experience_model=_AuthoredArtifactExperience(),
                    controller=_AuthoredArtifactController(()),
                ),
                affordances=_authored_artifact_registry(),
            )
            with self.assertRaisesRegex(ValueError, "at most 256 entries"):
                bounded_reader.scheduler_tick("authored-artifact:unbounded")

    def test_library_catalog_and_passages_form_one_resumable_uncredited_cycle(self):
        list_action = {
            "operation": "list",
            "reading_purpose": "Select attributable evidence for a bounded inquiry.",
        }
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            executor = _synthetic_library(base)
            experience = _LibraryExperience()
            controller = _LibraryController((list_action,))
            state_path = base / "library-cycle.sqlite3"
            supervisor = PersistentAutonomySupervisor(
                state_path,
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=_initiative_state(),
                clock=_clock(),
                cycle=HigherLevelAutonomyCycleAdapter(
                    memory=_Memory(),
                    experience_model=experience,
                    controller=controller,
                ),
                affordances=_library_registry(executor),
            )

            listed = supervisor.scheduler_tick("library:list")

            self.assertEqual(listed.status, "COMMITTED")
            listed_episode, listed_observed = _episode_observation(supervisor, listed)
            listed_payload = json.loads(listed_observed.observation_json)
            self.assertEqual(listed_payload["contract"], LIBRARY_CATALOG_CONTRACT)
            learned = json.loads(supervisor.state_bytes())
            catalog = learned["library_reading_state"]["catalog"]
            self.assertEqual(catalog["catalog_ref"], executor.catalog_ref)
            self.assertEqual(catalog["manifest_ref"], executor.manifest_ref)
            self.assertEqual(catalog["eligible_items"], listed_payload["eligible_items"])
            self.assertEqual(
                [item["item_path"] for item in catalog["eligible_items"]],
                ["works/first.txt", "works/second.txt"],
            )
            self.assertEqual(catalog["excluded_catalog_items"], 1)
            self.assertEqual(
                catalog["last_observation_ref"], listed_observed.observation_ref
            )
            listed_temporal = TemporalV2(**listed_episode["temporal"])
            self.assertEqual(catalog["temporal_ref"], listed_temporal.temporal_ref)
            self.assertEqual(
                catalog["moving_origin_ordinal"],
                listed_temporal.moving_origin_ordinal,
            )
            self.assertEqual(learned["affordance_utility"], {})
            self.assertEqual(learned["memory_utility"], {})
            self.assertNotIn("affordance_outcome_profiles", learned)
            self.assertNotIn("capability_evidence", learned)
            self.assertNotIn("capability_use_evidence", learned)
            self.assertEqual(controller.update_calls, 0)

            controller.actions.append(
                {
                    "cursor": 0,
                    "item_path": "works/first.txt",
                    "max_chars": 7,
                    "operation": "read",
                    "reading_purpose": "Read the first exact passage before interpretation.",
                }
            )
            first = supervisor.scheduler_tick("library:read:first")
            self.assertEqual(first.status, "COMMITTED")
            first_episode, first_observed = _episode_observation(supervisor, first)
            first_passage = library_observation_from_observable(first_observed)
            self.assertEqual(
                json.loads(first_observed.observation_json)["contract"],
                LIBRARY_PASSAGE_CONTRACT,
            )
            self.assertEqual(first_passage.span_start, 0)
            self.assertEqual(first_passage.next_cursor, 7)
            learned = json.loads(supervisor.state_bytes())
            progress = learned["library_reading_state"]["progress"][
                "works/first.txt"
            ]
            self.assertEqual(progress["last_span"], {"end": 7, "start": 0})
            self.assertEqual(progress["next_cursor"], first_passage.next_cursor)
            self.assertEqual(progress["artifact_ref"], first_passage.artifact_ref)
            first_temporal = TemporalV2(**first_episode["temporal"])
            self.assertEqual(progress["temporal_ref"], first_temporal.temporal_ref)
            self.assertEqual(
                progress["moving_origin_ordinal"], first_temporal.moving_origin_ordinal
            )
            self.assertEqual(
                experience.assessment_calls[-1]["observable_consequence"][
                    "observation"
                ]["content"],
                first_passage.content,
            )
            self.assertEqual(
                learned["situated_state"]["epistemic_status"],
                "MODEL_INFERENCE_FROM_SOURCE_BOUND_OBSERVATION_WITHOUT_VALUE_CREDIT",
            )
            self.assertEqual(learned["affordance_utility"], {})
            self.assertNotIn("capability_evidence", learned)
            self.assertEqual(
                experience.initiative_calls[1]["cognitive_state"][
                    "library_reading_state"
                ]["catalog"]["catalog_ref"],
                executor.catalog_ref,
            )

            committed_state = supervisor.state_bytes()
            committed_head = supervisor.state_head()
            resumed_experience = _LibraryExperience()
            resumed_controller = _LibraryController(
                (
                    {
                        "cursor": first_passage.next_cursor,
                        "item_path": "works/first.txt",
                        "max_chars": 99,
                        "operation": "read",
                        "reading_purpose": "Resume from the exact committed cursor.",
                    },
                )
            )
            restarted = PersistentAutonomySupervisor(
                state_path,
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=_initiative_state(),
                clock=_clock(),
                cycle=HigherLevelAutonomyCycleAdapter(
                    memory=_Memory(),
                    experience_model=resumed_experience,
                    controller=resumed_controller,
                ),
                affordances=_library_registry(executor),
            )
            self.assertEqual(restarted.state_bytes(), committed_state)
            self.assertEqual(restarted.state_head(), committed_head)

            second = restarted.scheduler_tick("library:read:second")

            self.assertEqual(second.status, "COMMITTED")
            _, second_observed = _episode_observation(restarted, second)
            second_passage = library_observation_from_observable(second_observed)
            self.assertEqual(second_passage.span_start, first_passage.next_cursor)
            self.assertEqual(
                first_passage.content + second_passage.content,
                "Alpha beta gamma delta epsilon.",
            )
            self.assertTrue(second_passage.eof)
            visible_progress = resumed_experience.initiative_calls[0][
                "cognitive_state"
            ]["library_reading_state"]["progress"]["works/first.txt"]
            self.assertEqual(visible_progress["next_cursor"], first_passage.next_cursor)
            self.assertEqual(
                resumed_controller.states[0]["library_reading_state"]["progress"]
                ["works/first.txt"]["next_cursor"],
                first_passage.next_cursor,
            )
            final_state = json.loads(restarted.state_bytes())
            final_progress = final_state["library_reading_state"]["progress"][
                "works/first.txt"
            ]
            self.assertEqual(final_progress["next_cursor"], second_passage.next_cursor)
            self.assertEqual(final_progress["last_span"]["start"], first_passage.next_cursor)
            self.assertEqual(final_state["affordance_utility"], {})
            self.assertNotIn("capability_evidence", final_state)

    def test_r1_scale_sequential_library_context_is_bounded_and_attributable(self):
        actions = (
            {
                "operation": "list",
                "reading_purpose": (
                    "Select attributable evidence for a bounded inquiry."
                ),
            },
            {
                "cursor": 0,
                "item_path": "works/first.txt",
                "max_chars": 7,
                "operation": "read",
                "reading_purpose": (
                    "Read the first exact passage before interpretation."
                ),
            },
            {
                "cursor": 7,
                "item_path": "works/first.txt",
                "max_chars": 99,
                "operation": "read",
                "reading_purpose": "Resume from the exact committed cursor.",
            },
        )
        oversized_history_record = {
            "epistemic_status": (
                "MODEL_INFERENCE_FROM_SOURCE_BOUND_OBSERVATION_WITHOUT_VALUE_CREDIT"
            ),
            "moving_origin_ordinal": 40,
            "observation_ref": "sha256:" + "d" * 64,
            "reasoned_judgment": "x" * 40_000,
        }
        initial_payload = json.loads(_initiative_state())
        initial_payload["outcome_assessment_evidence"] = [
            oversized_history_record
        ]
        initial_state = json.dumps(
            initial_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            executor = _synthetic_library(base)
            experience = _LibraryExperience()
            controller = _LibraryController(actions)
            state_path = base / "r1-scale-library-context.sqlite3"
            supervisor = PersistentAutonomySupervisor(
                state_path,
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=initial_state,
                clock=_clock(),
                cycle=HigherLevelAutonomyCycleAdapter(
                    memory=_Memory(),
                    experience_model=experience,
                    controller=controller,
                ),
                affordances=_library_registry(executor),
            )

            contexts = []
            observations = []
            triggers = (
                "library:r1-scale:list",
                "library:r1-scale:first",
                "library:r1-scale:continue",
            )
            for trigger in triggers:
                parent_state = supervisor.state_bytes()
                parent_payload = json.loads(parent_state)
                result = supervisor.scheduler_tick(trigger)
                self.assertEqual(result.status, "COMMITTED")
                episode, observed = _episode_observation(supervisor, result)
                context_json = episode["choice"]["context_json"]
                self.assertLessEqual(len(context_json), 65_536)
                context = json.loads(context_json)
                contexts.append(context)
                observations.append(observed)

                cognitive = context["cognitive_state"]
                working_set = cognitive["working_set"]
                self.assertEqual(
                    working_set["contract"],
                    "jenny.choice-cognitive-working-set.v1",
                )
                self.assertEqual(
                    working_set["canonical_state_ref"],
                    "sha256:" + hashlib.sha256(parent_state).hexdigest(),
                )
                canonical_history = parent_payload[
                    "outcome_assessment_evidence"
                ]
                metadata = working_set["history_channels"][
                    "outcome_assessment_evidence"
                ]
                self.assertEqual(metadata["canonical_count"], len(canonical_history))
                self.assertEqual(
                    metadata["included_count"], min(2, len(canonical_history))
                )
                self.assertEqual(
                    metadata["representation"],
                    "HASH_LINKED_BOUNDED_PROJECTIONS",
                )
                projected = cognitive["outcome_assessment_evidence"]
                expected_recent = canonical_history[-2:]
                self.assertEqual(
                    [item["canonical_record_ref"] for item in projected],
                    [content_ref(item) for item in expected_recent],
                )
                self.assertTrue(
                    all(
                        item["contract"]
                        == "jenny.choice-history-record-projection.v1"
                        and len(item["bounded_excerpt"]) <= 384
                        for item in projected
                    )
                )

            initial_projection = contexts[0]["cognitive_state"][
                "outcome_assessment_evidence"
            ][0]
            self.assertEqual(
                controller.states[0]["outcome_assessment_evidence"][0],
                oversized_history_record,
            )
            self.assertEqual(
                initial_projection["observation_ref"],
                oversized_history_record["observation_ref"],
            )
            self.assertEqual(
                initial_projection["epistemic_status"],
                oversized_history_record["epistemic_status"],
            )
            self.assertEqual(
                initial_projection["moving_origin_ordinal"],
                oversized_history_record["moving_origin_ordinal"],
            )
            listed_payload = json.loads(observations[0].observation_json)
            self.assertEqual(listed_payload["contract"], LIBRARY_CATALOG_CONTRACT)
            first_passage = library_observation_from_observable(observations[1])
            second_passage = library_observation_from_observable(observations[2])
            self.assertEqual(first_passage.source_ref, executor.source_ref)
            self.assertEqual(first_passage.catalog_ref, executor.catalog_ref)
            self.assertEqual(first_passage.manifest_ref, executor.manifest_ref)
            self.assertEqual(first_passage.next_cursor, 7)
            visible_progress = contexts[2]["cognitive_state"][
                "library_reading_state"
            ]["progress"]["works/first.txt"]
            self.assertEqual(visible_progress["next_cursor"], first_passage.next_cursor)
            self.assertEqual(visible_progress["artifact_ref"], first_passage.artifact_ref)
            self.assertEqual(
                visible_progress["last_observation_ref"],
                observations[1].observation_ref,
            )
            self.assertEqual(second_passage.span_start, first_passage.next_cursor)
            self.assertEqual(second_passage.source_ref, executor.source_ref)
            self.assertEqual(second_passage.catalog_ref, executor.catalog_ref)
            self.assertEqual(second_passage.manifest_ref, executor.manifest_ref)
            self.assertEqual(second_passage.artifact_ref, first_passage.artifact_ref)
            self.assertTrue(second_passage.eof)

            committed_state = supervisor.state_bytes()
            committed_head = supervisor.state_head()
            final_state = json.loads(committed_state)
            self.assertEqual(
                final_state["outcome_assessment_evidence"][0],
                oversized_history_record,
            )
            self.assertEqual(len(final_state["outcome_assessment_evidence"]), 4)
            final_progress = final_state["library_reading_state"]["progress"][
                "works/first.txt"
            ]
            self.assertEqual(final_progress["next_cursor"], second_passage.next_cursor)
            self.assertEqual(final_progress["last_span"]["start"], first_passage.next_cursor)
            self.assertEqual(final_progress["artifact_ref"], first_passage.artifact_ref)
            self.assertEqual(final_state["affordance_utility"], {})
            self.assertEqual(final_state["memory_utility"], {})
            self.assertNotIn("capability_evidence", final_state)
            self.assertEqual(controller.update_calls, 0)

            restarted = PersistentAutonomySupervisor(
                state_path,
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=_initiative_state(),
                clock=_clock(),
                cycle=HigherLevelAutonomyCycleAdapter(
                    memory=_Memory(),
                    experience_model=_LibraryExperience(),
                    controller=_LibraryController(()),
                ),
                affordances=_library_registry(executor),
            )
            self.assertEqual(restarted.state_bytes(), committed_state)
            self.assertEqual(restarted.state_head(), committed_head)

    def test_library_invalid_observations_and_cursor_cannot_advance_state(self):
        cases = ("foreign", "tampered", "bad-cursor")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                real = _synthetic_library(base)

                def executor(request, *, mode=case):
                    if mode == "bad-cursor":
                        return real(request)
                    receipt = real(request)
                    observed = receipt.observable_consequence
                    assert observed is not None
                    if mode == "foreign":
                        replacement = ObservableConsequence(
                            request_ref=observed.request_ref,
                            source_kind=observed.source_kind,
                            source_ref="sha256:" + "f" * 64,
                            observation_json=observed.observation_json,
                            artifact_refs=observed.artifact_refs,
                            evidence_refs=observed.evidence_refs,
                        )
                    else:
                        payload = json.loads(observed.observation_json)
                        payload["next_cursor"] += 1
                        replacement = ObservableConsequence(
                            request_ref=observed.request_ref,
                            source_kind=observed.source_kind,
                            source_ref=observed.source_ref,
                            observation_json=json.dumps(
                                payload,
                                ensure_ascii=False,
                                allow_nan=False,
                                separators=(",", ":"),
                                sort_keys=True,
                            ),
                            artifact_refs=observed.artifact_refs,
                            evidence_refs=observed.evidence_refs,
                        )
                    return AffordanceReceipt(
                        receipt.status,
                        receipt.output,
                        receipt.consequence,
                        replacement,
                    )

                action = {
                    "cursor": 9_999 if case == "bad-cursor" else 0,
                    "item_path": "works/first.txt",
                    "max_chars": 7,
                    "operation": "read",
                    "reading_purpose": "Attempt one bounded source read.",
                }
                controller = _LibraryController((action,))
                registry = DynamicAffordanceRegistry()
                registry.register(
                    LIBRARY_AFFORDANCE,
                    executor,
                    observable_source_ref=real.source_ref,
                    observable_source_kind=LIBRARY_SOURCE_KIND,
                )
                supervisor = PersistentAutonomySupervisor(
                    base / f"library-{case}.sqlite3",
                    genesis=JennyGenesis.owner_approved(
                        created_at_utc="2026-09-02T12:00:00Z"
                    ),
                    initial_state=_state(),
                    clock=_clock(),
                    cycle=HigherLevelAutonomyCycleAdapter(
                        memory=_Memory(),
                        experience_model=_ObservedOutcomeExperience(),
                        controller=controller,
                    ),
                    affordances=registry,
                )
                before_state = supervisor.state_bytes()
                before_head = supervisor.state_head()
                if case == "bad-cursor":
                    result = supervisor.scheduler_tick(
                        f"library:invalid:{case}"
                    )
                    self.assertEqual(result.status, "COMMITTED")
                    episode = json.loads(
                        supervisor.episode_item(result.episode_ref).payload_json
                    )
                    self.assertEqual(episode["receipt"]["status"], "ERROR")
                    self.assertIn("cursor", episode["receipt"]["output"])
                    learned = json.loads(supervisor.state_bytes())
                    self.assertNotIn("library_reading_state", learned)
                    self.assertEqual(learned["affordance_utility"], {})
                    self.assertEqual(learned["memory_utility"], {})
                    self.assertEqual(
                        learned["operational_receipt_evidence"][-1]["status"],
                        "ERROR",
                    )
                    self.assertIsNone(supervisor.pending_bytes())
                    continue

                with self.assertRaises(ValueError):
                    supervisor.scheduler_tick(f"library:invalid:{case}")
                self.assertEqual(supervisor.state_bytes(), before_state)
                after_head = supervisor.state_head()
                self.assertEqual(after_head.state_ref, before_head.state_ref)
                self.assertEqual(
                    after_head.moving_origin_ordinal,
                    before_head.moving_origin_ordinal,
                )
                self.assertEqual(after_head.last_event_ref, before_head.last_event_ref)
                self.assertNotIn(
                    "library_reading_state", json.loads(supervisor.state_bytes())
                )

    def test_reference_candidates_are_distrustfully_rejoined_to_canonical_bytes(self):
        class Backend:
            projected = []

            def search(self, request, *, limit):
                return (ReferenceMemoryHit(self.record_ref, 0.05),)

            def project(self, **value):
                self.projected.append(value)
                self.record_ref = value["record_ref"]
                return "cognee:synthetic-projection"

        with tempfile.TemporaryDirectory() as directory:
            memory, experience, cortex = _Memory(), _Experience(), _Cortex()
            adapter = HigherLevelAutonomyCycleAdapter(
                memory=memory,
                experience_model=experience,
                controller=OutcomeUtilityAffordanceController(
                    qualification_ref=QUALIFICATION
                ),
            )
            registry = DynamicAffordanceRegistry()
            registry.register(
                Affordance("cortex.respond", "ACT", "Run cortex.", "internal.cognition"),
                HigherLevelCortexAffordanceExecutor(cortex, _consequence),
            )
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "reference.sqlite3",
                genesis=JennyGenesis.owner_approved(created_at_utc="2026-09-02T12:00:00Z"),
                initial_state=_state(),
                clock=_clock(),
                cycle=adapter,
                affordances=registry,
            )
            supervisor.human_ingress("reference:seed", "Seed canonical evidence.")
            backend = Backend()
            canonical = SupervisorCanonicalMemory(supervisor)
            augmented = ReferenceAugmentedSemanticMemory(canonical, backend)
            pending = supervisor.pending_projections()
            supervisor.retry_pending_projections(
                SemanticMemoryConsolidationProjector(augmented)
            )
            recalled = augmented.recall("candidate", limit=4)
            self.assertEqual(recalled[0].record_ref, pending[0].episode_ref)
            self.assertEqual(recalled[0].content, canonical.get(pending[0].episode_ref).content)
            self.assertEqual(recalled[0].acquired_time_utc, "2026-09-02T12:00:00.002000Z")
            self.assertEqual(recalled[0].timezone, "America/New_York")

            backend.record_ref = "sha256:" + "f" * 64
            with self.assertRaises(KeyError):
                augmented.recall("tampered", limit=4)

    def test_observed_procedure_closes_the_cognee_to_model_loop_with_removal(self):
        class EvidenceAwareBackend(_ControllerBackend):
            def generate(self, *, system, user, max_new_tokens):
                self.calls.append((system, user, max_new_tokens))
                request = json.loads(user)
                value = dict(self.output)
                value["evidence_refs"] = [request["allowed_evidence_refs"][0]]
                return json.dumps(value)

        class Backend:
            def __init__(self):
                self.projected = []

            def search(self, request, *, limit):
                del request
                return tuple(
                    ReferenceMemoryHit(item["record_ref"], 0.05)
                    for item in self.projected[-limit:]
                )

            def project(self, **value):
                self.projected.append(value)
                return "cognee:procedural-case"

        class EmptyBackend:
            def search(self, request, *, limit):
                del request, limit
                return ()

            def project(self, **value):
                raise AssertionError("the removal arm must not project")

        class CapturingExperience(_Experience):
            def __init__(self):
                super().__init__()
                self.memory_inputs = []

            def generate_experience(self, request, memories, temporal):
                self.memory_inputs.append(tuple(memories))
                return super().generate_experience(request, memories, temporal)

        with tempfile.TemporaryDirectory() as directory:
            deferred = DeferredSemanticMemory()
            experience = CapturingExperience()
            cortex = _Cortex()
            controller_backend = EvidenceAwareBackend(
                {
                    "selected_affordance_id": "cortex.respond",
                    "rankings": [
                        {"affordance_id": "cortex.respond", "score": 0.7}
                    ],
                    "action_payload": "Use relevant evidence, then verify the result.",
                    "state_assessment": "One bounded request can be attempted.",
                    "resolution_target": "Produce and evaluate one attributable result.",
                    "expected_state_delta": "The receipt will expose the outcome.",
                    "evidence_refs": [],
                }
            )
            adapter = HigherLevelAutonomyCycleAdapter(
                memory=deferred,
                experience_model=experience,
                controller=FrozenModelAffordanceController(
                    controller_backend, qualification_ref=QUALIFICATION
                ),
            )
            registry = DynamicAffordanceRegistry()
            registry.register(
                Affordance(
                    "cortex.respond", "ACT", "Run cortex.", "internal.cognition"
                ),
                HigherLevelCortexAffordanceExecutor(cortex, _consequence),
            )
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "procedural-loop.sqlite3",
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=_state(),
                clock=_clock(),
                cycle=adapter,
                affordances=registry,
            )
            backend = Backend()
            canonical = SupervisorCanonicalMemory(supervisor)
            augmented = ReferenceAugmentedSemanticMemory(canonical, backend)
            deferred.bind(augmented)

            first = supervisor.human_ingress(
                "procedural-loop:first",
                "Resolve one bounded source form and retain what worked.",
            )
            self.assertEqual(first.status, "COMMITTED")
            self.assertEqual(experience.memory_inputs[0], ())
            supervisor.retry_pending_projections(
                SemanticMemoryConsolidationProjector(augmented)
            )
            self.assertEqual(len(backend.projected), 1)
            projected = json.loads(backend.projected[0]["content"])
            self.assertEqual(projected["memory_kind"], "PROCEDURAL_EXPERIENCE")
            self.assertEqual(
                projected["epistemic_status"], "OBSERVED_PROCEDURAL_OUTCOME"
            )

            second = supervisor.human_ingress(
                "procedural-loop:second",
                "Apply the comparable procedure to a fresh bounded form.",
            )
            self.assertEqual(second.status, "COMMITTED")
            learned_memories = experience.memory_inputs[1]
            self.assertEqual(len(learned_memories), 1)
            self.assertEqual(learned_memories[0].record_ref, first.episode_ref)
            self.assertEqual(
                json.loads(learned_memories[0].content)["case_ref"],
                projected["case_ref"],
            )
            second_controller_input = json.loads(controller_backend.calls[1][1])
            self.assertEqual(
                second_controller_input["memories"][0]["record_ref"],
                first.episode_ref,
            )
            self.assertEqual(cortex.calls[1][2][0].record_ref, first.episode_ref)

            removed = ReferenceAugmentedSemanticMemory(canonical, EmptyBackend())
            self.assertEqual(
                removed.recall(
                    "Apply the comparable procedure to a fresh bounded form.",
                    limit=4,
                ),
                (),
            )

            bootstrap = CapabilityAwareConsolidationProjector(
                SemanticMemoryConsolidationProjector(augmented),
                supervisor,
                backend,
            )
            bootstrapped = bootstrap.bootstrap_procedural_cases()
            self.assertEqual(len(bootstrapped), 2)
            self.assertEqual(len(backend.projected), 3)
            for value in backend.projected[-2:]:
                self.assertEqual(
                    json.loads(value["content"])["memory_kind"],
                    "PROCEDURAL_EXPERIENCE",
                )

    def test_canonical_memory_rejoins_candidates_beyond_first_store_page(self):
        items = []
        for ordinal in range(300):
            episode_ref = "sha256:" + f"{ordinal + 1:064x}"
            event_ref = "sha256:" + f"{ordinal + 1001:064x}"
            content = (
                "LATEPAGE-ANCHOR retained in the newest canonical episode."
                if ordinal == 299
                else f"Ordinary canonical memory {ordinal}."
            )
            payload = {
                "receipt": {"status": "COMPLETED", "consequence": []},
                "consolidation_proposal": content,
                "temporal": {
                    "event_time_utc": None,
                    "acquired_time_utc": None,
                    "recorded_time_utc": None,
                    "verified_time_utc": None,
                    "valid_from_utc": None,
                    "valid_until_utc": None,
                    "timezone": None,
                },
            }
            items.append(
                SimpleNamespace(
                    episode_ref=episode_ref,
                    event_ref=event_ref,
                    ordinal=ordinal,
                    payload_json=json.dumps(payload),
                )
            )

        class PagedSupervisor:
            def __init__(self, records):
                self.records = records
                self.calls = []

            def episode_items(self, *, after_ordinal=-1, limit=256):
                self.calls.append((after_ordinal, limit))
                return tuple(
                    item for item in self.records if item.ordinal > after_ordinal
                )[:limit]

            def episode_item(self, episode_ref):
                return next(
                    item for item in self.records if item.episode_ref == episode_ref
                )

        supervisor = PagedSupervisor(items)
        canonical = SupervisorCanonicalMemory(supervisor)
        newest = canonical.get(items[-1].episode_ref)
        self.assertIn("LATEPAGE-ANCHOR", newest.content)
        recalled = canonical.recall("LATEPAGE-ANCHOR", limit=1)
        self.assertEqual(recalled[0].record_ref, items[-1].episode_ref)
        self.assertEqual(
            canonical.store(
                newest.content,
                (items[-1].episode_ref, items[-1].event_ref),
            ),
            items[-1].episode_ref,
        )
        self.assertIn((255, 256), supervisor.calls)

    def test_large_capability_catalog_retrieves_oldest_relevant_head(self):
        initial = json.loads(_state())
        capabilities = []
        for index in range(300):
            record = {
                "capability_key": f"capability.catalog-{index:03d}",
                "revision_index": 1,
                "supersedes_capability_ref": None,
                "procedure": f"Apply catalog procedure {index}.",
                "applicability": "A matching bounded case.",
                "limits": "One synthetic catalog entry.",
            }
            record["capability_ref"] = "sha256:" + hashlib.sha256(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            capabilities.append(record)
        initial["capability_evidence"] = capabilities
        first = capabilities[0]

        class CapabilityBackend:
            def __init__(self):
                self.calls = []

            def search(self, request, *, limit):
                self.calls.append((request, limit))
                return (CapabilityModuleHit(first["capability_ref"], 0.01),)

            def project(self, **value):
                raise AssertionError("retrieval must not write the capability index")

        capability_backend = CapabilityBackend()
        backend = _ControllerBackend(
            {
                "selected_affordance_id": "cortex.respond",
                "action_payload": "Apply the oldest matching catalog capability.",
                "requires_extended_deliberation": False,
                "compute_rationale": "The exact matching capability is available.",
                "state_assessment": "The oldest catalog capability remains applicable.",
                "resolution_target": "Exercise retained catalog reachability.",
                "expected_state_delta": "The receipt will test the retained module.",
                "evidence_refs": [first["capability_ref"]],
                "selected_candidate_id": "use-oldest-catalog-capability",
                "candidate_preference_order": ["use-oldest-catalog-capability"],
                "affordance_preference_order": ["cortex.respond"],
                "selection_basis": "The oldest catalog entry exactly matches the bounded request.",
                "intent_candidates": [
                    {
                        "candidate_id": "use-oldest-catalog-capability",
                        "affordance_id": "cortex.respond",
                        "proposed_action": "Apply the oldest matching catalog capability.",
                        "desired_state_change": "Exercise retained catalog reachability.",
                        "rationale": "Its stated applicability matches.",
                        "predicted_consequences": ["The receipt will expose applicability."],
                        "unknowns": [],
                        "reversibility": "REVERSIBLE",
                        "required_capability_keys": [first["capability_key"]],
                        "evidence_refs": [first["capability_ref"]],
                        "reasons_for": ["The module remains active."],
                        "reasons_against": [],
                    }
                ],
            }
        )
        adapter = HigherLevelAutonomyCycleAdapter(
            memory=_Memory(),
            experience_model=_Experience(),
            capability_catalog=ReferenceAugmentedCapabilityCatalog(
                CanonicalCapabilityCatalog(), capability_backend
            ),
            controller=FrozenModelAffordanceController(
                backend,
                draft_backend=backend,
                qualification_ref=QUALIFICATION,
                require_intent_candidates=True,
                maximum_output_tokens=2_048,
                maximum_draft_output_tokens=2_048,
            ),
        )
        registry = DynamicAffordanceRegistry()
        registry.register(
            Affordance("cortex.respond", "ACT", "Run cortex.", "internal.cognition"),
            HigherLevelCortexAffordanceExecutor(_Cortex(), _consequence),
        )
        with tempfile.TemporaryDirectory() as directory:
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "large-capability-catalog.sqlite3",
                genesis=JennyGenesis.owner_approved(
                    created_at_utc="2026-09-02T12:00:00Z"
                ),
                initial_state=json.dumps(
                    initial, separators=(",", ":"), sort_keys=True
                ).encode(),
                clock=_clock(),
                cycle=adapter,
                affordances=registry,
            )
            result = supervisor.scheduler_tick("capability-catalog:oldest")
            self.assertEqual(result.status, "COMMITTED")
            controller_input = json.loads(backend.calls[0][1])
            self.assertEqual(len(controller_input["capability_modules"]), 1)
            self.assertEqual(
                controller_input["capability_modules"][0]["capability_key"],
                first["capability_key"],
            )
            self.assertEqual(capability_backend.calls[0][1], 256)
            learned = json.loads(supervisor.state_bytes())
            self.assertEqual(len(learned["capability_evidence"]), 300)
            self.assertEqual(
                learned["capability_use_evidence"][-1]["capability_keys"],
                [first["capability_key"]],
            )

    def test_stale_and_unknown_capability_hits_are_not_remapped(self):
        stale_ref = "sha256:" + "1" * 64
        current = {
            "capability_key": "capability.current",
            "revision_index": 2,
            "supersedes_capability_ref": stale_ref,
            "procedure": "Use the corrected current procedure.",
            "applicability": "Current bounded cases.",
            "limits": "Requires an observable check.",
        }
        current["capability_ref"] = "sha256:" + hashlib.sha256(
            json.dumps(
                current, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ).encode()
        ).hexdigest()
        valid = {
            "capability_key": "capability.valid",
            "revision_index": 1,
            "supersedes_capability_ref": None,
            "procedure": "Use the independently relevant procedure.",
            "applicability": "The present bounded request.",
            "limits": "One observed form.",
        }
        valid["capability_ref"] = "sha256:" + hashlib.sha256(
            json.dumps(
                valid, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ).encode()
        ).hexdigest()
        payload = json.loads(_state())
        payload["capability_evidence"] = [current, valid]
        state = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()

        class Backend:
            def search(self, request, *, limit):
                return (
                    CapabilityModuleHit(stale_ref, 0.01),
                    CapabilityModuleHit("sha256:" + "f" * 64, 0.02),
                    CapabilityModuleHit(valid["capability_ref"], 0.03),
                )

            def project(self, **value):
                raise AssertionError("recall must not project")

        catalog = ReferenceAugmentedCapabilityCatalog(
            CanonicalCapabilityCatalog(), Backend()
        )
        recalled = catalog.recall("present bounded request", state=state, limit=4)
        self.assertEqual(
            [item["active_capability_ref"] for item in recalled],
            [valid["capability_ref"]],
        )
        self.assertNotIn(current["capability_ref"], catalog.last_rejected_refs)
        self.assertEqual(
            catalog.last_rejected_refs,
            (stale_ref, "sha256:" + "f" * 64),
        )

    def test_stale_saturated_index_falls_back_without_revision_remapping(self):
        stale_ref = "sha256:" + "1" * 64

        def capability(key, revision, supersedes, procedure, applicability):
            record = {
                "capability_key": key,
                "revision_index": revision,
                "supersedes_capability_ref": supersedes,
                "procedure": procedure,
                "applicability": applicability,
                "limits": "Requires fresh observable evidence.",
            }
            record["capability_ref"] = "sha256:" + hashlib.sha256(
                json.dumps(
                    record, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                ).encode()
            ).hexdigest()
            return record

        current = capability(
            "capability.current", 2, stale_ref,
            "Use the current revision.", "Unrelated current cases."
        )
        valid = capability(
            "capability.valid", 1, None,
            "Apply the independently relevant procedure.",
            "The independently relevant present request.",
        )
        payload = json.loads(_state())
        payload["capability_evidence"] = [current, valid]
        state = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()

        class Backend:
            def search(self, request, *, limit):
                self.limit = limit
                unknown = tuple(
                    CapabilityModuleHit(
                        "sha256:" + f"{index + 1000:064x}", 0.02 + index / 1000
                    )
                    for index in range(255)
                )
                return (CapabilityModuleHit(stale_ref, 0.01), *unknown)

            def project(self, **value):
                raise AssertionError("recall must not project")

        backend = Backend()
        catalog = ReferenceAugmentedCapabilityCatalog(
            CanonicalCapabilityCatalog(), backend
        )
        recalled = catalog.recall(
            "independently relevant present request", state=state, limit=16
        )
        self.assertEqual(backend.limit, 256)
        self.assertEqual(
            [item["active_capability_ref"] for item in recalled],
            [valid["capability_ref"]],
        )
        self.assertNotIn(current["capability_ref"], catalog.last_rejected_refs)

    def test_malformed_active_capability_cannot_enter_reference_workspace(self):
        malformed = {
            "capability_key": "capability.malformed",
            "revision_index": 1,
            "supersedes_capability_ref": None,
            "procedure": None,
            "applicability": "A bounded case.",
            "limits": "Requires evidence.",
        }
        malformed["capability_ref"] = "sha256:" + hashlib.sha256(
            json.dumps(
                malformed, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ).encode()
        ).hexdigest()
        payload = json.loads(_state())
        payload["capability_evidence"] = [malformed]
        state = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        catalog = CanonicalCapabilityCatalog()
        with self.assertRaisesRegex(ValueError, "projection fields"):
            catalog.active_count(state=state)

    def test_empty_canonical_history_does_not_query_uncreated_cognee_collection(self):
        class Backend:
            calls = 0

            def search(self, request, *, limit):
                self.calls += 1
                raise AssertionError("empty canonical memory must bypass backend search")

            def project(self, **value):
                raise AssertionError("nothing may project before a canonical commit")

        class Cycle:
            qualification_ref = None

            def choose(self, **value):
                raise AssertionError("cycle choice is not part of this recall test")

            def learn(self, **value):
                raise AssertionError("cycle learning is not part of this recall test")

        with tempfile.TemporaryDirectory() as directory:
            registry = DynamicAffordanceRegistry()
            registry.register(
                Affordance("wait.context", "WAIT", "Wait.", "internal.cognition")
            )
            supervisor = PersistentAutonomySupervisor(
                Path(directory) / "empty.sqlite3",
                genesis=JennyGenesis.owner_approved(created_at_utc="2026-09-02T12:00:00Z"),
                initial_state=_state(),
                clock=_clock(),
                cycle=Cycle(),
                affordances=registry,
            )
            backend = Backend()
            augmented = ReferenceAugmentedSemanticMemory(
                SupervisorCanonicalMemory(supervisor), backend
            )
            self.assertEqual(augmented.recall("first request", limit=4), ())
            self.assertEqual(backend.calls, 0)

    def test_human_structured_affordance_keeps_validated_model_payload(self):
        temporal = _clock().sample(-1)
        payload = {
            "operation": "list",
            "reading_purpose": "Verify which library works are visible.",
        }
        backend = _ControllerBackend(
            {
                "selected_affordance_id": LIBRARY_AFFORDANCE_ID,
                "rankings": [
                    {"affordance_id": LIBRARY_AFFORDANCE_ID, "score": 0.9},
                    {"affordance_id": "cortex.respond", "score": 0.1},
                ],
                "action_payload": payload,
                "state_assessment": "The library has not yet been observed.",
                "resolution_target": "Observe the verified catalog.",
                "expected_state_delta": "Library availability becomes known.",
                "evidence_refs": [temporal.sample_ref],
            }
        )
        selection = FrozenModelAffordanceController(
            backend, qualification_ref=QUALIFICATION
        ).select(
            observation=CycleObservation(
                "human:library", "HUMAN", "Can you see your library?"
            ),
            temporal=temporal,
            affordances=(
                LIBRARY_AFFORDANCE,
                Affordance(
                    "cortex.respond", "ACT", "Respond.", "internal.cognition"
                ),
            ),
            memories=(),
            experience=_Experience().generate_experience(
                "Can you see your library?", (), type("T", (), {"now": 0})()
            ),
            state=_state(),
        )

        self.assertEqual(selection.selected_affordance_id, LIBRARY_AFFORDANCE_ID)
        self.assertEqual(
            selection.action_payload,
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )


if __name__ == "__main__":
    unittest.main()
