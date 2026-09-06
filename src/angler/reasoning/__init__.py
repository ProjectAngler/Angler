"""Trainable processing cores kept separate from foundation-model knowledge."""

from .adaptive_core import (
    AdaptiveFeedbackContext,
    AdaptiveFeedbackWrite,
    AdaptiveReasoningCore,
    AdaptiveReasoningTrajectory,
)
from .recurrent_core import (
    ReasoningCoreConfig,
    ReasoningTrajectory,
    RecurrentReasoningCore,
    reasoning_state_digest,
    restore_reasoning_state,
    snapshot_reasoning_state,
)
from .self_referential_memory import (
    SelfReferentialMemory,
    SelfReferentialState,
    detach_self_referential_state,
    restore_self_referential_state,
    self_referential_state_digest,
    snapshot_self_referential_state,
)
from .situated_reader import (
    LearnedSituatedMemoryReader,
    SituatedFeatureSpec,
    SituatedReaderOutput,
    encode_situated_features,
)
from .situated_feedback import (
    OMLSituatedFeedbackPolicy,
    SituatedFeedbackOutput,
    SituatedFeedbackPolicy,
    situated_outcome_loss,
)
from .outcome_aware_apprenticeship import (
    OutcomeAwareApprenticeshipCore,
    OutcomeAwareApprenticeshipOutput,
    OutcomeAwarePlasticState,
)
from .causal_neuromodulated_apprenticeship import (
    CausalNeuromodulatedApprenticeshipCore,
    CausalNeuromodulatedOutput,
)
from .content_addressed_causal_memory import (
    ContentAddressedCausalMemoryCore,
    ContentAddressedCausalMemoryOutput,
    ContentAddressedCausalMemoryState,
)
from .sparse_memory_causal_routing import SparseMemoryCausalRoutingCore
from .factorized_key_value_causal_memory import FactorizedKeyValueCausalMemoryCore
from .dnc_allocated_factorized_causal_memory import (
    DncAllocatedFactorizedCausalMemoryCore,
)
from .natural_trace_graph_causal_memory import (
    NaturalLanguageTraceGraphEncoder,
    NaturalTraceGraphCausalMemoryCore,
    NaturalTraceGraphCausalMemoryOutput,
    NaturalTraceGraphMemoryState,
    parse_step_trace,
)
from .oml_natural_trace_representation import (
    OMLNaturalTraceRepresentation,
    representation_metrics as natural_trace_representation_metrics,
)
from .structure_protected_oml import StructureProtectedOMLCore
from .structure_keyed_credit_memory import (
    StructureKeyedCreditEvent,
    StructureKeyedCreditMemoryCore,
    StructureKeyedCreditOutput,
    StructureKeyedCreditSnapshot,
    StructureKeyedCreditState,
)
from .prospective_dynamics import (
    CompositeProspectiveCreditCore,
    CompositeProspectiveCreditEvent,
    ProspectiveComponentStateIntegrity,
    ProspectiveDynamicsConfig,
    ProspectiveDynamicsSnapshot,
    ProspectiveDynamicsState,
    ProspectiveFocusOutput,
    ProspectiveResourceBudget,
)
from .scalable_procedural_core import (
    PlasticProcedureState,
    ProceduralCoreConfig,
    ProceduralCoreOutput,
    ScalableProceduralCore,
    plastic_state_digest,
    procedural_core_config,
    select_procedural_core_tier,
)
from .candidate_procedure_decoder import (
    CandidateProcedureDecode,
    CandidateProcedureDecoder,
    candidate_procedure_loss,
)
from .donor_candidate_fusion import DonorCandidateFusion
from .edge_aware_procedure_decoder import EdgeAwareActionTraceDecoder
from .role_aware_graph_decoder import (
    PUBLIC_ROLE_WIDTH,
    RoleAwareGraphProcedureDecoder,
    build_public_candidate_roles,
)
from .trace_conditioned_core import TraceConditionedProceduralCore
from .action_trace_matching import ActionTraceMatcherDecoder, ActionTraceProceduralCore
from .structured_relational_encoder import (
    PublicProcedureRelations,
    PublicRelationComponent,
    RelationalIncidenceTensors,
    SemanticRelationalFusion,
    StructuredActionTraceMatcherDecoder,
    StructuredActionTraceProceduralCore,
    StructuredRelationalEncoder,
    build_relational_incidence,
    parse_public_relations,
)

__all__ = [
    "AdaptiveFeedbackContext",
    "AdaptiveFeedbackWrite",
    "AdaptiveReasoningCore",
    "AdaptiveReasoningTrajectory",
    "ReasoningCoreConfig",
    "ReasoningTrajectory",
    "RecurrentReasoningCore",
    "reasoning_state_digest",
    "restore_reasoning_state",
    "snapshot_reasoning_state",
    "SelfReferentialMemory",
    "SelfReferentialState",
    "detach_self_referential_state",
    "restore_self_referential_state",
    "self_referential_state_digest",
    "snapshot_self_referential_state",
    "LearnedSituatedMemoryReader",
    "SituatedFeatureSpec",
    "SituatedReaderOutput",
    "encode_situated_features",
    "SituatedFeedbackOutput",
    "OMLSituatedFeedbackPolicy",
    "SituatedFeedbackPolicy",
    "situated_outcome_loss",
    "OutcomeAwareApprenticeshipCore",
    "OutcomeAwareApprenticeshipOutput",
    "OutcomeAwarePlasticState",
    "CausalNeuromodulatedApprenticeshipCore",
    "CausalNeuromodulatedOutput",
    "ContentAddressedCausalMemoryCore",
    "ContentAddressedCausalMemoryOutput",
    "ContentAddressedCausalMemoryState",
    "SparseMemoryCausalRoutingCore",
    "FactorizedKeyValueCausalMemoryCore",
    "DncAllocatedFactorizedCausalMemoryCore",
    "NaturalLanguageTraceGraphEncoder",
    "NaturalTraceGraphCausalMemoryCore",
    "NaturalTraceGraphCausalMemoryOutput",
    "NaturalTraceGraphMemoryState",
    "parse_step_trace",
    "OMLNaturalTraceRepresentation",
    "natural_trace_representation_metrics",
    "StructureProtectedOMLCore",
    "StructureKeyedCreditEvent",
    "StructureKeyedCreditMemoryCore",
    "StructureKeyedCreditOutput",
    "StructureKeyedCreditSnapshot",
    "StructureKeyedCreditState",
    "CompositeProspectiveCreditCore",
    "CompositeProspectiveCreditEvent",
    "ProspectiveComponentStateIntegrity",
    "ProspectiveDynamicsConfig",
    "ProspectiveDynamicsSnapshot",
    "ProspectiveDynamicsState",
    "ProspectiveFocusOutput",
    "ProspectiveResourceBudget",
    "PlasticProcedureState",
    "ProceduralCoreConfig",
    "ProceduralCoreOutput",
    "ScalableProceduralCore",
    "plastic_state_digest",
    "procedural_core_config",
    "select_procedural_core_tier",
    "CandidateProcedureDecode",
    "CandidateProcedureDecoder",
    "candidate_procedure_loss",
    "DonorCandidateFusion",
    "EdgeAwareActionTraceDecoder",
    "RoleAwareGraphProcedureDecoder",
    "PUBLIC_ROLE_WIDTH",
    "build_public_candidate_roles",
    "TraceConditionedProceduralCore",
    "ActionTraceMatcherDecoder",
    "ActionTraceProceduralCore",
    "PublicProcedureRelations",
    "PublicRelationComponent",
    "RelationalIncidenceTensors",
    "SemanticRelationalFusion",
    "StructuredActionTraceMatcherDecoder",
    "StructuredActionTraceProceduralCore",
    "StructuredRelationalEncoder",
    "build_relational_incidence",
    "parse_public_relations",
]
