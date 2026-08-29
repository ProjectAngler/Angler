"""Private synthetic software-pipeline reconstruction evaluator.

The public side contains only typed component contracts, grounded component
candidates, structural record states, successful observations from sibling
packages, an origin, a required output contract, and a step budget.  The
evaluator side owns executable component behavior and hidden integration
inputs.  A learner must commit one immutable pipeline before the evaluator
executes it; judging returns only terminal ``0.0`` or ``1.0``.

Every package receives a fresh namespace and fresh state/component identities.
Support packages demonstrate one repair motif at a time, while query packages
compose two motifs that never occurred together in one public observation.
There is deliberately no procedure enumeration, reference-pipeline field, or
intermediate evaluator feedback API in this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import hashlib
import itertools
import json
import random
import re
from typing import Literal

from angler.procedures.records import (
    ActionSchema,
    Goal,
    GroundAction,
    Record,
    State,
    Trace,
    Transition,
)


PipelinePartition = Literal["train", "development", "final"]
PipelineControlArm = Literal[
    "correct",
    "no_evidence",
    "wrong_evidence",
    "shuffled_outcome",
    "a_only",
    "b_only",
]

_NAMESPACE_PREFIX = "angler.microrepo"
_PARTITION_SIZES = {"train": 64, "development": 16, "final": 16}
_PARTITION_OFFSETS = {"train": 0, "development": 64, "final": 80}
_SEALED_OFFSET = 96
_SEALED_COUNT = 16
_MOTIFS = (
    "data_flow",
    "execution_order",
    "branch_routing",
    "stale_invalidation",
    "error_propagation",
)
_MOTIF_PAIRS = tuple(itertools.combinations(range(len(_MOTIFS)), 2))
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_QUALIFIED_NAME = re.compile(
    r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+$"
)
_PUBLIC_DIGEST_DOMAIN = b"project-angler.microrepo.public-task.v1\x00"
_CONTRACT_DIGEST_DOMAIN = b"project-angler.microrepo.component-contract.v1\x00"
_MECHANISM_DIGEST_DOMAIN = b"project-angler.microrepo.mechanism.v1\x00"
_PAIR_ADDRESS_DOMAIN = b"project-angler.microrepo.pair-address.v1\x00"
_OPAQUE_DOMAIN = b"project-angler.microrepo.opaque-surface.v1\x00"
_APPLIED_SUFFIX = "applied"
_TOPOLOGY_NODE_COUNT = 18
_TOPOLOGY_CHORD_COUNT = 4
_TOPOLOGY_EDGE_COUNT = _TOPOLOGY_NODE_COUNT + _TOPOLOGY_CHORD_COUNT
_TOPOLOGY_ATTEMPTS = 4_096
# Four chord sources occupy one of five relational families around an
# anonymous directed cycle.  The gap multisets all sum to eighteen and are
# intentionally distinct, but every resulting component still has the same
# node/edge and in/out-degree aggregates.  Fresh chord destinations and
# matchings vary the episode-level graph inside each family.
_MOTIF_SOURCE_GAPS = (
    (1, 1, 1, 15),
    (1, 1, 8, 8),
    (1, 3, 6, 8),
    (2, 2, 7, 7),
    (4, 4, 5, 5),
)


@dataclass(frozen=True, slots=True)
class PublicComponentContract:
    """One public component schema and its typed structural incidence."""

    schema: ActionSchema
    input_type: str
    output_type: str
    error_type: str
    state_reads: tuple[str, ...]
    state_writes: tuple[str, ...]
    incidence: tuple[Record, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.schema, ActionSchema):
            raise TypeError("component schema must be an ActionSchema")
        for label, value in (
            ("input_type", self.input_type),
            ("output_type", self.output_type),
            ("error_type", self.error_type),
        ):
            _require_qualified_name(value, label)
            if value.rsplit(".", maxsplit=1)[0] != self.schema.namespace:
                raise ValueError(f"component {label} must belong to its namespace")
        _require_atom_tuple(self.state_reads, "component state_reads")
        _require_atom_tuple(self.state_writes, "component state_writes")
        if not self.state_reads or not self.state_writes:
            raise ValueError("component state incidence cannot be empty")
        if type(self.incidence) is not tuple or any(
            not isinstance(record, Record) for record in self.incidence
        ):
            raise TypeError("component incidence must be an immutable Record tuple")
        if not self.incidence:
            raise ValueError("component incidence cannot be empty")
        if tuple(sorted(self.incidence)) != self.incidence:
            raise ValueError("component incidence must be canonically sorted")
        if len(set(self.incidence)) != len(self.incidence):
            raise ValueError("component incidence cannot contain duplicates")
        if any(record.namespace != self.schema.namespace for record in self.incidence):
            raise ValueError("component incidence must belong to its namespace")

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.to_canonical(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(
            _CONTRACT_DIGEST_DOMAIN + payload
        ).hexdigest()

    def to_canonical(self) -> dict[str, object]:
        return {
            "error_type": self.error_type,
            "incidence": [_record_payload(record) for record in self.incidence],
            "input_type": self.input_type,
            "output_type": self.output_type,
            "schema": _action_schema_payload(self.schema),
            "state_reads": list(self.state_reads),
            "state_writes": list(self.state_writes),
        }


@dataclass(frozen=True, slots=True)
class PublicSoftwarePipelineTask:
    """Complete learner projection with evaluator-private facts omitted."""

    components: tuple[PublicComponentContract, ...]
    grounded_candidates: tuple[GroundAction, ...]
    states: tuple[State, ...]
    observations: tuple[Trace, ...]
    origin: State
    required_output: Goal
    max_steps: int

    def __post_init__(self) -> None:
        _validate_public_task(self)

    def to_canonical(self) -> dict[str, object]:
        """Return a presentation-order-invariant public projection."""

        return {
            "components": [
                value.to_canonical()
                for value in sorted(self.components, key=lambda item: item.digest)
            ],
            "grounded_candidates": [
                _ground_action_payload(value)
                for value in sorted(
                    self.grounded_candidates,
                    key=lambda item: item.digest,
                )
            ],
            "max_steps": self.max_steps,
            "observations": [
                _trace_payload(value)
                for value in sorted(self.observations, key=lambda item: item.digest)
            ],
            "origin": _state_payload(self.origin),
            "required_output": _goal_payload(self.required_output),
            "states": [
                _state_payload(value)
                for value in sorted(self.states, key=lambda item: item.digest)
            ],
        }


@dataclass(frozen=True, slots=True)
class CommittedSoftwarePipeline:
    """Immutable, task-bound pipeline frozen before hidden execution."""

    public_digest: str
    actions: tuple[GroundAction, ...]
    stopped: bool

    def __post_init__(self) -> None:
        _require_digest(self.public_digest, "pipeline public_digest")
        if type(self.actions) is not tuple or any(
            not isinstance(action, GroundAction) for action in self.actions
        ):
            raise TypeError("pipeline actions must be an immutable GroundAction tuple")
        if type(self.stopped) is not bool:
            raise TypeError("pipeline stopped flag must be bool")


@dataclass(frozen=True, slots=True, repr=False)
class _PrivateIntegrationInput:
    payload: int
    branch: bool
    has_error: bool
    source_epoch: int
    cache_epoch: int


@dataclass(frozen=True, slots=True, repr=False)
class _PrivateExecutionFrame:
    integration_input: _PrivateIntegrationInput
    prepared: tuple[tuple[int, int], ...] = ()
    completed: frozenset[int] = frozenset()
    rejected: bool = False

    def prepared_value(self, motif: int) -> int | None:
        return dict(self.prepared).get(motif)


@dataclass(frozen=True, slots=True, repr=False)
class _PrivateComponentImplementation:
    action_digest: str
    motif: int
    stage: int
    variant: int
    distractor: bool = False

    def __post_init__(self) -> None:
        _require_digest(self.action_digest, "private component action digest")
        if self.motif not in range(len(_MOTIFS)):
            raise ValueError("private component motif is invalid")
        if self.stage not in (0, 1):
            raise ValueError("private component stage is invalid")
        if isinstance(self.variant, bool) or not isinstance(self.variant, int):
            raise TypeError("private component variant must be an integer")
        if type(self.distractor) is not bool:
            raise TypeError("private component distractor flag must be bool")

    def execute(self, frame: _PrivateExecutionFrame) -> _PrivateExecutionFrame:
        """Execute one private component without exposing intermediate state."""

        if frame.rejected:
            return frame
        if self.distractor:
            return replace(frame, rejected=True)
        signature = _motif_signature(
            self.motif,
            self.variant,
            frame.integration_input,
        )
        if self.stage == 0:
            prepared = dict(frame.prepared)
            prepared[self.motif] = signature
            return replace(frame, prepared=tuple(sorted(prepared.items())))
        if frame.prepared_value(self.motif) != signature:
            return replace(frame, rejected=True)
        return replace(
            frame,
            completed=frame.completed | frozenset((self.motif,)),
        )


@dataclass(frozen=True, slots=True, repr=False)
class _HiddenSoftwarePipelineSolution:
    public_digest: str
    implementations: tuple[_PrivateComponentImplementation, ...]
    integration_inputs: tuple[_PrivateIntegrationInput, ...]
    required_motifs: tuple[int, ...]
    mechanism_commitment: str
    mechanism_partition: PipelinePartition
    package_commitment: str

    def __post_init__(self) -> None:
        _require_digest(self.public_digest, "hidden public_digest")
        _require_digest(self.mechanism_commitment, "hidden mechanism commitment")
        _require_digest(self.package_commitment, "hidden package commitment")
        _validate_partition(self.mechanism_partition)
        if type(self.implementations) is not tuple or not self.implementations:
            raise ValueError("hidden component implementations cannot be empty")
        if any(
            not isinstance(value, _PrivateComponentImplementation)
            for value in self.implementations
        ):
            raise TypeError("hidden implementations contain an invalid value")
        action_digests = tuple(value.action_digest for value in self.implementations)
        if len(set(action_digests)) != len(action_digests):
            raise ValueError("hidden component action digests must be unique")
        if type(self.integration_inputs) is not tuple or not self.integration_inputs:
            raise ValueError("hidden integration batch cannot be empty")
        if any(
            not isinstance(value, _PrivateIntegrationInput)
            for value in self.integration_inputs
        ):
            raise TypeError("hidden integration batch contains an invalid value")
        if (
            type(self.required_motifs) is not tuple
            or not self.required_motifs
            or tuple(sorted(set(self.required_motifs))) != self.required_motifs
        ):
            raise ValueError("hidden required motifs must be unique and sorted")


@dataclass(frozen=True, slots=True, repr=False)
class GeneratedSoftwarePipelineTask:
    """Evaluator pairing; learner code receives only ``learner``."""

    learner: PublicSoftwarePipelineTask
    hidden: _HiddenSoftwarePipelineSolution

    def __post_init__(self) -> None:
        _validate_pairing(self.learner, self.hidden)


@dataclass(frozen=True, slots=True, repr=False)
class SoftwarePipelineStream:
    """Separate single-motif supports followed by composed query packages."""

    supports: tuple[GeneratedSoftwarePipelineTask, ...]
    queries: tuple[GeneratedSoftwarePipelineTask, ...]
    mechanism_commitment: str
    mechanism_partition: PipelinePartition
    control_arm: PipelineControlArm = "correct"

    def __post_init__(self) -> None:
        _validate_stream(self)


@dataclass(frozen=True, slots=True, repr=False)
class _PipelineMechanism:
    semantic_index: int
    motifs: tuple[int, int]
    variants: tuple[int, int]
    presentation_variant: int

    def __post_init__(self) -> None:
        if self.semantic_index < 0:
            raise ValueError("semantic index must be nonnegative")
        if (
            type(self.motifs) is not tuple
            or len(self.motifs) != 2
            or tuple(sorted(self.motifs)) != self.motifs
            or self.motifs[0] == self.motifs[1]
        ):
            raise ValueError("mechanism must compose two distinct sorted motifs")
        if type(self.variants) is not tuple or len(self.variants) != 2:
            raise ValueError("mechanism must bind two motif variants")


@dataclass(frozen=True, slots=True)
class _PackageSurface:
    task: PublicSoftwarePipelineTask
    implementations: tuple[_PrivateComponentImplementation, ...]
    package_commitment: str


def software_pipeline_mechanism_partition(
    partition: PipelinePartition,
) -> tuple[str, ...]:
    """Return opaque commitments for exactly one semantic partition."""

    _validate_partition(partition)
    return tuple(
        _mechanism_commitment(mechanism)
        for mechanism in _semantic_partition(partition)
    )


def make_software_pipeline_stream(
    seed: int,
    *,
    surface_seed: int | None = None,
    supports_per_motif: int = 2,
    queries: int = 2,
    maximum_steps: int = 4,
    mechanism_commitment: str | None = None,
    mechanism_partition: PipelinePartition = "train",
) -> SoftwarePipelineStream:
    """Build one deterministic single-motif-to-composition curriculum."""

    _validate_seed(seed, "seed")
    if surface_seed is None:
        surface_seed = _domain_seed(seed, "default-surface", 0, 0)
    _validate_seed(surface_seed, "surface_seed")
    _validate_positive_count(supports_per_motif, "supports_per_motif")
    _validate_positive_count(queries, "queries")
    if (
        isinstance(maximum_steps, bool)
        or not isinstance(maximum_steps, int)
        or not 2 <= maximum_steps <= 4
    ):
        raise ValueError("maximum_steps must be an integer from two through four")
    if maximum_steps < 4:
        raise ValueError("composed queries require a four-step public budget")
    _validate_partition(mechanism_partition)

    mechanisms = _semantic_partition(mechanism_partition)
    by_commitment = {
        _mechanism_commitment(value): value for value in mechanisms
    }
    if mechanism_commitment is None:
        mechanism = mechanisms[
            _domain_seed(seed, "mechanism-selection", 0, 0) % len(mechanisms)
        ]
        commitment = _mechanism_commitment(mechanism)
    else:
        _require_digest(mechanism_commitment, "mechanism_commitment")
        try:
            mechanism = by_commitment[mechanism_commitment]
        except KeyError as error:
            raise ValueError(
                "mechanism_commitment is outside the declared partition"
            ) from error
        commitment = mechanism_commitment

    support_pairs: list[GeneratedSoftwarePipelineTask] = []
    # Signatures are reserved before presentation-order shuffling.  This turns
    # an accidental support/query episode-key collision into generation failure
    # rather than silently permitting a retrieval shortcut.
    topology_signatures: set[tuple[object, ...]] = set()
    support_index = 0
    for motif_position, motif in enumerate(mechanism.motifs):
        for local_index in range(supports_per_motif):
            pair = _make_package_pair(
                seed,
                surface_seed=surface_seed,
                scope="support",
                package_index=support_index,
                motifs=(motif,),
                variants=(mechanism.variants[motif_position],),
                mechanism=mechanism,
                mechanism_commitment=commitment,
                partition=mechanism_partition,
                expose_observation=True,
                maximum_steps=min(maximum_steps, 3),
                presentation_variant=(
                    mechanism.presentation_variant + local_index + motif_position
                ),
                topology_signatures=topology_signatures,
            )
            support_pairs.append(pair)
            support_index += 1

    query_pairs = [
        _make_package_pair(
            seed,
            surface_seed=surface_seed,
            scope="query",
            package_index=index,
            motifs=mechanism.motifs,
            variants=mechanism.variants,
            mechanism=mechanism,
            mechanism_commitment=commitment,
            partition=mechanism_partition,
            expose_observation=False,
            maximum_steps=maximum_steps,
            presentation_variant=mechanism.presentation_variant + index,
            topology_signatures=topology_signatures,
        )
        for index in range(queries)
    ]
    random.Random(_domain_seed(seed, "support-order", 0, 0)).shuffle(
        support_pairs
    )
    random.Random(_domain_seed(seed, "query-order", 0, 0)).shuffle(query_pairs)
    return SoftwarePipelineStream(
        supports=tuple(support_pairs),
        queries=tuple(query_pairs),
        mechanism_commitment=commitment,
        mechanism_partition=mechanism_partition,
        control_arm="correct",
    )


def make_software_pipeline_control_stream(
    stream: SoftwarePipelineStream,
    arm: PipelineControlArm,
) -> SoftwarePipelineStream:
    """Construct a matched public-evidence control without hidden execution."""

    if not isinstance(stream, SoftwarePipelineStream):
        raise TypeError("stream must be a SoftwarePipelineStream")
    if arm not in (
        "correct",
        "no_evidence",
        "wrong_evidence",
        "shuffled_outcome",
        "a_only",
        "b_only",
    ):
        raise ValueError("control arm is invalid")
    if arm == "correct":
        return stream
    if arm in ("a_only", "b_only"):
        query_motifs = stream.queries[0].hidden.required_motifs
        wanted = query_motifs[0 if arm == "a_only" else 1]
        supports = tuple(
            pair
            for pair in stream.supports
            if pair.hidden.required_motifs == (wanted,)
        )
        if not supports:
            raise RuntimeError("motif-pure support control is empty")
    else:
        supports = tuple(_reproject_control_pair(pair, arm) for pair in stream.supports)
    return SoftwarePipelineStream(
        supports=supports,
        queries=stream.queries,
        mechanism_commitment=stream.mechanism_commitment,
        mechanism_partition=stream.mechanism_partition,
        control_arm=arm,
    )


def commit_software_pipeline(
    task: PublicSoftwarePipelineTask,
    actions: Sequence[GroundAction],
    *,
    stopped: bool,
) -> CommittedSoftwarePipeline:
    """Validate and freeze one declared pipeline without executing it."""

    if not isinstance(task, PublicSoftwarePipelineTask):
        raise TypeError("task must be a PublicSoftwarePipelineTask")
    if not isinstance(actions, Sequence) or isinstance(
        actions,
        (str, bytes, bytearray),
    ):
        raise TypeError("actions must be a finite sequence")
    if type(stopped) is not bool:
        raise TypeError("stopped must be bool")
    pipeline = CommittedSoftwarePipeline(
        public_digest=_public_digest(task),
        actions=tuple(actions),
        stopped=stopped,
    )
    _validate_pipeline_for_task(task, pipeline)
    return pipeline


def judge_software_pipeline_attempt(
    pair: GeneratedSoftwarePipelineTask,
    pipeline: CommittedSoftwarePipeline,
) -> float:
    """Execute one immutable pipeline once and reveal only terminal exactness."""

    if not isinstance(pair, GeneratedSoftwarePipelineTask):
        raise TypeError("pair must be a GeneratedSoftwarePipelineTask")
    if not isinstance(pipeline, CommittedSoftwarePipeline):
        raise TypeError("pipeline must be a CommittedSoftwarePipeline")
    _validate_pairing(pair.learner, pair.hidden)
    _validate_pipeline_for_task(pair.learner, pipeline)
    if pipeline.public_digest != pair.hidden.public_digest:
        raise ValueError("pipeline and evaluator solution bind different tasks")
    return _execute_committed_pipeline(pair.hidden, pipeline)


def _execute_committed_pipeline(
    solution: _HiddenSoftwarePipelineSolution,
    pipeline: CommittedSoftwarePipeline,
) -> float:
    """Run the hidden integration batch as one indivisible evaluator call."""

    implementations = {
        value.action_digest: value for value in solution.implementations
    }
    required = frozenset(solution.required_motifs)
    for integration_input in solution.integration_inputs:
        frame = _PrivateExecutionFrame(integration_input=integration_input)
        for action in pipeline.actions:
            frame = implementations[action.digest].execute(frame)
        if frame.rejected or frame.completed != required:
            return 0.0
    return 1.0


def _make_package_pair(
    seed: int,
    *,
    surface_seed: int,
    scope: str,
    package_index: int,
    motifs: tuple[int, ...],
    variants: tuple[int, ...],
    mechanism: _PipelineMechanism,
    mechanism_commitment: str,
    partition: PipelinePartition,
    expose_observation: bool,
    maximum_steps: int,
    presentation_variant: int,
    topology_signatures: set[tuple[object, ...]],
) -> GeneratedSoftwarePipelineTask:
    package_nonce = _opaque_token(
        surface_seed,
        mechanism_commitment,
        scope,
        package_index,
        "package",
        0,
    )
    namespace = f"{_NAMESPACE_PREFIX}_{package_nonce}"
    package_commitment = _digest_payload(
        b"project-angler.microrepo.package.v1\x00",
        {
            "mechanism": mechanism.semantic_index,
            "motifs": motifs,
            "package_index": package_index,
            "scope": scope,
            "seed": seed,
        },
    )
    surface = _build_package_surface(
        surface_seed,
        topology_seed=seed,
        namespace=namespace,
        package_nonce=package_nonce,
        mechanism_commitment=mechanism_commitment,
        scope=scope,
        package_index=package_index,
        motifs=motifs,
        variants=variants,
        expose_observation=expose_observation,
        maximum_steps=maximum_steps,
        presentation_variant=presentation_variant,
        topology_signatures=topology_signatures,
    )
    integration_inputs = tuple(
        _private_integration_input(
            seed,
            mechanism.semantic_index,
            scope,
            package_index,
            index,
        )
        for index in range(5)
    )
    hidden = _HiddenSoftwarePipelineSolution(
        public_digest=_public_digest(surface.task),
        implementations=surface.implementations,
        integration_inputs=integration_inputs,
        required_motifs=tuple(sorted(motifs)),
        mechanism_commitment=mechanism_commitment,
        mechanism_partition=partition,
        package_commitment=package_commitment,
    )
    return GeneratedSoftwarePipelineTask(surface.task, hidden)


def _build_package_surface(
    surface_seed: int,
    *,
    topology_seed: int,
    namespace: str,
    package_nonce: str,
    mechanism_commitment: str,
    scope: str,
    package_index: int,
    motifs: tuple[int, ...],
    variants: tuple[int, ...],
    expose_observation: bool,
    maximum_steps: int,
    presentation_variant: int,
    topology_signatures: set[tuple[object, ...]],
) -> _PackageSurface:
    token_counter = itertools.count()

    def token(role: str) -> str:
        return _opaque_token(
            surface_seed,
            mechanism_commitment,
            scope,
            package_index,
            role,
            next(token_counter),
        )

    components: list[PublicComponentContract] = []
    implementations: list[_PrivateComponentImplementation] = []
    correct_by_motif: list[tuple[GroundAction, GroundAction]] = []
    origin_tokens: list[str] = []

    for motif, variant in zip(motifs, variants, strict=True):
        input_type = f"{namespace}.type_{token('input-type')}"
        middle_type = f"{namespace}.type_{token('middle-type')}"
        output_type = f"{namespace}.type_{token('output-type')}"
        error_type = f"{namespace}.error_{token('error-type')}"
        origin_token = token("origin-state")
        middle_token = token("middle-state")
        final_token = token("final-state")
        origin_tokens.append(origin_token)

        topology_nodes = tuple(
            token("topology-node") for _ in range(_TOPOLOGY_NODE_COUNT)
        )
        base_edges, transform_zero, transform_one = _fresh_topology_transforms(
            topology_seed,
            scope=scope,
            package_index=package_index,
            motif=motif,
            reserved=topology_signatures,
        )

        first = _make_component_contract(
            namespace,
            component_token=token("component"),
            input_type=input_type,
            output_type=middle_type,
            error_type=error_type,
            state_reads=(origin_token,),
            state_writes=(middle_token,),
            topology_nodes=topology_nodes,
            topology_edges=base_edges,
        )
        alternative_zero = _make_component_contract(
            namespace,
            component_token=token("component"),
            input_type=middle_type,
            output_type=output_type,
            error_type=error_type,
            state_reads=(middle_token,),
            state_writes=(final_token,),
            topology_nodes=topology_nodes,
            topology_edges=transform_zero,
        )
        alternative_one = _make_component_contract(
            namespace,
            component_token=token("component"),
            input_type=middle_type,
            output_type=output_type,
            error_type=error_type,
            state_reads=(middle_token,),
            state_writes=(final_token,),
            topology_nodes=topology_nodes,
            topology_edges=transform_one,
        )
        # Both alternatives have the same goal-relevant type and state
        # contract.  A mechanism-private polarity decides which anonymous
        # topology is executable.  Sibling-package observations reveal that
        # polarity; a query package by itself is counterfactually ambiguous.
        selected = alternative_zero if variant % 2 == 0 else alternative_one
        for contract, stage, is_distractor in (
            (first, 0, False),
            (
                alternative_zero,
                1,
                alternative_zero is not selected,
            ),
            (
                alternative_one,
                1,
                alternative_one is not selected,
            ),
        ):
            components.append(contract)
            implementations.append(
                _PrivateComponentImplementation(
                    action_digest=contract.schema.ground().digest,
                    motif=motif,
                    stage=stage,
                    variant=variant,
                    distractor=is_distractor,
                )
            )
        correct_by_motif.append((first.schema.ground(), selected.schema.ground()))

    origin = _state_from_tokens(namespace, origin_tokens)
    correct_actions = tuple(
        action
        for pair in correct_by_motif
        for action in pair
    )
    correct_trace = _make_public_trace(
        origin,
        tuple(components),
        correct_actions,
    )
    final_state = correct_trace.final_state
    public_states = tuple(
        dict.fromkeys(
            (
                origin,
                *(value.after for value in correct_trace.transitions),
            )
        )
    )
    if not expose_observation and len(motifs) == 2:
        public_states = _variant_blind_progress_state_closure(
            origin,
            tuple(components),
        )
    observations: tuple[Trace, ...] = ()
    if expose_observation:
        observations = (correct_trace,)

    component_order = list(components)
    candidate_order = [value.schema.ground() for value in components]
    state_order = list(dict.fromkeys(public_states))
    random.Random(
        _domain_seed(surface_seed, "component-order", package_index, presentation_variant)
    ).shuffle(component_order)
    random.Random(
        _domain_seed(surface_seed, "candidate-order", package_index, presentation_variant)
    ).shuffle(candidate_order)
    random.Random(
        _domain_seed(surface_seed, "state-order", package_index, presentation_variant)
    ).shuffle(state_order)
    task = PublicSoftwarePipelineTask(
        components=tuple(component_order),
        grounded_candidates=tuple(candidate_order),
        states=tuple(state_order),
        observations=observations,
        origin=origin,
        required_output=Goal.from_records(
            namespace,
            final_state.records,
            exact=True,
        ),
        max_steps=maximum_steps,
    )
    return _PackageSurface(
        task=task,
        implementations=tuple(implementations),
        package_commitment=_digest_payload(
            b"project-angler.microrepo.package-surface.v1\x00",
            {"namespace": namespace, "package_nonce": package_nonce},
        ),
    )


def _make_component_contract(
    namespace: str,
    *,
    component_token: str,
    input_type: str,
    output_type: str,
    error_type: str,
    state_reads: tuple[str, ...],
    state_writes: tuple[str, ...],
    topology_nodes: tuple[str, ...],
    topology_edges: tuple[tuple[int, int], ...],
) -> PublicComponentContract:
    if len(topology_nodes) != _TOPOLOGY_NODE_COUNT:
        raise ValueError("component topology has the wrong node count")
    if (
        len(topology_edges) != _TOPOLOGY_EDGE_COUNT
        or len(set(topology_edges)) != len(topology_edges)
        or any(
            source not in range(_TOPOLOGY_NODE_COUNT)
            or target not in range(_TOPOLOGY_NODE_COUNT)
            or source == target
            for source, target in topology_edges
        )
    ):
        raise ValueError("component topology edges are invalid")
    schema = ActionSchema(
        f"{namespace}.component_{component_token}",
        (),
        description="typed component with public structural incidence",
    )
    records = [
        Record(f"{namespace}.accepts", (schema.name, input_type)),
        Record(f"{namespace}.returns", (schema.name, output_type)),
        Record(f"{namespace}.errors", (schema.name, error_type)),
    ]
    records.extend(
        Record(f"{namespace}.reads", (schema.name, value))
        for value in state_reads
    )
    records.extend(
        Record(f"{namespace}.writes", (schema.name, value))
        for value in state_writes
    )
    records.extend(
        Record(
            f"{namespace}.relates",
            (schema.name, topology_nodes[source], topology_nodes[target]),
        )
        for source, target in topology_edges
    )
    return PublicComponentContract(
        schema=schema,
        input_type=input_type,
        output_type=output_type,
        error_type=error_type,
        state_reads=state_reads,
        state_writes=state_writes,
        incidence=tuple(sorted(records)),
    )


def _fresh_topology_transforms(
    seed: int,
    *,
    scope: str,
    package_index: int,
    motif: int,
    reserved: set[tuple[object, ...]],
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
]:
    """Sample one fresh ``G`` and two degree-preserving graph transforms.

    The retry is part of deterministic generation, not learner-visible
    inference.  It enforces that no component graph in a later package can be
    retrieved by an exact directed-WL episode key from an earlier package.
    ``T0`` and ``T1`` cyclically reassign one uniformly sampled destination
    assignment forward or backward across the cycle-ordered chord sources.
    Their candidate-only marginals are therefore exactly exchangeable, while
    the joint relation between ``G`` and either completion remains directional.
    """

    if motif not in range(len(_MOTIFS)):
        raise ValueError("topology motif is invalid")
    for attempt in range(_TOPOLOGY_ATTEMPTS):
        rng = random.Random(
            _domain_seed(
                seed,
                f"topology-{scope}-{motif}",
                package_index,
                attempt,
            )
        )
        gaps = list(_MOTIF_SOURCE_GAPS[motif])
        rng.shuffle(gaps)
        cursor = rng.randrange(_TOPOLOGY_NODE_COUNT)
        sources = []
        for gap in gaps:
            sources.append(cursor)
            cursor = (cursor + gap) % _TOPOLOGY_NODE_COUNT
        if len(set(sources)) != _TOPOLOGY_CHORD_COUNT:
            continue
        source_set = set(sources)
        destinations = [
            node
            for node in range(_TOPOLOGY_NODE_COUNT)
            if all(
                (node + offset) % _TOPOLOGY_NODE_COUNT not in source_set
                for offset in (-1, 0, 1)
            )
        ]
        if len(destinations) < _TOPOLOGY_CHORD_COUNT:
            continue
        rng.shuffle(destinations)
        destinations = destinations[:_TOPOLOGY_CHORD_COUNT]
        cycle = tuple(
            (node, (node + 1) % _TOPOLOGY_NODE_COUNT)
            for node in range(_TOPOLOGY_NODE_COUNT)
        )
        base_chords, zero_chords, one_chords = _cyclic_destination_transforms(
            tuple(sources), tuple(destinations)
        )
        base = tuple(sorted((*cycle, *base_chords)))
        transform_zero = tuple(sorted((*cycle, *zero_chords)))
        transform_one = tuple(sorted((*cycle, *one_chords)))
        _validate_topology_transform(base, transform_zero)
        _validate_topology_transform(base, transform_one)
        signatures = {
            _anonymous_topology_signature(base),
            _anonymous_topology_signature(transform_zero),
            _anonymous_topology_signature(transform_one),
        }
        # Distinct isomorphism-invariant signatures prove that neither twin
        # collapses to G (or to the other twin) under anonymous renaming.
        if len(signatures) != 3 or signatures & reserved:
            continue
        reserved.update(signatures)
        return base, transform_zero, transform_one
    raise RuntimeError("could not generate a fresh asymmetric topology")


def _cyclic_destination_transforms(
    sources: tuple[int, ...],
    destinations: tuple[int, ...],
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
]:
    if (
        len(sources) != _TOPOLOGY_CHORD_COUNT
        or len(destinations) != _TOPOLOGY_CHORD_COUNT
        or len(set(sources)) != len(sources)
        or len(set(destinations)) != len(destinations)
    ):
        raise ValueError("cyclic topology assignments must be unique and complete")
    base = tuple(zip(sources, destinations, strict=True))
    forward = tuple(
        (
            source,
            destinations[(index + 1) % _TOPOLOGY_CHORD_COUNT],
        )
        for index, source in enumerate(sources)
    )
    backward = tuple(
        (
            source,
            destinations[(index - 1) % _TOPOLOGY_CHORD_COUNT],
        )
        for index, source in enumerate(sources)
    )
    return base, forward, backward


def _validate_topology_transform(
    base: tuple[tuple[int, int], ...],
    transformed: tuple[tuple[int, int], ...],
) -> None:
    if (
        len(base) != _TOPOLOGY_EDGE_COUNT
        or len(transformed) != _TOPOLOGY_EDGE_COUNT
        or len(set(base)) != len(base)
        or len(set(transformed)) != len(transformed)
    ):
        raise RuntimeError("topology transform changed the edge aggregate")

    def degree_profile(edges: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
        return tuple(
            sorted(
                (
                    sum(target == node for _, target in edges),
                    sum(source == node for source, _ in edges),
                )
                for node in range(_TOPOLOGY_NODE_COUNT)
            )
        )

    if degree_profile(base) != degree_profile(transformed):
        raise RuntimeError("topology transform changed the degree aggregate")


def _anonymous_topology_signature(
    edges: tuple[tuple[int, int], ...],
) -> tuple[object, ...]:
    """Directed WL signature used only to reject generator collisions."""

    nodes = tuple(range(_TOPOLOGY_NODE_COUNT))
    labels: dict[int, object] = {
        node: (
            sum(target == node for _, target in edges),
            sum(source == node for source, _ in edges),
        )
        for node in nodes
    }
    for _ in nodes:
        rows = {
            node: (
                labels[node],
                tuple(sorted(labels[source] for source, target in edges if target == node)),
                tuple(sorted(labels[target] for source, target in edges if source == node)),
            )
            for node in nodes
        }
        vocabulary = {
            value: index
            for index, value in enumerate(sorted(set(rows.values()), key=repr))
        }
        labels = {node: vocabulary[value] for node, value in rows.items()}
    return (
        tuple(sorted(labels.values())),
        tuple(sorted((labels[source], labels[target]) for source, target in edges)),
    )


def _contract_topology_signature(
    contract: PublicComponentContract,
) -> tuple[object, ...]:
    edges = tuple(
        (record.arguments[1], record.arguments[2])
        for record in contract.incidence
        if record.predicate.endswith(".relates")
    )
    nodes = sorted({value for edge in edges for value in edge})
    if len(nodes) != _TOPOLOGY_NODE_COUNT:
        raise ValueError("component topology has the wrong anonymous node count")
    indices = {value: index for index, value in enumerate(nodes)}
    return _anonymous_topology_signature(
        tuple((indices[source], indices[target]) for source, target in edges)
    )


def _make_public_trace(
    origin: State,
    components: tuple[PublicComponentContract, ...],
    actions: tuple[GroundAction, ...],
) -> Trace:
    contracts = {value.schema.digest: value for value in components}
    transitions: list[Transition] = []
    current = origin
    for action in actions:
        contract = contracts[action.schema.digest]
        after = _state_from_tokens(
            current.namespace,
            (*_held_tokens(current), *contract.state_writes),
        )
        transitions.append(
            Transition(
                before=current,
                action=action,
                after=after,
                applied=True,
                outcome=f"{current.namespace}.{_APPLIED_SUFFIX}",
            )
        )
        current = after
    return Trace(initial=origin, transitions=tuple(transitions))


def _variant_blind_progress_state_closure(
    origin: State,
    components: tuple[PublicComponentContract, ...],
) -> tuple[State, ...]:
    """Expose every public progress combination for two independent chains.

    The closure depends only on public type links and read/write effects.  In
    particular, it never selects between the two completion twins: their
    public effects must be identical, so either yields the same progress
    state.  This prevents one valid motif interleaving from being privileged
    merely because it was used to construct the hidden evaluation package.
    """

    produced_types = {component.output_type for component in components}
    roots = tuple(
        sorted(
            (
                component
                for component in components
                if component.input_type not in produced_types
            ),
            key=lambda component: component.schema.digest,
        )
    )
    if len(roots) != 2:
        raise RuntimeError("composed query must expose exactly two public roots")
    chains = []
    for root in roots:
        completions = tuple(
            component
            for component in components
            if component.input_type == root.output_type
        )
        if len(completions) != 2:
            raise RuntimeError("public root must have exactly two completion twins")
        effects = {
            (component.state_reads, component.state_writes)
            for component in completions
        }
        if len(effects) != 1:
            raise RuntimeError("completion twins must share one public effect")
        chains.append((root, completions[0]))

    origin_tokens = _held_tokens(origin)
    states = []
    for progress in itertools.product(range(3), repeat=2):
        tokens = list(origin_tokens)
        for level, (root, completion) in zip(progress, chains, strict=True):
            if level >= 1:
                tokens.extend(root.state_writes)
            if level >= 2:
                tokens.extend(completion.state_writes)
        states.append(_state_from_tokens(origin.namespace, tokens))
    unique = tuple(dict.fromkeys(states))
    if len(unique) != 9:
        raise RuntimeError("composed query progress closure must contain nine states")
    return unique


def _state_from_tokens(namespace: str, tokens: Sequence[str]) -> State:
    return State.from_records(
        namespace,
        (
            Record(f"{namespace}.holds", (value,))
            for value in sorted(set(tokens))
        ),
    )


def _held_tokens(state: State) -> tuple[str, ...]:
    return tuple(
        record.arguments[0]
        for record in state.records
        if record.predicate.endswith(".holds")
    )


def _private_integration_input(
    seed: int,
    semantic_index: int,
    scope: str,
    package_index: int,
    case_index: int,
) -> _PrivateIntegrationInput:
    rng = random.Random(
        _domain_seed(
            seed ^ semantic_index,
            f"integration-{scope}",
            package_index,
            case_index,
        )
    )
    source_epoch = 2 + rng.randrange(97)
    return _PrivateIntegrationInput(
        payload=1 + rng.randrange(10_000),
        branch=bool(rng.randrange(2)),
        has_error=bool(case_index % 2),
        source_epoch=source_epoch,
        cache_epoch=max(0, source_epoch - (case_index % 3)),
    )


def _motif_signature(
    motif: int,
    variant: int,
    value: _PrivateIntegrationInput,
) -> int:
    if motif == 0:
        return (value.payload * (variant + 3) + 17) % 65_521
    if motif == 1:
        return (value.payload + value.source_epoch * (variant + 5)) % 65_521
    if motif == 2:
        return (value.payload ^ ((variant + 11) if value.branch else 0)) % 65_521
    if motif == 3:
        return (
            value.payload
            + value.source_epoch * 31
            - value.cache_epoch * (variant + 1)
        ) % 65_521
    if motif == 4:
        return (
            value.payload + (10_007 if value.has_error else 0) + variant
        ) % 65_521
    raise AssertionError("validated motif was not handled")


def _reproject_control_pair(
    pair: GeneratedSoftwarePipelineTask,
    arm: PipelineControlArm,
) -> GeneratedSoftwarePipelineTask:
    task = pair.learner
    if arm == "no_evidence":
        observations: tuple[Trace, ...] = ()
    elif arm == "wrong_evidence":
        observations = tuple(
            _replace_observation_actions(task, trace)
            for trace in task.observations
        )
    elif arm == "shuffled_outcome":
        observations = tuple(
            _shuffle_observation_outcomes(task, trace)
            for trace in task.observations
        )
    else:
        raise AssertionError("validated control arm was not handled")
    learner = replace(task, observations=observations)
    hidden = replace(pair.hidden, public_digest=_public_digest(learner))
    return GeneratedSoftwarePipelineTask(learner, hidden)


def _replace_observation_actions(
    task: PublicSoftwarePipelineTask,
    trace: Trace,
) -> Trace:
    contracts = {value.schema.digest: value for value in task.components}
    actions = {value.schema.digest: value for value in task.grounded_candidates}
    transitions: list[Transition] = []
    changed = False
    for transition in trace.transitions:
        contract = contracts[transition.action.schema.digest]
        signature = (
            contract.input_type,
            contract.output_type,
            contract.error_type,
            contract.state_reads,
            contract.state_writes,
        )
        alternatives = tuple(
            value
            for value in task.components
            if value.schema.digest != contract.schema.digest
            and (
                value.input_type,
                value.output_type,
                value.error_type,
                value.state_reads,
                value.state_writes,
            )
            == signature
        )
        if alternatives:
            replacement = min(alternatives, key=lambda value: value.digest)
            transitions.append(
                replace(
                    transition,
                    action=actions[replacement.schema.digest],
                )
            )
            changed = True
        else:
            transitions.append(transition)
    if not changed:
        raise RuntimeError("wrong-evidence control found no counterfactual twin")
    return Trace(initial=trace.initial, transitions=tuple(transitions))


def _counterfactual_solution(
    solution: _HiddenSoftwarePipelineSolution,
) -> _HiddenSoftwarePipelineSolution:
    """Flip private twin meaning while preserving the complete public task."""

    stage_one_by_motif: dict[int, list[_PrivateComponentImplementation]] = {}
    for implementation in solution.implementations:
        if implementation.stage == 1:
            stage_one_by_motif.setdefault(implementation.motif, []).append(
                implementation
            )
    if any(len(values) != 2 for values in stage_one_by_motif.values()):
        raise RuntimeError("counterfactual twin structure is incomplete")
    flipped = tuple(
        replace(value, distractor=not value.distractor)
        if value.stage == 1
        else value
        for value in solution.implementations
    )
    return replace(solution, implementations=flipped)


def _shuffle_observation_outcomes(
    task: PublicSoftwarePipelineTask,
    trace: Trace,
) -> Trace:
    afters = tuple(value.after for value in trace.transitions)
    rotated = afters[1:] + afters[:1]
    transitions: list[Transition] = []
    current = trace.initial
    for original, after in zip(trace.transitions, rotated, strict=True):
        if after == current:
            alternatives = tuple(value for value in task.states if value != current)
            after = alternatives[0]
        transitions.append(
            Transition(
                before=current,
                action=original.action,
                after=after,
                applied=True,
                outcome=f"{current.namespace}.{_APPLIED_SUFFIX}",
            )
        )
        current = after
    return Trace(initial=trace.initial, transitions=tuple(transitions))


def _validate_public_task(task: PublicSoftwarePipelineTask) -> None:
    if type(task.components) is not tuple or not task.components:
        raise ValueError("task components must be a non-empty immutable tuple")
    if any(not isinstance(value, PublicComponentContract) for value in task.components):
        raise TypeError("task components contain an invalid value")
    component_digests = tuple(value.digest for value in task.components)
    schema_digests = tuple(value.schema.digest for value in task.components)
    if len(set(component_digests)) != len(component_digests):
        raise ValueError("task component contracts must be unique")
    if len(set(schema_digests)) != len(schema_digests):
        raise ValueError("task component schemas must be unique")
    if type(task.grounded_candidates) is not tuple or not task.grounded_candidates:
        raise ValueError("grounded candidates must be a non-empty immutable tuple")
    if any(not isinstance(value, GroundAction) for value in task.grounded_candidates):
        raise TypeError("grounded candidates contain an invalid value")
    candidate_digests = tuple(value.digest for value in task.grounded_candidates)
    if len(set(candidate_digests)) != len(candidate_digests):
        raise ValueError("grounded candidates must be unique")
    if {value.schema.digest for value in task.grounded_candidates} != set(schema_digests):
        raise ValueError("grounded candidates must cover component schemas exactly")
    if any(value.arguments for value in task.grounded_candidates):
        raise ValueError("initial MicroRepo candidates must be zero-argument")
    if type(task.states) is not tuple or not task.states:
        raise ValueError("task states must be a non-empty immutable tuple")
    if any(not isinstance(value, State) for value in task.states):
        raise TypeError("task states contain an invalid value")
    if len({value.digest for value in task.states}) != len(task.states):
        raise ValueError("task states must be unique")
    if not isinstance(task.origin, State) or task.origin not in task.states:
        raise ValueError("task origin must be one declared state")
    if not isinstance(task.required_output, Goal):
        raise TypeError("task required_output must be a Goal")
    if not task.required_output.exact:
        raise ValueError("task required_output must be exact")
    namespace = task.origin.namespace
    if any(value.namespace != namespace for value in task.states):
        raise ValueError("task states must share one namespace")
    if task.required_output.namespace != namespace:
        raise ValueError("required output must share the task namespace")
    if any(value.schema.namespace != namespace for value in task.components):
        raise ValueError("task components must share the task namespace")
    if not any(
        value.records == task.required_output.required for value in task.states
    ):
        raise ValueError("required output must match one declared public state")
    if (
        isinstance(task.max_steps, bool)
        or not isinstance(task.max_steps, int)
        or not 1 <= task.max_steps <= 4
    ):
        raise ValueError("task max_steps must be an integer from one through four")
    if type(task.observations) is not tuple or any(
        not isinstance(value, Trace) for value in task.observations
    ):
        raise TypeError("task observations must be an immutable Trace tuple")
    declared_states = set(task.states)
    declared_actions = set(task.grounded_candidates)
    for trace in task.observations:
        if trace.initial not in declared_states:
            raise ValueError("observation initial state is undeclared")
        for transition in trace.transitions:
            if transition.before not in declared_states or transition.after not in declared_states:
                raise ValueError("observation contains an undeclared state")
            if transition.action not in declared_actions:
                raise ValueError("observation contains an undeclared component")


def _validate_pairing(
    task: PublicSoftwarePipelineTask,
    solution: _HiddenSoftwarePipelineSolution,
) -> None:
    if not isinstance(task, PublicSoftwarePipelineTask):
        raise TypeError("learner projection must be a PublicSoftwarePipelineTask")
    if not isinstance(solution, _HiddenSoftwarePipelineSolution):
        raise TypeError("solution must be evaluator-owned hidden state")
    if _public_digest(task) != solution.public_digest:
        raise ValueError("public task and evaluator solution do not match")
    if {value.digest for value in task.grounded_candidates} != {
        value.action_digest for value in solution.implementations
    }:
        raise ValueError("public candidates and private components do not match")


def _validate_stream(stream: SoftwarePipelineStream) -> None:
    _validate_partition(stream.mechanism_partition)
    if stream.control_arm not in (
        "correct",
        "no_evidence",
        "wrong_evidence",
        "shuffled_outcome",
        "a_only",
        "b_only",
    ):
        raise ValueError("stream control arm is invalid")
    _require_digest(stream.mechanism_commitment, "mechanism_commitment")
    if stream.mechanism_commitment not in software_pipeline_mechanism_partition(
        stream.mechanism_partition
    ):
        raise ValueError("stream mechanism is outside its declared partition")
    if type(stream.supports) is not tuple or not stream.supports:
        raise ValueError("pipeline stream requires support packages")
    if type(stream.queries) is not tuple or not stream.queries:
        raise ValueError("pipeline stream requires query packages")
    pairs = (*stream.supports, *stream.queries)
    if any(not isinstance(value, GeneratedSoftwarePipelineTask) for value in pairs):
        raise TypeError("pipeline streams contain generated task pairs")
    if any(value.learner.observations for value in stream.queries):
        raise ValueError("query packages cannot contain execution observations")
    if stream.control_arm == "no_evidence":
        if any(value.learner.observations for value in stream.supports):
            raise ValueError("no-evidence supports cannot expose observations")
    elif any(not value.learner.observations for value in stream.supports):
        raise ValueError("evidence-bearing supports must expose observations")
    if any(
        value.hidden.mechanism_commitment != stream.mechanism_commitment
        or value.hidden.mechanism_partition != stream.mechanism_partition
        for value in pairs
    ):
        raise ValueError("stream packages differ from the declared mechanism")

    package_commitments = tuple(value.hidden.package_commitment for value in pairs)
    if len(set(package_commitments)) != len(package_commitments):
        raise ValueError("every package must have a fresh private commitment")
    public_digests = tuple(_public_digest(value.learner) for value in pairs)
    if len(set(public_digests)) != len(public_digests):
        raise ValueError("every package must have a fresh public projection")
    namespaces = tuple(value.learner.origin.namespace for value in pairs)
    if len(set(namespaces)) != len(namespaces):
        raise ValueError("every package must have a fresh opaque namespace")

    state_sets = [
        {state.digest for state in value.learner.states} for value in pairs
    ]
    component_sets = [
        {component.digest for component in value.learner.components}
        for value in pairs
    ]
    action_sets = [
        {action.digest for action in value.learner.grounded_candidates}
        for value in pairs
    ]
    address_sets = [
        {
            _pair_address(state.digest, action.digest)
            for state in value.learner.states
            for action in value.learner.grounded_candidates
        }
        for value in pairs
    ]
    for collections, label in (
        (state_sets, "state"),
        (component_sets, "component"),
        (action_sets, "grounded action"),
        (address_sets, "pair address"),
    ):
        for left_index, left in enumerate(collections):
            if any(left & right for right in collections[left_index + 1 :]):
                raise ValueError(f"pipeline packages have overlapping {label} identities")

    topology_sets = [
        {_contract_topology_signature(component) for component in value.learner.components}
        for value in pairs
    ]
    if any(
        len(signatures) != len(pair.learner.components)
        for signatures, pair in zip(topology_sets, pairs, strict=True)
    ):
        raise ValueError("a package contains a collapsed anonymous topology")
    for left_index, left in enumerate(topology_sets):
        if any(left & right for right in topology_sets[left_index + 1 :]):
            raise ValueError("pipeline packages repeat an anonymous topology")

    query_motif_sets = {value.hidden.required_motifs for value in stream.queries}
    if len(query_motif_sets) != 1:
        raise ValueError("query packages must share one composed motif pair")
    query_motifs = next(iter(query_motif_sets))
    if len(query_motifs) != 2:
        raise ValueError("query packages must compose exactly two motifs")
    support_motifs = tuple(value.hidden.required_motifs for value in stream.supports)
    if any(len(value) != 1 for value in support_motifs):
        raise ValueError("support packages must demonstrate separate motifs")
    represented = set(value[0] for value in support_motifs)
    if stream.control_arm in ("a_only", "b_only"):
        expected = {query_motifs[0 if stream.control_arm == "a_only" else 1]}
        if represented != expected:
            raise ValueError("motif-pure control contains the wrong support family")
    elif represented != set(query_motifs):
        raise ValueError("supports must cover both composed query motifs separately")


def _validate_pipeline_for_task(
    task: PublicSoftwarePipelineTask,
    pipeline: CommittedSoftwarePipeline,
) -> None:
    if pipeline.public_digest != _public_digest(task):
        raise ValueError("pipeline and public task do not match")
    if len(pipeline.actions) > task.max_steps:
        raise ValueError("pipeline exceeds the public step budget")
    declared = set(task.grounded_candidates)
    if any(value not in declared for value in pipeline.actions):
        raise ValueError("pipeline contains an undeclared component")
    if len(pipeline.actions) < task.max_steps and not pipeline.stopped:
        raise ValueError("a short pipeline requires explicit STOP")


def _semantic_partition(partition: str) -> tuple[_PipelineMechanism, ...]:
    if partition == "sealed":
        start = _SEALED_OFFSET
        count = _SEALED_COUNT
    else:
        _validate_partition(partition)
        start = _PARTITION_OFFSETS[partition]
        count = _PARTITION_SIZES[partition]
    return tuple(_mechanism_from_index(index) for index in range(start, start + count))


def _mechanism_from_index(index: int) -> _PipelineMechanism:
    pair = _MOTIF_PAIRS[(index * 7 + index // 10) % len(_MOTIF_PAIRS)]
    return _PipelineMechanism(
        semantic_index=index,
        motifs=pair,
        variants=((index * 3 + 1) % 11, (index * 5 + 2) % 13),
        presentation_variant=(index * 17 + 3) % 19,
    )


def _mechanism_commitment(mechanism: _PipelineMechanism) -> str:
    return _digest_payload(
        _MECHANISM_DIGEST_DOMAIN,
        {
            "semantic_index": mechanism.semantic_index,
            "motifs": mechanism.motifs,
            "presentation_variant": mechanism.presentation_variant,
            "variants": mechanism.variants,
        },
    )


def _pair_address(state_digest: str, action_digest: str) -> str:
    _require_digest(state_digest, "pair state digest")
    _require_digest(action_digest, "pair action digest")
    material = (
        _PAIR_ADDRESS_DOMAIN
        + state_digest.encode("ascii")
        + b"\x00"
        + action_digest.encode("ascii")
    )
    return "sha256:" + hashlib.sha256(material).hexdigest()


def _public_digest(task: PublicSoftwarePipelineTask) -> str:
    payload = json.dumps(
        task.to_canonical(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(_PUBLIC_DIGEST_DOMAIN + payload).hexdigest()


def _record_payload(record: Record) -> dict[str, object]:
    return {"arguments": list(record.arguments), "predicate": record.predicate}


def _state_payload(state: State) -> dict[str, object]:
    return {
        "namespace": state.namespace,
        "records": [_record_payload(value) for value in state.records],
    }


def _action_schema_payload(action: ActionSchema) -> dict[str, object]:
    return {
        "description": action.description,
        "name": action.name,
        "parameters": [
            {"name": value.name, "type_name": value.type_name}
            for value in action.parameters
        ],
    }


def _ground_action_payload(action: GroundAction) -> dict[str, object]:
    return {
        "arguments": list(action.arguments),
        "schema": _action_schema_payload(action.schema),
    }


def _goal_payload(goal: Goal) -> dict[str, object]:
    return {
        "exact": goal.exact,
        "forbidden": [_record_payload(value) for value in goal.forbidden],
        "namespace": goal.namespace,
        "required": [_record_payload(value) for value in goal.required],
    }


def _trace_payload(trace: Trace) -> dict[str, object]:
    return {
        "initial": _state_payload(trace.initial),
        "transitions": [
            {
                "action": _ground_action_payload(value.action),
                "after": _state_payload(value.after),
                "applied": value.applied,
                "before": _state_payload(value.before),
                "outcome": value.outcome,
            }
            for value in trace.transitions
        ],
    }


def _opaque_token(
    surface_seed: int,
    commitment: str,
    scope: str,
    package_index: int,
    role: str,
    role_index: int,
) -> str:
    material = (
        _OPAQUE_DOMAIN
        + str(surface_seed).encode("ascii")
        + b"\x00"
        + commitment.encode("ascii")
        + b"\x00"
        + scope.encode("ascii")
        + b"\x00"
        + str(package_index).encode("ascii")
        + b"\x00"
        + role.encode("ascii")
        + b"\x00"
        + str(role_index).encode("ascii")
    )
    return hashlib.sha256(material).hexdigest()[:20]


def _domain_seed(seed: int, scope: str, index: int, variant: int) -> int:
    material = (
        f"project-angler.microrepo.domain.v1\x00{seed}\x00{scope}\x00"
        f"{index}\x00{variant}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest(), "big")


def _digest_payload(domain: bytes, payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(domain + encoded).hexdigest()


def _require_digest(value: str, label: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical sha256 digest")


def _require_qualified_name(value: str, label: str) -> None:
    if not isinstance(value, str) or _QUALIFIED_NAME.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical qualified name")


def _require_atom_tuple(value: object, label: str) -> None:
    if type(value) is not tuple or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise TypeError(f"{label} must be an immutable non-empty-string tuple")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} cannot contain duplicates")


def _validate_partition(partition: str) -> None:
    if partition not in _PARTITION_SIZES:
        raise ValueError("partition must be train, development, or final")


def _validate_seed(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")


def _validate_positive_count(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


__all__ = [
    "CommittedSoftwarePipeline",
    "GeneratedSoftwarePipelineTask",
    "PipelineControlArm",
    "PipelinePartition",
    "PublicComponentContract",
    "PublicSoftwarePipelineTask",
    "SoftwarePipelineStream",
    "commit_software_pipeline",
    "judge_software_pipeline_attempt",
    "make_software_pipeline_control_stream",
    "make_software_pipeline_stream",
    "software_pipeline_mechanism_partition",
]
