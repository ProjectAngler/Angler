"""Build small random-interaction corpora for causal-operator experiments.

The generator samples grounded actions without inspecting domain state.  A
domain executor is the sole source of applicability and successor feedback;
blocked samples are counted and never rewritten into successful transitions.
No goal, route, solver, shortest path, or evaluator case is consulted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from angler.procedures.alignment import (  # noqa: E402
    StructuralIsomorphismCandidate,
    find_structural_isomorphisms,
)
from angler.procedures.induction import (  # noqa: E402
    MDLOperatorInducer,
    OperatorCandidate,
    TraceSubsegment,
    cluster_subsegments,
    extract_trace_subsegments,
)
from angler.procedures.records import (  # noqa: E402
    ActionSchema,
    GroundAction,
    State,
    Trace,
    Transition,
)
from angler.worlds import relational_boxes as boxes  # noqa: E402
from angler.worlds import relational_files as files  # noqa: E402
from angler.worlds import relational_tokens as tokens  # noqa: E402


_GENERATOR_VERSION = "angler.causal-operator-experience.v1"
_POSITIONS = tuple(f"position_{index}" for index in range(4))


class ExperienceGenerationError(RuntimeError):
    """Raised when bounded random interaction cannot produce a valid corpus."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class CandidateSupport:
    namespace: str
    operator_digest: str
    support: int
    body_steps: int
    effects: int
    reconstruction_exemplars: int

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, str) or not self.namespace:
            raise ExperienceGenerationError("support namespace must be non-empty")
        if not isinstance(self.operator_digest, str) or not self.operator_digest.startswith(
            "sha256:"
        ):
            raise ExperienceGenerationError("support operator_digest must be sha256")
        for value in (
            self.support,
            self.body_steps,
            self.effects,
            self.reconstruction_exemplars,
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ExperienceGenerationError("support counts must be positive")


@dataclass(frozen=True, slots=True)
class DomainExperienceCorpus:
    """Executed traces and their unpromoted within-domain operator hypotheses."""

    namespace: str
    traces: tuple[Trace, ...]
    subsegments: tuple[TraceSubsegment, ...]
    candidates: tuple[OperatorCandidate, ...]
    attempted_actions: int
    blocked_attempts: int

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, str) or not self.namespace:
            raise ExperienceGenerationError("corpus namespace must be non-empty")
        if type(self.traces) is not tuple or not self.traces:
            raise ExperienceGenerationError("corpus requires traces")
        if type(self.subsegments) is not tuple or not self.subsegments:
            raise ExperienceGenerationError("corpus requires trace subsegments")
        if type(self.candidates) is not tuple or not self.candidates:
            raise ExperienceGenerationError("corpus requires induced candidates")
        if any(trace.initial.namespace != self.namespace for trace in self.traces):
            raise ExperienceGenerationError("trace crossed its corpus namespace")
        if any(item.namespace != self.namespace for item in self.subsegments):
            raise ExperienceGenerationError("subsegment crossed its corpus namespace")
        if any(
            item.operator.namespace != self.namespace for item in self.candidates
        ):
            raise ExperienceGenerationError("candidate crossed its corpus namespace")
        if (
            isinstance(self.attempted_actions, bool)
            or not isinstance(self.attempted_actions, int)
            or self.attempted_actions <= 0
            or isinstance(self.blocked_attempts, bool)
            or not isinstance(self.blocked_attempts, int)
            or not 0 <= self.blocked_attempts < self.attempted_actions
        ):
            raise ExperienceGenerationError("corpus attempt counts are invalid")

    @property
    def digest(self) -> str:
        return _digest(
            {
                "attempted_actions": self.attempted_actions,
                "blocked_attempts": self.blocked_attempts,
                "candidates": [item.operator.digest for item in self.candidates],
                "namespace": self.namespace,
                "subsegments": [item.digest for item in self.subsegments],
                "traces": [item.digest for item in self.traces],
            }
        )


@dataclass(frozen=True, slots=True)
class CausalOperatorExperience:
    """Phase-4 input bundle with an uncertified, externally checkable trio."""

    seed: int
    heldout_entity_prefix: str
    corpora: tuple[DomainExperienceCorpus, ...]
    selected_candidates: tuple[OperatorCandidate, OperatorCandidate, OperatorCandidate]
    pairwise_alignments: tuple[StructuralIsomorphismCandidate, ...]
    support: tuple[CandidateSupport, ...]

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ExperienceGenerationError("seed must be an integer")
        if (
            not isinstance(self.heldout_entity_prefix, str)
            or not self.heldout_entity_prefix
        ):
            raise ExperienceGenerationError("heldout entity prefix must be non-empty")
        if type(self.corpora) is not tuple or len(self.corpora) != 3:
            raise ExperienceGenerationError("experience requires three domain corpora")
        if len({item.namespace for item in self.corpora}) != 3:
            raise ExperienceGenerationError("experience corpora must be independent")
        if type(self.selected_candidates) is not tuple or len(
            self.selected_candidates
        ) != 3:
            raise ExperienceGenerationError("experience requires a candidate trio")
        if len(
            {item.operator.namespace for item in self.selected_candidates}
        ) != 3:
            raise ExperienceGenerationError("selected trio must span three namespaces")
        if type(self.pairwise_alignments) is not tuple or len(
            self.pairwise_alignments
        ) != 3:
            raise ExperienceGenerationError("selected trio requires three alignments")
        if type(self.support) is not tuple or len(self.support) != 3:
            raise ExperienceGenerationError("selected trio requires support metadata")

    @property
    def digest(self) -> str:
        return _digest(self.to_metadata())

    def to_metadata(self) -> dict[str, Any]:
        return {
            "corpora": [
                {
                    "attempted_actions": item.attempted_actions,
                    "blocked_attempts": item.blocked_attempts,
                    "candidate_count": len(item.candidates),
                    "digest": item.digest,
                    "namespace": item.namespace,
                    "subsegment_count": len(item.subsegments),
                    "trace_count": len(item.traces),
                }
                for item in self.corpora
            ],
            "generator": _GENERATOR_VERSION,
            "heldout_entity_prefix": self.heldout_entity_prefix,
            "pairwise_alignments": [item.digest for item in self.pairwise_alignments],
            "seed": self.seed,
            "selected_candidates": [
                item.operator.digest for item in self.selected_candidates
            ],
            "support": [
                {
                    "body_steps": item.body_steps,
                    "effects": item.effects,
                    "namespace": item.namespace,
                    "operator_digest": item.operator_digest,
                    "reconstruction_exemplars": item.reconstruction_exemplars,
                    "support": item.support,
                }
                for item in self.support
            ],
        }


@dataclass(frozen=True, slots=True)
class _DomainSpec:
    namespace: str
    schema: ActionSchema
    make_initial: Callable[[tuple[str, ...]], State]
    execute: Callable[[State, GroundAction], Transition]


def _token_initial(entities: tuple[str, ...]) -> State:
    return tokens.make_token_state((*entities, None))


def _file_initial(entities: tuple[str, ...]) -> State:
    links = tuple(
        (source, destination)
        for source in _POSITIONS
        for destination in _POSITIONS
        if source != destination
    )
    return files.make_file_state(
        tuple(zip(entities, _POSITIONS, strict=False)),
        links,
    )


def _box_initial(entities: tuple[str, ...]) -> State:
    contents = {
        position: ((entities[index],) if index < len(entities) else ())
        for index, position in enumerate(_POSITIONS)
    }
    # Capacity is an observed attribute, not part of the procedure's identity.
    # Vary it across independently generated traces so anti-unification can
    # learn a value role instead of accidentally memorizing ``limit_3``.
    capacity_material = "\x00".join(entities).encode("utf-8")
    capacity = 1 + (hashlib.sha256(capacity_material).digest()[0] % 3)
    capacities = {position: capacity for position in _POSITIONS}
    return boxes.make_box_state(contents, capacities)


_DOMAINS = (
    _DomainSpec(tokens.NAMESPACE, tokens.MOVE_TOKEN, _token_initial, tokens.execute_token_action),
    _DomainSpec(files.NAMESPACE, files.RELOCATE_FILE, _file_initial, files.execute_file_action),
    _DomainSpec(boxes.NAMESPACE, boxes.TRANSFER_ITEM, _box_initial, boxes.execute_box_action),
)


def _domain_rng(seed: int, namespace: str) -> random.Random:
    material = f"{_GENERATOR_VERSION}\x00{seed}\x00{namespace}".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))


def _fresh_entities(seed: int, namespace: str, trace_index: int) -> tuple[str, ...]:
    domain_tag = namespace.rsplit(".", maxsplit=1)[-1]
    trace_nonce = hashlib.sha256(
        f"{_GENERATOR_VERSION}\x00{seed}\x00{namespace}\x00{trace_index}".encode(
            "utf-8"
        )
    ).hexdigest()[:10]
    return tuple(
        f"train_{domain_tag}_{trace_nonce}_entity_{index}"
        for index in range(3)
    )


def _random_executed_trace(
    spec: _DomainSpec,
    *,
    seed: int,
    trace_index: int,
    steps: int,
) -> tuple[Trace, int, int]:
    rng = _domain_rng(seed + trace_index * 1009, spec.namespace)
    entities = _fresh_entities(seed, spec.namespace, trace_index)
    initial = spec.make_initial(entities)
    state = initial
    transitions: list[Transition] = []
    attempts = 0
    blocked = 0
    ceiling = steps * len(entities) * len(_POSITIONS) ** 2 * 20
    while len(transitions) < steps:
        attempts += 1
        if attempts > ceiling:
            raise ExperienceGenerationError(
                f"random interaction exhausted its bound in {spec.namespace}"
            )
        action = spec.schema.ground(
            rng.choice(entities),
            rng.choice(_POSITIONS),
            rng.choice(_POSITIONS),
        )
        observed = spec.execute(state, action)
        if not observed.applied:
            blocked += 1
            continue
        transitions.append(observed)
        state = observed.after
    return Trace(initial, tuple(transitions), goal=None), attempts, blocked


def _build_domain_corpus(
    spec: _DomainSpec,
    *,
    seed: int,
    traces_per_domain: int,
    steps_per_trace: int,
    minimum_support: int,
) -> DomainExperienceCorpus:
    traces: list[Trace] = []
    attempts = 0
    blocked = 0
    for trace_index in range(traces_per_domain):
        trace, trace_attempts, trace_blocked = _random_executed_trace(
            spec,
            seed=seed,
            trace_index=trace_index,
            steps=steps_per_trace,
        )
        traces.append(trace)
        attempts += trace_attempts
        blocked += trace_blocked

    subsegments = tuple(
        segment
        for trace in traces
        for segment in extract_trace_subsegments(
            trace,
            minimum_length=2,
            maximum_length=min(4, steps_per_trace),
        )
        if segment.delta.changed
    )
    inducer = MDLOperatorInducer(
        minimum_support=minimum_support,
        minimum_savings=1,
    )
    candidates = tuple(
        sorted(
            (
                candidate
                for cluster in cluster_subsegments(subsegments)
                if (candidate := inducer.allocate(cluster)) is not None
            ),
            key=lambda item: item.operator.digest,
        )
    )
    return DomainExperienceCorpus(
        namespace=spec.namespace,
        traces=tuple(traces),
        subsegments=tuple(sorted(subsegments, key=lambda item: item.digest)),
        candidates=candidates,
        attempted_actions=attempts,
        blocked_attempts=blocked,
    )


def _best_alignment(
    source: OperatorCandidate,
    target: OperatorCandidate,
) -> StructuralIsomorphismCandidate | None:
    matches = find_structural_isomorphisms(source.operator, target.operator)
    if not matches:
        return None
    return max(
        matches,
        key=lambda item: (
            item.coverage.matched_effects,
            item.coverage.matched_preconditions,
            item.digest,
        ),
    )


def _select_structural_trio(
    corpora: tuple[DomainExperienceCorpus, ...],
) -> tuple[
    tuple[OperatorCandidate, OperatorCandidate, OperatorCandidate],
    tuple[StructuralIsomorphismCandidate, ...],
]:
    choices: list[
        tuple[
            tuple[int, int, int, str],
            tuple[OperatorCandidate, OperatorCandidate, OperatorCandidate],
            tuple[StructuralIsomorphismCandidate, ...],
        ]
    ] = []
    for trio in itertools.product(*(item.candidates for item in corpora)):
        if len({item.operator.namespace for item in trio}) != 3:
            continue
        # The first held-out slice evaluates a two-action chunk.  Longer
        # candidates remain in each corpus for later composition experiments,
        # but are not silently substituted for that evaluator contract.
        if any(len(item.operator.body) != 2 for item in trio):
            continue
        alignments = tuple(
            _best_alignment(trio[left], trio[right])
            for left, right in ((0, 1), (0, 2), (1, 2))
        )
        if any(item is None for item in alignments):
            continue
        verified_shape = tuple(item for item in alignments if item is not None)
        score = (
            min(len(item.operator.exemplars) for item in trio),
            sum(item.coverage.matched_effects for item in verified_shape),
            sum(item.score.savings for item in trio),
            "|".join(item.operator.digest for item in trio),
        )
        choices.append((score, trio, verified_shape))
    if not choices:
        raise ExperienceGenerationError(
            "random corpora produced no pairwise structural relocation trio"
        )
    _, trio, alignments = max(choices, key=lambda item: item[0])
    return trio, alignments


def build_causal_operator_experience(
    *,
    seed: int = 42_017,
    traces_per_domain: int = 40,
    steps_per_trace: int = 4,
    minimum_support: int = 2,
    heldout_entity_prefix: str = "heldout_",
) -> CausalOperatorExperience:
    """Generate random executed experience and an uncertified structural trio."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    for value, label, minimum in (
        (traces_per_domain, "traces_per_domain", 2),
        (steps_per_trace, "steps_per_trace", 2),
        (minimum_support, "minimum_support", 2),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{label} must be an integer")
        if value < minimum:
            raise ValueError(f"{label} must be at least {minimum}")
    if steps_per_trace > 4:
        raise ValueError("steps_per_trace is bounded to four for this experiment")
    if (
        not isinstance(heldout_entity_prefix, str)
        or not heldout_entity_prefix
        or heldout_entity_prefix.startswith("train_")
    ):
        raise ValueError("heldout_entity_prefix must be non-empty and distinct")

    corpora = tuple(
        _build_domain_corpus(
            spec,
            seed=seed,
            traces_per_domain=traces_per_domain,
            steps_per_trace=steps_per_trace,
            minimum_support=minimum_support,
        )
        for spec in _DOMAINS
    )
    trio, alignments = _select_structural_trio(corpora)
    support = tuple(
        CandidateSupport(
            namespace=item.operator.namespace,
            operator_digest=item.operator.digest,
            support=len(item.operator.exemplars),
            body_steps=len(item.operator.body),
            effects=len(item.operator.effects),
            reconstruction_exemplars=sum(
                exemplar.reconstruction is not None
                for exemplar in item.operator.exemplars
            ),
        )
        for item in trio
    )
    result = CausalOperatorExperience(
        seed=seed,
        heldout_entity_prefix=heldout_entity_prefix,
        corpora=corpora,
        selected_candidates=trio,
        pairwise_alignments=alignments,
        support=support,
    )
    if any(
        argument.startswith(heldout_entity_prefix)
        for corpus in result.corpora
        for trace in corpus.traces
        for transition in trace.transitions
        for argument in transition.action.arguments
    ):
        raise ExperienceGenerationError("held-out entity leaked into training experience")
    return result


__all__ = [
    "CandidateSupport",
    "CausalOperatorExperience",
    "DomainExperienceCorpus",
    "ExperienceGenerationError",
    "build_causal_operator_experience",
]
