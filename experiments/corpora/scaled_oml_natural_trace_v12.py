"""Scaled, leakage-separated natural-trace corpus for V12 OML.

The generator may construct and judge synthetic software procedures.  Learner
payloads contain only public natural language, chronological outcome feedback,
Moving-Origin coordinates, and challenge candidates.  Family keys, action
keys, renderer plans, topology signatures, transition classes, seeds, and
answer indices remain in generator metadata or supervision.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Literal

from experiments.corpora.causal_neuromodulated_apprenticeship_v6 import (
    FinalPartitionSealedError,
)
from experiments.corpora.outcome_aware_apprenticeship_v5 import (
    ApprenticeshipChallenge,
    CorpusMechanism,
    GeneratorMetadata,
    MechanismSupervision,
    MovingOriginCoordinates,
    OutcomeEpisode,
    PublicApprenticeshipStream,
)


CORPUS_ID = "angler.scaled-oml-natural-trace.v12"
TRAIN_FAMILIES = 48
DEVELOPMENT_FAMILIES = 12
FINAL_FAMILIES = 12
TRAIN_INNER_MECHANISMS = 384
TRAIN_OUTER_UPDATES = 192
TRAIN_OUTER_SLOTS = 8
TRAIN_OUTER_MECHANISMS = TRAIN_OUTER_UPDATES * TRAIN_OUTER_SLOTS
DEVELOPMENT_MECHANISMS = 48
FINAL_MECHANISMS = 48
EPISODES_PER_MECHANISM = 6
CANDIDATES_PER_CHALLENGE = 4

Partition = Literal["train", "development", "final"]
SurfaceScope = Literal["inner", "outer", "development", "final"]
Transition = Literal["insertion", "removal", "reorder", "replacement"]
OutcomeLabel = Literal["SUCCESS", "FAILURE"]

_SALT = b"project-angler.scaled-oml-natural-trace.v12\x00"
_OUTCOME_PATTERNS: tuple[tuple[OutcomeLabel, ...], ...] = (
    ("FAILURE", "SUCCESS", "FAILURE", "SUCCESS", "SUCCESS", "FAILURE"),
    ("SUCCESS", "FAILURE", "SUCCESS", "FAILURE", "FAILURE", "SUCCESS"),
    ("FAILURE", "SUCCESS", "SUCCESS", "FAILURE", "SUCCESS", "FAILURE"),
    ("SUCCESS", "FAILURE", "FAILURE", "SUCCESS", "FAILURE", "SUCCESS"),
)
_TRANSITIONS: tuple[Transition, ...] = (
    "insertion",
    "removal",
    "reorder",
    "replacement",
)


@dataclass(frozen=True, slots=True)
class _ActionSpec:
    key: str
    verbs: tuple[str, str, str, str]
    object_text: str
    purpose_text: str


# Every action owns four reviewed synonymous verbs and two sentence frames,
# yielding eight public render plans.  Keys are generator-only.
_ACTION_SPECS: tuple[_ActionSpec, ...] = (
    _ActionSpec("inspect_boundary", ("inspect", "review", "examine", "survey"), "the declared dependency boundary", "before state changes begin"),
    _ActionSpec("record_checkpoint", ("record", "capture", "save", "preserve"), "a reversible workspace checkpoint", "before any mutation"),
    _ActionSpec("parse_input", ("parse", "read", "decode", "interpret"), "the authoritative synthetic input", "into its public intermediate form"),
    _ActionSpec("check_schema", ("check", "validate", "verify", "assess"), "the intermediate representation", "against the declared schema"),
    _ActionSpec("plan_change", ("plan", "outline", "prepare", "formulate"), "the bounded workspace change", "from observed public facts"),
    _ActionSpec("stage_candidate", ("stage", "prepare", "assemble", "place"), "the candidate artifact", "without activating it"),
    _ActionSpec("apply_change", ("apply", "perform", "make", "execute"), "the requested bounded change", "inside the isolated workspace"),
    _ActionSpec("rebuild_projection", ("rebuild", "regenerate", "reconstruct", "refresh"), "the derived state projection", "from authoritative inputs"),
    _ActionSpec("reconcile_state", ("reconcile", "align", "merge", "integrate"), "the changed workspace state", "into one consistent projection"),
    _ActionSpec("compare_boundary", ("compare", "contrast", "check", "measure"), "the candidate state", "against the acceptance boundary"),
    _ActionSpec("run_unit_check", ("run", "execute", "perform", "complete"), "the focused unit check", "over the changed interface"),
    _ActionSpec("run_integration_check", ("run", "execute", "perform", "complete"), "the integration check", "across the dependency path"),
    _ActionSpec("probe_rollback", ("probe", "exercise", "test", "verify"), "the rollback path", "and restore the candidate state"),
    _ActionSpec("audit_result", ("audit", "review", "inspect", "assess"), "the candidate result", "against the declared request"),
    _ActionSpec("verify_result", ("verify", "validate", "check", "confirm"), "the completed workspace", "with the acceptance check"),
    _ActionSpec("publish_artifact", ("publish", "release", "emit", "expose"), "the verified synthetic artifact", "to the isolated output boundary"),
    _ActionSpec("activate_state", ("activate", "enable", "promote", "start"), "the already verified candidate state", "within the synthetic workspace"),
    _ActionSpec("inventory_capacity", ("inventory", "enumerate", "review", "measure"), "the declared resource capacity", "before allocation"),
    _ActionSpec("allocate_resources", ("allocate", "reserve", "assign", "provision"), "the admitted synthetic resources", "within declared limits"),
    _ActionSpec("normalize_record", ("normalize", "canonicalize", "standardize", "regularize"), "the validated synthetic record", "into canonical form"),
    _ActionSpec("index_record", ("index", "catalog", "register", "organize"), "the canonical record", "for bounded retrieval"),
    _ActionSpec("replay_evidence", ("replay", "reproduce", "rerun", "recheck"), "the recorded evidence", "through the acceptance path"),
    _ActionSpec("confirm_receipt", ("confirm", "record", "capture", "document"), "the objective execution receipt", "after verification"),
    _ActionSpec("release_lock", ("release", "clear", "drop", "remove"), "the synchronization boundary", "after the bounded operation"),
)
_ACTION_BY_KEY = {value.key: value for value in _ACTION_SPECS}
_ACTION_KEYS = tuple(_ACTION_BY_KEY)

_SYSTEMS = (
    "catalog service", "event processor", "policy adapter", "document index",
    "queue consumer", "configuration compiler", "artifact registry", "workflow gateway",
    "schema translator", "test coordinator", "cache manager", "release planner",
    "state reconciler", "package assembler", "contract monitor", "projection builder",
)
_ARTIFACTS = (
    "routing manifest", "compatibility layer", "validation profile", "dependency lock",
    "state projection", "interface contract", "generated index", "migration map",
    "event envelope", "build recipe", "cache projection", "release descriptor",
)
_CHANGES = (
    "support the revised record shape", "preserve a newly required output field",
    "adopt the updated dependency boundary", "repair stale state propagation",
    "accept the revised public contract", "handle the new optional branch",
    "separate preparation from activation", "retain format-transition compatibility",
    "reconstruct a missing derived entry", "bound a previously implicit transition",
)
_CONSTRAINTS = (
    "the existing rollback path", "unrelated public behavior", "the declared interface boundary",
    "previously accepted records", "the read-only source contract", "the bounded workspace scope",
    "the stable serialization format", "the existing error semantics", "the audit trail",
    "the current compatibility envelope",
)
_CHECKS = (
    "the contract suite", "the integration harness", "the replay check",
    "the compatibility test", "the state-transition verifier", "the package acceptance test",
    "the dependency audit", "the reconstruction check", "the publication check",
    "the rollback verifier",
)

_TASK_TEMPLATES = (
    "The synthetic {system} must {change} around its {artifact} while preserving {constraint}.",
    "A bounded change to the {artifact} in the synthetic {system} must {change} without weakening {constraint}.",
    "Within the synthetic {system}, revise the {artifact} to {change} and retain {constraint}.",
    "The isolated {system} workspace needs its {artifact} to {change} while keeping {constraint} intact.",
    "For this synthetic {system}, the {artifact} must {change}; {constraint} remains mandatory.",
    "Update the synthetic {system}'s {artifact} so it can {change}, with {constraint} preserved.",
    "The generated {system} scenario requires the {artifact} to {change} under {constraint}.",
    "In the bounded {system} exercise, make the {artifact} {change} and maintain {constraint}.",
)
_REQUEST_TEMPLATES = (
    "Complete the {artifact} work in the {system} so {check} passes and {constraint} remains intact.",
    "Produce a {system} candidate for the {artifact} accepted by {check} without weakening {constraint}.",
    "Follow a bounded {artifact} procedure for the {system} that satisfies {check} while retaining {constraint}.",
    "Finish the {system} change to the {artifact} and demonstrate it through {check}, preserving {constraint}.",
    "Return a verified {system} workspace for the {artifact} under {constraint}; {check} must pass.",
    "Use the available {system} actions around the {artifact} to meet {check} without compromising {constraint}.",
    "Construct an acceptable {artifact} result for the {system} under {constraint} and confirm it with {check}.",
    "Resolve the {system} request for the {artifact} inside {constraint}; {check} must pass.",
)
_TRANSITION_DESCRIPTIONS = {
    "insertion": "The revised boundary includes one additional prerequisite.",
    "removal": "The revised boundary no longer needs one obsolete operation.",
    "reorder": "A revised dependency must complete before its former predecessor.",
    "replacement": "The revised boundary calls for an alternate bounded operation.",
}

_PLAN_IDS: dict[SurfaceScope, tuple[int, ...]] = {
    "inner": (0, 1, 2, 3),
    "outer": (4, 5),
    "development": (6,),
    "final": (7,),
}
_TEMPLATE_IDS: dict[SurfaceScope, tuple[int, ...]] = {
    "inner": (0, 1, 2, 3),
    "outer": (4, 5),
    "development": (6,),
    "final": (7,),
}


@dataclass(frozen=True, slots=True)
class _FamilySpec:
    key: str
    partition: Partition
    ordinal: int
    support_steps: tuple[str, ...]
    challenge_steps: tuple[str, ...]
    transition: Transition

    @property
    def support_signature(self) -> str:
        return _structure_signature(self.support_steps)

    @property
    def challenge_signature(self) -> str:
        return _structure_signature(self.challenge_steps)


@dataclass(frozen=True, slots=True)
class TraceRepresentationContrasts:
    """Evaluator-only public traces; never part of a learner payload."""

    reference_trace_text: str
    paraphrase_trace_text: str
    reordered_trace_text: str
    omitted_trace_text: str
    inserted_trace_text: str
    replacement_trace_text: str


def _rng(*parts: object) -> random.Random:
    payload = "\x1f".join(str(value) for value in parts).encode("utf-8")
    digest = hashlib.sha256(_SALT + payload).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _structure_signature(steps: tuple[str, ...]) -> str:
    return "nodes:" + ",".join(steps) + "|edges:" + ",".join(
        f"{left}>{right}" for left, right in zip(steps, steps[1:])
    )


def _ordered_ngrams(steps: tuple[str, ...]) -> frozenset[tuple[str, ...]]:
    return frozenset(
        tuple(steps[index : index + width])
        for width in (2, 3)
        for index in range(len(steps) - width + 1)
    )


def _transition_steps(
    support: tuple[str, ...],
    transition: Transition,
    rng: random.Random,
) -> tuple[str, ...]:
    values = list(support)
    if transition == "insertion":
        candidate = rng.choice(tuple(key for key in _ACTION_KEYS if key not in values))
        values.insert(rng.randrange(len(values) + 1), candidate)
    elif transition == "removal":
        values.pop(rng.randrange(len(values)))
    elif transition == "reorder":
        left = rng.randrange(len(values) - 1)
        values[left], values[left + 1] = values[left + 1], values[left]
    elif transition == "replacement":
        index = rng.randrange(len(values))
        values[index] = rng.choice(tuple(key for key in _ACTION_KEYS if key not in values))
    else:  # pragma: no cover - guarded by the private type/table
        raise ValueError("unknown transition")
    result = tuple(values)
    if result == support:
        raise RuntimeError("transition did not alter the support procedure")
    return result


def _family_dimensions(partition: Partition, index: int) -> tuple[int, Transition]:
    if partition == "train":
        length = 3 + index % 4
        transition = _TRANSITIONS[index // 12]
    else:
        length = 3 + (index + (1 if partition == "final" else 0)) % 4
        transition = _TRANSITIONS[index // 3]
    return length, transition


def _generate_partition_families(
    partition: Partition,
    count: int,
    *,
    ordinal_start: int,
    forbidden_ngrams: set[tuple[str, ...]],
) -> tuple[tuple[_FamilySpec, ...], set[tuple[str, ...]]]:
    families: list[_FamilySpec] = []
    partition_ngrams: set[tuple[str, ...]] = set()
    signatures: set[tuple[str, str]] = set()
    for index in range(count):
        length, transition = _family_dimensions(partition, index)
        accepted: _FamilySpec | None = None
        for attempt in range(20_000):
            rng = _rng("family", partition, index, attempt)
            support = tuple(rng.sample(_ACTION_KEYS, length))
            challenge = _transition_steps(support, transition, rng)
            ngrams = set(_ordered_ngrams(support) | _ordered_ngrams(challenge))
            signature = (
                _structure_signature(support),
                _structure_signature(challenge),
            )
            if ngrams & forbidden_ngrams or signature in signatures:
                continue
            accepted = _FamilySpec(
                key=f"v12-{partition}-topology-{index:02d}",
                partition=partition,
                ordinal=ordinal_start + index,
                support_steps=support,
                challenge_steps=challenge,
                transition=transition,
            )
            signatures.add(signature)
            partition_ngrams.update(ngrams)
            break
        if accepted is None:
            raise RuntimeError(
                f"unable to construct {partition} topology family {index} "
                "without crossing reserved ordered n-grams"
            )
        families.append(accepted)
    return tuple(families), partition_ngrams


_TRAIN_FAMILY_VALUES, _TRAIN_NGRAMS = _generate_partition_families(
    "train", TRAIN_FAMILIES, ordinal_start=0, forbidden_ngrams=set()
)
_DEVELOPMENT_FAMILY_VALUES, _DEVELOPMENT_NGRAMS = _generate_partition_families(
    "development",
    DEVELOPMENT_FAMILIES,
    ordinal_start=TRAIN_FAMILIES,
    forbidden_ngrams=set(_TRAIN_NGRAMS),
)
_FINAL_FAMILY_VALUES, _FINAL_NGRAMS = _generate_partition_families(
    "final",
    FINAL_FAMILIES,
    ordinal_start=TRAIN_FAMILIES + DEVELOPMENT_FAMILIES,
    forbidden_ngrams=set(_TRAIN_NGRAMS | _DEVELOPMENT_NGRAMS),
)
_FAMILIES = (
    *_TRAIN_FAMILY_VALUES,
    *_DEVELOPMENT_FAMILY_VALUES,
    *_FINAL_FAMILY_VALUES,
)
_FAMILY_BY_KEY = {value.key: value for value in _FAMILIES}


def _validate_family_design() -> None:
    expected = {"train": 48, "development": 12, "final": 12}
    action_sets: dict[str, set[str]] = {}
    signature_sets: dict[str, set[tuple[str, str]]] = {}
    ngram_sets: dict[str, set[tuple[str, ...]]] = {}
    for partition, count in expected.items():
        values = tuple(value for value in _FAMILIES if value.partition == partition)
        if len(values) != count:
            raise RuntimeError("scaled family count is invalid")
        if {len(value.support_steps) for value in values} != {3, 4, 5, 6}:
            raise RuntimeError("scaled family lengths are incomplete")
        if any(
            sum(len(value.support_steps) == length for value in values) != count // 4
            for length in (3, 4, 5, 6)
        ):
            raise RuntimeError("scaled family lengths are not balanced")
        if any(
            sum(value.transition == transition for value in values) != count // 4
            for transition in _TRANSITIONS
        ):
            raise RuntimeError("scaled family transitions are not balanced")
        action_sets[partition] = {
            action
            for value in values
            for action in (*value.support_steps, *value.challenge_steps)
        }
        signature_sets[partition] = {
            (value.support_signature, value.challenge_signature) for value in values
        }
        ngram_sets[partition] = {
            ngram
            for value in values
            for ngram in (_ordered_ngrams(value.support_steps) | _ordered_ngrams(value.challenge_steps))
        }
    for heldout in ("development", "final"):
        if not action_sets[heldout] <= action_sets["train"]:
            raise RuntimeError("heldout action vocabulary is not shared with training")
    for left, right in (
        ("train", "development"),
        ("train", "final"),
        ("development", "final"),
    ):
        if signature_sets[left] & signature_sets[right]:
            raise RuntimeError("complete topology signatures cross partitions")
        if ngram_sets[left] & ngram_sets[right]:
            raise RuntimeError("ordered heldout n-grams cross partitions")


_validate_family_design()


def _context(scope: SurfaceScope, family: _FamilySpec, surface_ordinal: int) -> dict[str, str]:
    # Mixed-radix Latin enumeration gives each surface a counterbalanced
    # context tuple before the 1.92-million-tuple cycle repeats.  No family key
    # selects a lexical subset, so topology cannot reserve context vocabulary.
    context_index = surface_ordinal % (
        len(_SYSTEMS)
        * len(_ARTIFACTS)
        * len(_CHANGES)
        * len(_CONSTRAINTS)
        * len(_CHECKS)
    )
    system_index = context_index % len(_SYSTEMS)
    context_index //= len(_SYSTEMS)
    artifact_index = context_index % len(_ARTIFACTS)
    context_index //= len(_ARTIFACTS)
    change_index = context_index % len(_CHANGES)
    context_index //= len(_CHANGES)
    constraint_index = context_index % len(_CONSTRAINTS)
    context_index //= len(_CONSTRAINTS)
    check_index = context_index % len(_CHECKS)
    return {
        "system": _SYSTEMS[system_index],
        "artifact": _ARTIFACTS[artifact_index],
        "change": _CHANGES[change_index],
        "constraint": _CONSTRAINTS[constraint_index],
        "check": _CHECKS[check_index],
        "scope": scope,
    }


def _render_action(
    action_key: str,
    context: dict[str, str],
    *,
    surface_ordinal: int,
    plan_override: int | None = None,
) -> str:
    spec = _ACTION_BY_KEY[action_key]
    scope = context["scope"]
    assert scope in _PLAN_IDS
    plans = _PLAN_IDS[scope]  # type: ignore[index]
    action_ordinal = _ACTION_KEYS.index(action_key)
    plan = (
        plans[(surface_ordinal + 5 * action_ordinal) % len(plans)]
        if plan_override is None
        else plan_override
    )
    verb = spec.verbs[plan % len(spec.verbs)]
    if plan == 0:
        text = f"{verb.capitalize()} {spec.object_text} {spec.purpose_text} for the {context['artifact']}."
    elif plan == 1:
        text = f"For the {context['system']}, {verb} {spec.object_text} {spec.purpose_text}."
    elif plan == 2:
        text = f"{verb.capitalize()} {spec.object_text} around the {context['artifact']} {spec.purpose_text}."
    elif plan == 3:
        text = f"Within the {context['system']}, {verb} {spec.object_text} {spec.purpose_text}."
    elif plan == 4:
        text = f"Before proceeding in the {context['system']}, {verb} {spec.object_text} {spec.purpose_text}."
    elif plan == 5:
        text = f"Using the {context['artifact']} boundary, {verb} {spec.object_text} {spec.purpose_text}."
    elif plan == 6:
        text = f"In this generated workspace, {verb} {spec.object_text} for the {context['system']} {spec.purpose_text}."
    elif plan == 7:
        text = f"The next bounded operation must {verb} {spec.object_text} {spec.purpose_text} around the {context['artifact']}."
    elif plan == 8:
        text = f"As a bounded workspace step, {verb} {spec.object_text} {spec.purpose_text} for the {context['system']}."
    else:  # pragma: no cover - plan tables are frozen above
        raise RuntimeError("invalid paraphrase plan")
    return (
        text
        + f" It must support the request to {context['change']} while preserving "
        + f"{context['constraint']}."
    )


def _trace_text(
    steps: tuple[str, ...],
    context: dict[str, str],
    *,
    surface_ordinal: int,
    plan_override: int | None = None,
) -> str:
    rendered = {
        action: _render_action(
            action,
            context,
            surface_ordinal=surface_ordinal,
            plan_override=plan_override,
        )
        for action in set(steps)
    }
    return " ".join(
        f"Step {index + 1}: {rendered[action]}"
        for index, action in enumerate(steps)
    )


def build_trace_representation_contrasts(
    mechanism: CorpusMechanism,
) -> TraceRepresentationContrasts:
    """Build hidden-label representation diagnostics without parsing IDs.

    The reference and reordered traces use the exact same rendered sentence
    map, so their raw step-sentence multisets are equal.  The paraphrase uses a
    ninth evaluator-only frame over the same actions and context.  None of
    these traces enters OML training or ``to_learner_payload``.
    """

    if not isinstance(mechanism, CorpusMechanism):
        raise TypeError("mechanism must be CorpusMechanism")
    try:
        family = _FAMILY_BY_KEY[mechanism.metadata.generator_family]
    except KeyError as exc:
        raise ValueError("mechanism does not belong to the V12 generator") from exc
    token_bytes = hashlib.sha256(
        _SALT + mechanism.metadata.mechanism_ref.encode("utf-8")
    ).digest()
    token = int.from_bytes(token_bytes[:8], "big")
    surface = 4_000_000 + token % 1_000_000
    context = _context("development", family, surface)
    target = family.challenge_steps
    values = {
        "reference_trace_text": _trace_text(
            target,
            context,
            surface_ordinal=surface,
            plan_override=6,
        ),
        "paraphrase_trace_text": _trace_text(
            target,
            context,
            surface_ordinal=surface,
            plan_override=8,
        ),
    }
    for field, mode, offset in (
        ("reordered_trace_text", "reorder", 1),
        ("omitted_trace_text", "removal", 2),
        ("inserted_trace_text", "insertion", 3),
        ("replacement_trace_text", "replacement", 4),
    ):
        values[field] = _trace_text(
            _corrupt_steps(target, mode, token=token + offset),
            context,
            surface_ordinal=surface,
            plan_override=6,
        )
    return TraceRepresentationContrasts(**values)


def _task_text(
    context: dict[str, str],
    family: _FamilySpec,
    *,
    surface_ordinal: int,
    heldout: bool,
) -> str:
    scope = context["scope"]
    template = _template_id(scope, family, surface_ordinal, request=False)
    text = _TASK_TEMPLATES[template].format(**context)
    if heldout:
        text += " " + _TRANSITION_DESCRIPTIONS[family.transition]
    return text


def _request_text(
    context: dict[str, str],
    family: _FamilySpec,
    *,
    surface_ordinal: int,
) -> str:
    scope = context["scope"]
    template = _template_id(scope, family, surface_ordinal, request=True)
    return _REQUEST_TEMPLATES[template].format(**context)


def _template_id(
    scope: str,
    family: _FamilySpec,
    surface_ordinal: int,
    *,
    request: bool,
) -> int:
    if scope not in _TEMPLATE_IDS:
        raise ValueError("surface scope has no public template route")
    templates = _TEMPLATE_IDS[scope]  # type: ignore[index]
    offset = 3 * family.ordinal + 1 if request else family.ordinal
    return templates[(surface_ordinal + offset) % len(templates)]


def _diagnostic(context: dict[str, str], outcome: OutcomeLabel) -> str:
    if outcome == "SUCCESS":
        return (
            f"Objective verification passed: {context['check']} accepted the result "
            f"and {context['constraint']} remained intact."
        )
    return (
        f"Objective verification failed: {context['check']} rejected the result "
        f"under {context['constraint']}."
    )


def _different_action(steps: tuple[str, ...], token: int) -> str:
    available = tuple(value for value in _ACTION_KEYS if value not in steps)
    return available[token % len(available)]


def _corrupt_steps(
    steps: tuple[str, ...],
    mode: Transition,
    *,
    token: int,
) -> tuple[str, ...]:
    values = list(steps)
    if mode == "reorder":
        left = token % (len(values) - 1)
        values[left], values[left + 1] = values[left + 1], values[left]
    elif mode == "removal":
        values.pop(token % len(values))
    elif mode == "insertion":
        values.insert(token % (len(values) + 1), _different_action(steps, token))
    elif mode == "replacement":
        values[token % len(values)] = _different_action(steps, token)
    else:  # pragma: no cover
        raise ValueError("unknown corruption mode")
    result = tuple(values)
    if result == steps:
        raise RuntimeError("corruption did not change the trace")
    return result


def _build_mechanism(
    family: _FamilySpec,
    *,
    scope: SurfaceScope,
    mechanism_index: int,
    family_local_index: int,
    ordinal_start: int,
) -> CorpusMechanism:
    outcomes = _OUTCOME_PATTERNS[mechanism_index % len(_OUTCOME_PATTERNS)]
    failure_modes = tuple(
        _TRANSITIONS[(mechanism_index + offset) % len(_TRANSITIONS)]
        for offset in range(3)
    )
    failure_cursor = 0
    episodes: list[OutcomeEpisode] = []
    surface_base = {
        "inner": 0,
        "outer": 1_000_000,
        "development": 2_000_000,
        "final": 3_000_000,
    }[scope] + mechanism_index * 17 + (family_local_index if scope == "outer" else 0)
    for episode_index, outcome in enumerate(outcomes):
        surface = surface_base + episode_index
        context = _context(scope, family, surface)
        if outcome == "SUCCESS":
            steps = family.support_steps
        else:
            mode = failure_modes[failure_cursor]
            failure_cursor += 1
            steps = _corrupt_steps(
                family.support_steps,
                mode,
                token=mechanism_index * 7 + episode_index,
            )
        episodes.append(
            OutcomeEpisode(
                task_text=_task_text(
                    context,
                    family,
                    surface_ordinal=surface,
                    heldout=False,
                ),
                request_text=_request_text(
                    context,
                    family,
                    surface_ordinal=surface,
                ),
                action_trace_text=_trace_text(
                    steps,
                    context,
                    surface_ordinal=surface,
                ),
                outcome_label=outcome,
                objective_diagnostic_text=_diagnostic(context, outcome),
                temporal=MovingOriginCoordinates(
                    acquired_ordinal=ordinal_start + episode_index,
                    age=EPISODES_PER_MECHANISM - episode_index - 1,
                    landmark_relations=((
                        "stream_start",
                        "AT" if episode_index == 0 else "AFTER",
                    ),),
                ),
            )
        )

    surface = surface_base + EPISODES_PER_MECHANISM + 1
    context = _context(scope, family, surface)
    target_steps = family.challenge_steps
    # Reorder and safe replacement give two length-matched distractors.  The
    # third alternates shorter/longer so target length rank is balanced.
    third_mode: Transition = "removal" if mechanism_index % 2 == 0 else "insertion"
    failed_steps = (
        _corrupt_steps(target_steps, "reorder", token=mechanism_index * 11 + 1),
        _corrupt_steps(target_steps, "replacement", token=mechanism_index * 11 + 2),
        _corrupt_steps(target_steps, third_mode, token=mechanism_index * 11 + 3),
    )
    target_trace = _trace_text(target_steps, context, surface_ordinal=surface)
    failed_traces = [
        _trace_text(value, context, surface_ordinal=surface)
        for value in failed_steps
    ]
    if len(set((target_trace, *failed_traces))) != CANDIDATES_PER_CHALLENGE:
        raise RuntimeError("challenge candidate construction produced a duplicate")
    target_position = (family.ordinal + family_local_index) % CANDIDATES_PER_CHALLENGE
    candidates = list(failed_traces)
    candidates.insert(target_position, target_trace)
    mechanism_ref = (
        f"{family.partition}-v12-{scope}-mechanism-"
        f"{family.ordinal:02d}-{family_local_index:04d}"
    )
    return CorpusMechanism(
        public=PublicApprenticeshipStream(
            episodes=tuple(episodes),
            challenge=ApprenticeshipChallenge(
                task_text=_task_text(
                    context,
                    family,
                    surface_ordinal=surface,
                    heldout=True,
                ),
                request_text=_request_text(
                    context,
                    family,
                    surface_ordinal=surface,
                ),
                candidate_action_trace_texts=tuple(candidates),
            ),
        ),
        supervision=MechanismSupervision(
            target_successful_candidate_indices=(target_position,),
            relevant_failed_candidate_indices=tuple(
                index
                for index in range(CANDIDATES_PER_CHALLENGE)
                if index != target_position
            ),
        ),
        metadata=GeneratorMetadata(
            partition=family.partition,
            mechanism_ref=mechanism_ref,
            generator_family=family.key,
            support_structure_signature=family.support_signature,
            challenge_structure_signature=family.challenge_signature,
            heldout_variant=family.transition,
        ),
    )


@dataclass(frozen=True, slots=True)
class ScaledOmlNaturalTraceCorpus:
    train_inner: tuple[CorpusMechanism, ...]
    development: tuple[CorpusMechanism, ...]
    _final: tuple[CorpusMechanism, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.train_inner) != TRAIN_INNER_MECHANISMS:
            raise ValueError("train_inner has the wrong mechanism count")
        if len(self.development) != DEVELOPMENT_MECHANISMS:
            raise ValueError("development has the wrong mechanism count")
        if self._final is not None and len(self._final) != FINAL_MECHANISMS:
            raise ValueError("final has the wrong mechanism count")
        if any(not value.gradient_authorized for value in self.train_inner):
            raise ValueError("train_inner must be gradient-authorized")
        if any(value.gradient_authorized for value in self.development):
            raise ValueError("development must not authorize gradients")
        if self._final is not None and any(value.gradient_authorized for value in self._final):
            raise ValueError("final must not authorize gradients")

    @property
    def final(self) -> tuple[CorpusMechanism, ...]:
        if self._final is None:
            raise FinalPartitionSealedError(
                "final remains sealed until the V12 development gate authorizes it"
            )
        return self._final

    @property
    def final_is_open(self) -> bool:
        return self._final is not None

    def partition(self, name: str) -> tuple[CorpusMechanism, ...]:
        if name == "train_inner":
            return self.train_inner
        if name == "development":
            return self.development
        if name == "final":
            return self.final
        raise ValueError("partition is invalid")

    def train_inner_for_update(self, update_index: int) -> tuple[CorpusMechanism, ...]:
        if type(update_index) is not int or not 0 <= update_index < TRAIN_OUTER_UPDATES:
            raise ValueError("update_index is outside the frozen V12 schedule")
        start = update_index * TRAIN_OUTER_SLOTS
        return tuple(
            # 49 is coprime to 384 and separates the eight slots across eight
            # topology families while retaining exactly four total exposures
            # per mechanism over the 192-update schedule.
            self.train_inner[((start + slot) * 49) % TRAIN_INNER_MECHANISMS]
            for slot in range(TRAIN_OUTER_SLOTS)
        )

    def train_outer(self, update_index: int, slot_index: int) -> CorpusMechanism:
        if type(update_index) is not int or not 0 <= update_index < TRAIN_OUTER_UPDATES:
            raise ValueError("update_index is outside the frozen V12 schedule")
        if type(slot_index) is not int or not 0 <= slot_index < TRAIN_OUTER_SLOTS:
            raise ValueError("slot_index is outside the frozen V12 schedule")
        outer_index = update_index * TRAIN_OUTER_SLOTS + slot_index
        family = _TRAIN_FAMILY_VALUES[outer_index % TRAIN_FAMILIES]
        family_local = outer_index // TRAIN_FAMILIES
        return _build_mechanism(
            family,
            scope="outer",
            mechanism_index=outer_index,
            family_local_index=family_local,
            ordinal_start=1_000_000 + outer_index * EPISODES_PER_MECHANISM,
        )

    def learner_payload_size_bytes(self, *, include_outer: bool = False) -> int:
        values = [value.to_learner_payload() for value in self.train_inner]
        values.extend(value.to_learner_payload() for value in self.development)
        if self._final is not None:
            values.extend(value.to_learner_payload() for value in self._final)
        if include_outer:
            values.extend(
                self.train_outer(update, slot).to_learner_payload()
                for update in range(TRAIN_OUTER_UPDATES)
                for slot in range(TRAIN_OUTER_SLOTS)
            )
        return len(
            json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )


def _materialize_partition(
    families: tuple[_FamilySpec, ...],
    *,
    scope: SurfaceScope,
    mechanisms_per_family: int,
    ordinal_base: int,
) -> tuple[CorpusMechanism, ...]:
    values: list[CorpusMechanism] = []
    global_index = 0
    for family in families:
        for local_index in range(mechanisms_per_family):
            values.append(
                _build_mechanism(
                    family,
                    scope=scope,
                    mechanism_index=global_index,
                    family_local_index=local_index,
                    ordinal_start=ordinal_base + global_index * EPISODES_PER_MECHANISM,
                )
            )
            global_index += 1
    return tuple(values)


def build_scaled_oml_natural_trace_v12(
    *,
    include_final: bool = False,
) -> ScaledOmlNaturalTraceCorpus:
    if type(include_final) is not bool:
        raise TypeError("include_final must be bool")
    train_inner = _materialize_partition(
        _TRAIN_FAMILY_VALUES,
        scope="inner",
        mechanisms_per_family=8,
        ordinal_base=0,
    )
    development = _materialize_partition(
        _DEVELOPMENT_FAMILY_VALUES,
        scope="development",
        mechanisms_per_family=4,
        ordinal_base=2_000_000,
    )
    final = (
        _materialize_partition(
            _FINAL_FAMILY_VALUES,
            scope="final",
            mechanisms_per_family=4,
            ordinal_base=3_000_000,
        )
        if include_final
        else None
    )
    return ScaledOmlNaturalTraceCorpus(
        train_inner=train_inner,
        development=development,
        _final=final,
    )


__all__ = [
    "CANDIDATES_PER_CHALLENGE",
    "CORPUS_ID",
    "DEVELOPMENT_FAMILIES",
    "DEVELOPMENT_MECHANISMS",
    "EPISODES_PER_MECHANISM",
    "FINAL_FAMILIES",
    "FINAL_MECHANISMS",
    "FinalPartitionSealedError",
    "ScaledOmlNaturalTraceCorpus",
    "TraceRepresentationContrasts",
    "TRAIN_FAMILIES",
    "TRAIN_INNER_MECHANISMS",
    "TRAIN_OUTER_MECHANISMS",
    "TRAIN_OUTER_SLOTS",
    "TRAIN_OUTER_UPDATES",
    "build_scaled_oml_natural_trace_v12",
    "build_trace_representation_contrasts",
]
