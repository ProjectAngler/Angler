"""Broad synthetic corpus for causal neuromodulated apprenticeship V6.

This successor preserves the V5 learner-facing API while expanding the
training generator from four topology families to sixteen.  Development and
final use eight additional, partition-disjoint topology families.  Generator
metadata and outer-loss labels remain separate from the public stream.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Literal

from experiments.corpora.outcome_aware_apprenticeship_v5 import (
    ApprenticeshipChallenge,
    CorpusMechanism,
    GeneratorMetadata,
    MechanismSupervision,
    MovingOriginCoordinates,
    OutcomeEpisode,
    PublicApprenticeshipStream,
)


CORPUS_ID = "angler.causal-neuromodulated-apprenticeship.v6"
TRAIN_MECHANISMS = 64
DEVELOPMENT_MECHANISMS = 8
FINAL_MECHANISMS = 8
EPISODES_PER_MECHANISM = 6
CANDIDATES_PER_CHALLENGE = 4
TRAIN_GENERATOR_FAMILIES = 16

Partition = Literal["train", "development", "final"]
OutcomeLabel = Literal["SUCCESS", "FAILURE"]

_GENERATION_SALT = b"project-angler.causal-neuromodulated-apprenticeship.v6\x00"
_OUTCOME_PATTERNS: tuple[tuple[OutcomeLabel, ...], ...] = (
    ("FAILURE", "SUCCESS", "FAILURE", "SUCCESS", "SUCCESS", "FAILURE"),
    ("SUCCESS", "FAILURE", "SUCCESS", "FAILURE", "FAILURE", "SUCCESS"),
    ("FAILURE", "SUCCESS", "SUCCESS", "FAILURE", "SUCCESS", "FAILURE"),
    ("SUCCESS", "FAILURE", "FAILURE", "SUCCESS", "FAILURE", "SUCCESS"),
)


class FinalPartitionSealedError(RuntimeError):
    """Raised when final data is requested before development authorization."""


@dataclass(frozen=True, slots=True)
class CausalNeuromodulatedApprenticeshipCorpus:
    """Frozen V6 partitions with strict non-feature generator provenance."""

    train: tuple[CorpusMechanism, ...]
    development: tuple[CorpusMechanism, ...]
    _final: tuple[CorpusMechanism, ...] | None = None

    def __post_init__(self) -> None:
        expected = {"train": TRAIN_MECHANISMS, "development": DEVELOPMENT_MECHANISMS}
        partitions = ["train", "development"]
        if self._final is not None:
            expected["final"] = FINAL_MECHANISMS
            partitions.append("final")
        seen_refs: set[str] = set()
        family_sets: dict[str, set[str]] = {}
        topology_sets: dict[str, set[tuple[str, str]]] = {}
        for partition in partitions:
            values = self._final if partition == "final" else getattr(self, partition)
            assert values is not None
            if type(values) is not tuple or len(values) != expected[partition]:
                raise ValueError(f"{partition} has the wrong mechanism count")
            if any(value.metadata.partition != partition for value in values):
                raise ValueError("mechanism appears in the wrong partition")
            refs = {value.metadata.mechanism_ref for value in values}
            if len(refs) != len(values) or refs & seen_refs:
                raise ValueError("mechanism references must be globally disjoint")
            seen_refs.update(refs)
            family_sets[partition] = {
                value.metadata.generator_family for value in values
            }
            topology_sets[partition] = {
                (
                    value.metadata.support_structure_signature,
                    value.metadata.challenge_structure_signature,
                )
                for value in values
            }
        if len(family_sets["train"]) < 12:
            raise ValueError("V6 training requires at least twelve topology families")
        comparisons = [("train", "development")]
        if self._final is not None:
            comparisons.extend((("train", "final"), ("development", "final")))
        for left, right in comparisons:
            if family_sets[left] & family_sets[right]:
                raise ValueError("generator families must be partition-disjoint")
            if topology_sets[left] & topology_sets[right]:
                raise ValueError("topology signatures must be partition-disjoint")

    def partition(self, name: Partition) -> tuple[CorpusMechanism, ...]:
        if name not in ("train", "development", "final"):
            raise ValueError("partition is invalid")
        if name == "final":
            return self.final
        return getattr(self, name)

    @property
    def final(self) -> tuple[CorpusMechanism, ...]:
        if self._final is None:
            raise FinalPartitionSealedError(
                "final remains sealed until the development gate explicitly authorizes it"
            )
        return self._final

    @property
    def final_is_open(self) -> bool:
        return self._final is not None

    @property
    def all_mechanisms(self) -> tuple[CorpusMechanism, ...]:
        if self._final is None:
            return (*self.train, *self.development)
        return (*self.train, *self.development, *self._final)

    def learner_payload_size_bytes(self) -> int:
        payload = [value.to_learner_payload() for value in self.all_mechanisms]
        return len(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )


@dataclass(frozen=True, slots=True)
class _FamilySpec:
    key: str
    partition: Partition
    support_steps: tuple[str, ...]
    challenge_steps: tuple[str, ...]
    task_focus: str
    heldout_variant: str
    heldout_description: str
    target_position_offset: int

    def __post_init__(self) -> None:
        if self.support_steps == self.challenge_steps:
            raise ValueError("family challenge must alter procedure structure")
        if not 0 <= self.target_position_offset < CANDIDATES_PER_CHALLENGE:
            raise ValueError("target position offset is invalid")

    @property
    def support_signature(self) -> str:
        return _structure_signature(self.support_steps)

    @property
    def challenge_signature(self) -> str:
        return _structure_signature(self.challenge_steps)


_FAMILIES: tuple[_FamilySpec, ...] = (
    _FamilySpec("inspect_edit_test", "train", ("inspect", "modify", "validate"), ("inspect", "checkpoint", "modify", "validate"), "a bounded source edit", "checkpoint-insertion", "a reversible checkpoint is required before modification", 2),
    _FamilySpec("snapshot_migrate_verify", "train", ("checkpoint", "migrate", "validate"), ("checkpoint", "migrate", "rollback_probe", "validate"), "a representation migration", "rollback-probe-insertion", "the migration must survive a rollback probe before validation", 0),
    _FamilySpec("parse_transform_emit", "train", ("parse", "transform", "emit", "validate"), ("parse", "schema_check", "transform", "emit", "validate"), "a generated artifact transformation", "schema-gate-insertion", "a schema gate now separates parsing from transformation", 3),
    _FamilySpec("invalidate_rebuild_compare", "train", ("invalidate", "rebuild", "compare"), ("invalidate", "rebuild", "observe", "compare"), "a stale derived-state rebuild", "observation-barrier", "the rebuilt state must be observed before comparison", 1),
    _FamilySpec("isolate_patch_merge", "train", ("isolate", "modify", "reconcile", "integration_test"), ("isolate", "refresh_dependency", "modify", "reconcile", "integration_test"), "an isolated dependency patch", "dependency-refresh-insertion", "the isolated patch now depends on a refreshed projection", 1),
    _FamilySpec("paired_branch_reconcile", "train", ("inspect", "branch_left", "branch_right", "reconcile", "validate"), ("checkpoint", "inspect", "branch_left", "branch_right", "reconcile", "validate"), "a two-branch reconciliation", "checkpointed-diamond", "the two-branch merge now starts from a common checkpoint", 3),
    _FamilySpec("stage_validate_activate", "train", ("stage", "validate", "activate"), ("stage", "unit_test", "integration_test", "activate"), "a staged activation", "split-validation", "the former check is split into unit and integration gates", 0),
    _FamilySpec("inventory_allocate_execute", "train", ("inventory", "allocate", "execute", "receipt"), ("inventory", "capacity_check", "allocate", "execute", "receipt"), "a resource-bounded execution", "capacity-gate-insertion", "capacity must be checked before allocation", 2),
    _FamilySpec("reproduce_diagnose_repair", "train", ("reproduce", "diagnose", "modify", "regression_test"), ("reproduce", "counterexample", "diagnose", "modify", "regression_test"), "a reproducible defect repair", "counterexample-insertion", "a fresh counterexample must be captured before diagnosis", 3),
    _FamilySpec("fetch_normalize_index", "train", ("fetch", "validate_input", "normalize", "index"), ("fetch", "quarantine", "validate_input", "normalize", "index"), "an indexed data projection", "quarantine-gate", "retrieved input must pass through quarantine before validation", 1),
    _FamilySpec("lock_update_release", "train", ("lock", "modify", "unlock", "validate"), ("lock", "contention_probe", "modify", "unlock", "validate"), "a synchronized state update", "contention-probe-insertion", "the lock must be tested for contention before mutation", 0),
    _FamilySpec("backup_convert_restore", "train", ("backup", "convert", "restore_probe", "validate"), ("backup", "compatibility_scan", "convert", "restore_probe", "validate"), "a reversible format conversion", "compatibility-scan-insertion", "compatibility must be scanned before conversion", 2),
    _FamilySpec("discover_plan_apply", "train", ("discover", "plan", "apply", "audit"), ("discover", "plan", "dry_run", "apply", "audit"), "a planned workspace change", "dry-run-insertion", "the plan must complete a dry run before application", 2),
    _FamilySpec("classify_route_confirm", "train", ("classify", "route", "process", "confirm"), ("classify", "fallback_probe", "route", "process", "confirm"), "a routed event transformation", "fallback-probe-insertion", "fallback behavior must be probed before routing", 0),
    _FamilySpec("snapshot_prune_compact", "train", ("checkpoint", "prune", "compact", "replay"), ("checkpoint", "integrity_scan", "prune", "compact", "replay"), "a replayable state compaction", "integrity-scan-insertion", "state integrity must be scanned before pruning", 1),
    _FamilySpec("version_build_publish", "train", ("inspect", "version", "build", "publish"), ("inspect", "version", "provenance_check", "build", "publish"), "a versioned artifact publication", "provenance-gate-insertion", "provenance must be checked before the build", 3),
    _FamilySpec("shard_quorum_publish", "development", ("inspect", "stage_left", "stage_right", "quorum", "publish"), ("inspect", "stage_left", "stage_right", "quorum", "audit", "publish"), "a sharded quorum publication", "post-quorum-audit", "the quorum must be audited before publication", 0),
    _FamilySpec("expand_backfill_switch", "development", ("schema_expand", "backfill", "shadow_check", "switch"), ("checkpoint", "schema_expand", "backfill", "shadow_check", "switch"), "an online schema expansion", "checkpointed-backfill", "the backfill now begins from a reversible checkpoint", 2),
    _FamilySpec("lease_rotate_revoke", "development", ("inspect", "lease_new", "probe", "lease_old_revoke"), ("inspect", "lease_new", "dual_read", "probe", "lease_old_revoke"), "a synthetic lease rotation", "dual-read-transition", "old and new leases must overlap for one read-only transition", 0),
    _FamilySpec("queue_drain_refill", "development", ("drain", "reconfigure", "refill", "validate"), ("drain", "reconfigure", "replay_probe", "refill", "validate"), "a queue reconfiguration", "pre-refill-replay-probe", "a replay probe must succeed before the queue is refilled", 2),
    _FamilySpec("ledger_rewrite_seal", "final", ("checkpoint", "rewrite", "reconcile", "seal"), ("checkpoint", "rewrite", "reconcile", "independent_audit", "seal"), "a synthetic ledger rewrite", "independent-pre-seal-audit", "an independent audit is required before sealing", 1),
    _FamilySpec("region_failover_activate", "final", ("stage_left", "stage_right", "compare", "failover", "activate"), ("stage_left", "stage_right", "compare", "health_window", "failover", "activate"), "a multi-region activation", "health-window-insertion", "a stable health window is required before failover", 3),
    _FamilySpec("protocol_roundtrip_commit", "final", ("negotiate", "translate", "roundtrip", "commit"), ("negotiate", "compatibility_matrix", "translate", "roundtrip", "commit"), "a protocol translation", "compatibility-matrix-insertion", "a compatibility matrix must be checked before translation", 1),
    _FamilySpec("plugin_sandbox_activate", "final", ("discover", "sandbox", "validate", "activate"), ("discover", "permission_audit", "sandbox", "validate", "activate"), "a sandboxed extension activation", "permission-audit-insertion", "declared permissions must be audited before sandbox execution", 3),
)


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
    "accept the reordered public contract", "handle the new optional branch",
    "separate preparation from activation", "retain compatibility during a format transition",
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


_ROLE_SENTENCES = {
    "inspect": "Inspect the current artifact and record its dependency boundary.",
    "modify": "Apply the requested bounded change to the artifact.",
    "validate": "Run the declared acceptance check against the completed workspace.",
    "checkpoint": "Create a reversible checkpoint before changing workspace state.",
    "migrate": "Migrate the artifact to the revised representation.",
    "rollback_probe": "Exercise the rollback path and restore the candidate state.",
    "parse": "Parse the authoritative input into a canonical intermediate form.",
    "schema_check": "Check the intermediate form against the declared schema.",
    "transform": "Transform the canonical intermediate form into the requested shape.",
    "emit": "Emit the transformed artifact into the isolated output area.",
    "invalidate": "Invalidate the stale derived state before replacement.",
    "rebuild": "Rebuild derived state from the revised authoritative inputs.",
    "observe": "Observe and record the externally visible state transition.",
    "compare": "Compare the candidate state with the declared acceptance boundary.",
    "isolate": "Create an isolated work area for the dependency-sensitive change.",
    "refresh_dependency": "Refresh the dependency projection from its authoritative source.",
    "reconcile": "Reconcile the changed parts into one consistent workspace state.",
    "integration_test": "Run the integration check across the complete dependency path.",
    "branch_left": "Apply the first independent branch of the requested change.",
    "branch_right": "Apply the second independent branch of the requested change.",
    "stage": "Stage the candidate artifact without activating it.",
    "unit_test": "Run the focused unit checks for the changed boundary.",
    "activate": "Activate the already verified candidate state.",
    "inventory": "Inventory the resources and limits declared for this execution.",
    "capacity_check": "Check that declared capacity can contain the planned operation.",
    "allocate": "Allocate only the resources admitted by the inventory.",
    "execute": "Execute the bounded operation in the prepared workspace.",
    "receipt": "Record an objective execution receipt.",
    "reproduce": "Reproduce the reported defect in the isolated workspace.",
    "counterexample": "Capture a fresh counterexample that exercises the same boundary.",
    "diagnose": "Diagnose the observable boundary that failed.",
    "regression_test": "Run the regression check over the repaired behavior.",
    "fetch": "Fetch the declared synthetic input into the bounded workspace.",
    "quarantine": "Place the retrieved input in a non-authoritative quarantine area.",
    "validate_input": "Validate the quarantined input before using it.",
    "normalize": "Normalize the validated input into canonical form.",
    "index": "Index the canonical form for later retrieval.",
    "lock": "Acquire the declared synchronization boundary.",
    "contention_probe": "Probe whether another operation currently holds conflicting state.",
    "unlock": "Release the synchronization boundary after mutation.",
    "backup": "Create a bounded backup of the artifact to be converted.",
    "compatibility_scan": "Scan the input for compatibility with the destination format.",
    "convert": "Convert the artifact into the destination format.",
    "restore_probe": "Test restoration from the bounded backup.",
    "discover": "Discover the declared workspace facts needed for planning.",
    "plan": "Construct a bounded change plan from the observed facts.",
    "dry_run": "Execute the plan without committing workspace changes.",
    "apply": "Apply the already checked plan to the workspace.",
    "audit": "Audit the candidate result against the declared request.",
    "classify": "Classify the incoming synthetic event by its public contract.",
    "fallback_probe": "Probe the fallback path without committing an event.",
    "route": "Route the event to the matching declared handler.",
    "process": "Process the routed event within the bounded handler.",
    "confirm": "Confirm the resulting event state with the declared check.",
    "integrity_scan": "Scan checkpoint integrity before removing derived material.",
    "prune": "Prune derived entries no longer referenced by accepted state.",
    "compact": "Compact the retained state into one canonical projection.",
    "replay": "Replay the compacted projection through the acceptance check.",
    "version": "Assign the declared next version to the candidate artifact.",
    "provenance_check": "Check source and dependency provenance before building.",
    "build": "Build the versioned candidate in the isolated workspace.",
    "publish": "Publish the already verified synthetic candidate.",
    "stage_left": "Stage the first independent shard.",
    "stage_right": "Stage the second independent shard.",
    "quorum": "Confirm that the staged shards agree on one state.",
    "schema_expand": "Expand the schema without removing the prior representation.",
    "backfill": "Backfill the new representation from accepted records.",
    "shadow_check": "Compare the old and new representations under shadow reads.",
    "switch": "Switch reads to the verified new representation.",
    "lease_new": "Create the replacement synthetic lease without activating it.",
    "dual_read": "Read through both lease paths without mutating state.",
    "probe": "Probe the replacement path with a bounded synthetic request.",
    "lease_old_revoke": "Revoke the obsolete synthetic lease after verification.",
    "drain": "Drain queued synthetic work without accepting new items.",
    "reconfigure": "Apply the queue configuration change while drained.",
    "replay_probe": "Replay one bounded item through the reconfigured queue.",
    "refill": "Resume acceptance of queued work after the probe.",
    "rewrite": "Rewrite accepted records into the revised canonical layout.",
    "independent_audit": "Run an independent consistency audit before sealing.",
    "seal": "Seal the verified synthetic ledger state.",
    "health_window": "Observe a stable bounded health window before failover.",
    "failover": "Move synthetic traffic to the staged region.",
    "negotiate": "Negotiate the declared protocol capabilities.",
    "compatibility_matrix": "Check the capability pair against the compatibility matrix.",
    "translate": "Translate the request into the negotiated representation.",
    "roundtrip": "Round-trip the translation through the synthetic peer.",
    "commit": "Commit the verified protocol representation.",
    "permission_audit": "Audit declared extension permissions before execution.",
    "sandbox": "Run the extension in a bounded synthetic sandbox.",
    "blind_change": "Change the artifact immediately without inspecting its boundary.",
}


def _rng(*parts: object) -> random.Random:
    payload = "\x1f".join(str(value) for value in parts).encode("utf-8")
    digest = hashlib.sha256(_GENERATION_SALT + payload).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _structure_signature(steps: tuple[str, ...]) -> str:
    edges = tuple(f"{left}>{right}" for left, right in zip(steps, steps[1:]))
    return f"nodes:{','.join(steps)}|edges:{','.join(edges)}"


def _context(family: _FamilySpec, mechanism_index: int, surface_index: int) -> dict[str, str]:
    rng = _rng(family.key, mechanism_index, surface_index, "surface")
    return {
        "system": _SYSTEMS[rng.randrange(len(_SYSTEMS))],
        "artifact": _ARTIFACTS[rng.randrange(len(_ARTIFACTS))],
        "change": _CHANGES[(mechanism_index + surface_index) % len(_CHANGES)],
        "constraint": _CONSTRAINTS[(2 * mechanism_index + surface_index) % len(_CONSTRAINTS)],
        "check": _CHECKS[(3 * mechanism_index + surface_index) % len(_CHECKS)],
    }


def _task_text(
    context: dict[str, str],
    family: _FamilySpec,
    *,
    heldout: bool,
) -> str:
    text = (
        f"The synthetic {context['system']} workspace requires {family.task_focus} around "
        f"its {context['artifact']}. It must {context['change']} while preserving "
        f"{context['constraint']}."
    )
    if heldout:
        text += f" In this fresh structural variant, {family.heldout_description}."
    return text


def _request_text(context: dict[str, str]) -> str:
    return (
        f"Complete the requested work so that {context['check']} passes without weakening "
        f"{context['constraint']}."
    )


def _step_text(role: str, context: dict[str, str]) -> str:
    if role == "modify":
        return f"Update the {context['artifact']} to {context['change']}."
    try:
        return _ROLE_SENTENCES[role]
    except KeyError as exc:
        raise ValueError(f"unknown synthetic procedure role: {role}") from exc


def _trace_text(steps: tuple[str, ...], context: dict[str, str]) -> str:
    return " ".join(
        f"Step {index + 1}: {_step_text(role, context)}"
        for index, role in enumerate(steps)
    )


def _failed_steps(steps: tuple[str, ...], mode: str) -> tuple[str, ...]:
    if mode == "reordered":
        values = list(steps)
        left = 1 if len(values) > 3 else 0
        values[left], values[left + 1] = values[left + 1], values[left]
        return tuple(values)
    if mode == "omitted":
        return steps[:-1]
    if mode == "bypass":
        return ("blind_change", *steps[1:])
    raise ValueError("unknown failure mode")


def _diagnostic(context: dict[str, str], outcome: OutcomeLabel, mode: str) -> str:
    if outcome == "SUCCESS":
        return (
            f"Objective verification passed: {context['check']} accepted the result and "
            f"{context['constraint']} remained intact."
        )
    return {
        "reordered": (
            f"Objective verification failed: a dependent operation observed stale state, "
            f"and {context['check']} rejected the workspace."
        ),
        "omitted": (
            "Objective verification failed: completion evidence was absent, so the "
            "workspace could not be accepted."
        ),
        "bypass": (
            f"Objective verification failed: the trace bypassed a declared boundary and "
            f"did not preserve {context['constraint']}."
        ),
    }[mode]


def _build_mechanism(
    family: _FamilySpec,
    partition_index: int,
    family_local_index: int,
    *,
    ordinal_start: int,
) -> CorpusMechanism:
    outcomes = _OUTCOME_PATTERNS[partition_index % len(_OUTCOME_PATTERNS)]
    failure_modes = ("reordered", "omitted", "bypass")
    failure_cursor = 0
    episodes = []
    for episode_index, outcome in enumerate(outcomes):
        context = _context(family, partition_index, episode_index)
        if outcome == "SUCCESS":
            steps = family.support_steps
            mode = "reordered"
        else:
            mode = failure_modes[failure_cursor]
            failure_cursor += 1
            steps = _failed_steps(family.support_steps, mode)
        episodes.append(
            OutcomeEpisode(
                task_text=_task_text(context, family, heldout=False),
                request_text=_request_text(context),
                action_trace_text=_trace_text(steps, context),
                outcome_label=outcome,
                objective_diagnostic_text=_diagnostic(context, outcome, mode),
                temporal=MovingOriginCoordinates(
                    acquired_ordinal=ordinal_start + episode_index,
                    age=EPISODES_PER_MECHANISM - episode_index - 1,
                    landmark_relations=(("stream_start", "AT" if episode_index == 0 else "AFTER"),),
                ),
            )
        )

    context = _context(family, partition_index, EPISODES_PER_MECHANISM + 1)
    target_trace = _trace_text(family.challenge_steps, context)
    failed_traces = [
        _trace_text(_failed_steps(family.challenge_steps, mode), context)
        for mode in failure_modes
    ]
    _rng(family.key, partition_index, "failed-candidate-order").shuffle(failed_traces)
    target_position = (
        family.target_position_offset + family_local_index
    ) % CANDIDATES_PER_CHALLENGE
    candidates = list(failed_traces)
    candidates.insert(target_position, target_trace)
    mechanism_ref = (
        f"{family.partition}-v6-mechanism-{family.key}-{family_local_index:02d}"
    )
    return CorpusMechanism(
        public=PublicApprenticeshipStream(
            episodes=tuple(episodes),
            challenge=ApprenticeshipChallenge(
                task_text=_task_text(context, family, heldout=True),
                request_text=_request_text(context),
                candidate_action_trace_texts=tuple(candidates),
            ),
        ),
        supervision=MechanismSupervision(
            target_successful_candidate_indices=(target_position,),
            relevant_failed_candidate_indices=tuple(
                index for index in range(CANDIDATES_PER_CHALLENGE) if index != target_position
            ),
        ),
        metadata=GeneratorMetadata(
            partition=family.partition,
            mechanism_ref=mechanism_ref,
            generator_family=family.key,
            support_structure_signature=family.support_signature,
            challenge_structure_signature=family.challenge_signature,
            heldout_variant=family.heldout_variant,
        ),
    )


def build_causal_neuromodulated_apprenticeship_v6(
    *,
    include_final: bool = False,
) -> CausalNeuromodulatedApprenticeshipCorpus:
    """Build V6 without final unless a caller explicitly opens that gate."""

    if type(include_final) is not bool:
        raise TypeError("include_final must be bool")

    by_partition: dict[str, list[CorpusMechanism]] = {
        "train": [],
        "development": [],
        "final": [],
    }
    counts_per_family = {"train": 4, "development": 2, "final": 2}
    partition_indices = {"train": 0, "development": 0, "final": 0}
    ordinal = 0
    for family in _FAMILIES:
        if family.partition == "final" and not include_final:
            continue
        for family_local_index in range(counts_per_family[family.partition]):
            partition_index = partition_indices[family.partition]
            partition_indices[family.partition] += 1
            mechanism = _build_mechanism(
                family,
                partition_index,
                family_local_index,
                ordinal_start=ordinal,
            )
            by_partition[family.partition].append(mechanism)
            ordinal += EPISODES_PER_MECHANISM
    return CausalNeuromodulatedApprenticeshipCorpus(
        train=tuple(by_partition["train"]),
        development=tuple(by_partition["development"]),
        _final=tuple(by_partition["final"]) if include_final else None,
    )


__all__ = [
    "CANDIDATES_PER_CHALLENGE",
    "CORPUS_ID",
    "CausalNeuromodulatedApprenticeshipCorpus",
    "DEVELOPMENT_MECHANISMS",
    "EPISODES_PER_MECHANISM",
    "FINAL_MECHANISMS",
    "FinalPartitionSealedError",
    "TRAIN_GENERATOR_FAMILIES",
    "TRAIN_MECHANISMS",
    "build_causal_neuromodulated_apprenticeship_v6",
]
