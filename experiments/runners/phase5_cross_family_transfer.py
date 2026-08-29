"""Frozen-core transfer probe from latent programs to precedence graphs.

This runner starts from a trained Phase-5 reversible policy, acquires the
original latent-order skills into one bounded online state, then presents a
second public evidence family without replaying the first.  Each online write
uses one sampled permutation and one scalar outcome.  Disjoint queries are
scored without writes, paired with reset-state and reversible-core-off causal
controls.

The experiment is intentionally a cheap falsification before adding a learned
family-neutral fact encoder.  The precedence graph is losslessly packed into
the existing 14 public item fields; no solver or target rank is computed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from angler.procedures.skill_memory import (
    ProceduralSkillState,
    procedural_skill_state_digest,
)
from angler.reasoning.recurrent_core import reasoning_state_digest
from experiments.evaluators.relational_procedure_transfer_suite import (
    GeneratedRelationalProcedureTask,
    PublicRelationalProcedureTask,
    make_relational_procedure_transfer_stream,
    score_relational_procedure_answer,
)
from experiments.evaluators.skill_memory_suite import (
    GeneratedSkillMemoryTask,
    make_skill_memory_composition_curriculum,
    score_skill_memory_answer,
)
from experiments.runners import phase5_skill_memory_stream as phase5


_REPORT_VERSION = "angler.phase5-cross-family-transfer.v3"


class SharedPublicFactAdapter(nn.Module):
    """One learned permutation-equivariant encoder for public relation facts.

    The adapter binds to the public precedence-edge schema, not to an opaque
    family, task, skill, or evaluator identity.  It receives only the existing
    five public 14-feature rows.  There is no answer head, target, or graph
    algorithm.  A zero output projection makes attachment an exact identity
    before scalar-outcome training.
    """

    def __init__(self, feature_width: int = 14, hidden_width: int = 32) -> None:
        super().__init__()
        if feature_width <= 0 or hidden_width <= 0 or hidden_width % 4:
            raise ValueError("fact-adapter widths must be positive and head-compatible")
        self.feature_width = feature_width
        self.input_norm = nn.LayerNorm(feature_width, elementwise_affine=False)
        self.input_projection = nn.Linear(feature_width, hidden_width)
        self.attention_norm = nn.LayerNorm(hidden_width)
        self.attention = nn.MultiheadAttention(
            hidden_width,
            num_heads=4,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(hidden_width)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_width, 2 * hidden_width),
            nn.SiLU(),
            nn.Linear(2 * hidden_width, hidden_width),
        )
        self.output_norm = nn.LayerNorm(hidden_width)
        self.output_projection = nn.Linear(
            hidden_width,
            feature_width,
            bias=False,
        )
        nn.init.zeros_(self.output_projection.weight)

    @staticmethod
    def applies_to(public_task: Any) -> bool:
        return isinstance(public_task, PublicRelationalProcedureTask)

    def forward(self, public_features: torch.Tensor) -> torch.Tensor:
        if public_features.shape != (5, self.feature_width):
            raise ValueError("fact adapter requires five public feature rows")
        hidden = F.silu(self.input_projection(self.input_norm(public_features)))
        normalized = self.attention_norm(hidden).unsqueeze(0)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        hidden = hidden + attended.squeeze(0)
        hidden = hidden + self.feed_forward(self.feed_forward_norm(hidden))
        delta = self.output_projection(self.output_norm(hidden))
        return public_features + delta


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("a score summary requires at least one value")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "mean": mean,
        "minimum": min(values),
        "maximum": max(values),
        "population_standard_deviation": math.sqrt(variance),
    }


def _state_element_count(state: ProceduralSkillState) -> int:
    return sum(
        value.numel()
        for value in (
            state.fast_weights,
            state.slot_latents,
            state.key_offsets,
            state.occupied,
            state.write_counts,
        )
    )


def _frozen_core_fingerprint(policy: phase5.SkillMemoryPolicy) -> str:
    return phase5._named_state_fingerprint(
        policy,
        include=lambda name: not name.startswith("public_fact_adapter."),
        domain=b"project-angler.cross-family-frozen-core.v1",
    )


def _frozen_core_parameter_identity(policy: phase5.SkillMemoryPolicy) -> str:
    digest = hashlib.sha256(b"project-angler.cross-family-core-identity.v1\x00")
    selected = 0
    for name, parameter in policy.named_parameters():
        if name.startswith("public_fact_adapter."):
            continue
        encoded = (
            f"{name}\x00{id(parameter)}\x00{parameter.data_ptr()}\x00"
            f"{tuple(parameter.shape)}\x00{parameter.dtype}\x00{parameter.device}"
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        selected += 1
    if not selected:
        raise RuntimeError("frozen core identity selected no parameters")
    return "sha256:" + digest.hexdigest()


def _train_shared_public_fact_adapter(
    policy: phase5.SkillMemoryPolicy,
    *,
    seed: int,
    meta_steps: int,
    supports_per_procedure: int,
    queries_per_procedure: int,
    learning_rate: float,
) -> dict[str, Any]:
    """Align public fact structures using scalar deployed outcomes only."""

    if isinstance(meta_steps, bool) or not isinstance(meta_steps, int) or meta_steps <= 0:
        raise ValueError("adapter meta_steps must be a positive integer")
    adapter = getattr(policy, "public_fact_adapter", None)
    if not isinstance(adapter, SharedPublicFactAdapter):
        raise RuntimeError("shared public fact adapter is not attached")
    for name, parameter in policy.named_parameters():
        parameter.requires_grad_(name.startswith("public_fact_adapter."))
    trainable = tuple(parameter for parameter in policy.parameters() if parameter.requires_grad)
    if set(name for name, parameter in policy.named_parameters() if parameter.requires_grad) != {
        name for name, _ in adapter.named_parameters(prefix="public_fact_adapter")
    }:
        raise RuntimeError("fact adaptation exposed an undeclared trainable parameter")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        weight_decay=0.0,
    )
    core_before = _frozen_core_fingerprint(policy)
    core_identity_before = _frozen_core_parameter_identity(policy)
    adapter_before = phase5._named_state_fingerprint(
        policy,
        include=lambda name: name.startswith("public_fact_adapter."),
        domain=b"project-angler.cross-family-public-fact-adapter.v1",
    )
    losses: list[float] = []
    gradient_norms: list[float] = []
    nonzero_reward_steps = 0
    adapter.train()
    policy.stable_compiler.eval()
    for step in range(meta_steps):
        episode_seed = seed + 100_003 * (step + 1)
        phase5._seed_reproducible_stage(
            episode_seed,
            "cross-family-fact-adapter",
            next(policy.parameters()).device,
        )
        stream = make_relational_procedure_transfer_stream(
            episode_seed,
            supports_per_procedure=supports_per_procedure,
            queries_per_procedure=queries_per_procedure,
        )
        state = policy.initial_state(1)
        for pair in stream.supports:
            proposal = phase5.propose_task(
                policy,
                pair.learner,
                state,
                greedy=False,
                temperature=1.25,
            )
            scalar = score_relational_procedure_answer(
                pair.learner,
                pair.hidden,
                proposal.answer,
            )
            staged = phase5.propose_differentiable_feedback(
                policy,
                proposal,
                scalar,
                state,
            )
            state = staged.candidate_state

        query_losses: list[torch.Tensor] = []
        informative = False
        for query_index, pair in enumerate(stream.queries):
            scores = policy.score_task(pair.learner, state)
            candidates = phase5._on_policy_reward_candidate_set(
                scores.logits,
                step,
                query_index,
            )
            scalar_scores = phase5._scalar_attempt_scores(
                pair,
                candidates,
                score_relational_procedure_answer,
            )
            informative = informative or len(set(scalar_scores)) > 1
            query_losses.append(
                phase5._scalar_on_policy_reward_loss(
                    scores.logits,
                    candidates,
                    scalar_scores,
                )
            )
        loss = torch.stack(query_losses).mean()
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("fact-adapter training produced non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not any(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all().item())
            and bool(parameter.grad.detach().count_nonzero())
            for parameter in trainable
        ):
            raise RuntimeError("scalar outcomes did not reach the public fact adapter")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            5.0,
            error_if_nonfinite=True,
        )
        optimizer.step()
        losses.append(float(loss.detach().item()))
        gradient_norms.append(float(gradient_norm.detach().item()))
        nonzero_reward_steps += int(informative)
    adapter.eval()
    policy.requires_grad_(False)
    adapter_after = phase5._named_state_fingerprint(
        policy,
        include=lambda name: name.startswith("public_fact_adapter."),
        domain=b"project-angler.cross-family-public-fact-adapter.v1",
    )
    if _frozen_core_fingerprint(policy) != core_before:
        raise RuntimeError("fact-adapter training changed the frozen Angler core")
    if _frozen_core_parameter_identity(policy) != core_identity_before:
        raise RuntimeError("fact-adapter training replaced a frozen core parameter")
    if adapter_after == adapter_before:
        raise RuntimeError("fact-adapter training left its declared seam unchanged")
    return {
        "meta_steps": meta_steps,
        "fresh_opaque_mappings": meta_steps,
        "supports_per_procedure_per_mapping": supports_per_procedure,
        "queries_per_procedure_per_mapping": queries_per_procedure,
        "attempted_outputs_per_query": 4,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "minimum_loss": min(losses),
        "mean_gradient_norm": sum(gradient_norms) / len(gradient_norms),
        "informative_reward_steps": nonzero_reward_steps,
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "adapter_fingerprint_before": adapter_before,
        "adapter_fingerprint_after": adapter_after,
        "frozen_core_fingerprint_before": core_before,
        "frozen_core_fingerprint_after": _frozen_core_fingerprint(policy),
        "training_signal": (
            "four attempted public permutations plus centered scalar outcomes"
        ),
        "target_order_used": False,
        "deterministic_solver_used": False,
        "old_family_replay_used": False,
    }


def _candidate_answer(
    policy: phase5.SkillMemoryPolicy,
    public_task: Any,
    state: ProceduralSkillState,
    *,
    include_reversible_transition: bool = True,
) -> tuple[str, ...]:
    scores = policy.score_task(
        public_task,
        state,
        include_reversible_transition=include_reversible_transition,
    )
    candidate_index = int(scores.logits.argmax(dim=-1).item())
    permutation = phase5._PERMUTATIONS[candidate_index]
    return tuple(public_task.items[index].symbol for index in permutation)


def _score_pairs(
    policy: phase5.SkillMemoryPolicy,
    state: ProceduralSkillState,
    pairs: Sequence[Any],
    judge: Callable[[Any, Any, tuple[str, ...]], float],
    *,
    include_reversible_transition: bool = True,
) -> list[float]:
    state_before = procedural_skill_state_digest(state)
    values: list[float] = []
    with torch.no_grad():
        for pair in pairs:
            answer = _candidate_answer(
                policy,
                pair.learner,
                state,
                include_reversible_transition=include_reversible_transition,
            )
            values.append(float(judge(pair.learner, pair.hidden, answer)))
    if procedural_skill_state_digest(state) != state_before:
        raise RuntimeError("a no-feedback query changed the competence state")
    return values


def _acquire_pairs(
    policy: phase5.SkillMemoryPolicy,
    state: ProceduralSkillState,
    pairs: Sequence[Any],
    judge: Callable[[Any, Any, tuple[str, ...]], float],
) -> tuple[ProceduralSkillState, dict[str, Any]]:
    scores: list[float] = []
    accepted = 0
    core_accepted = 0
    delta_norms: list[float] = []
    incoming_elements = _state_element_count(state)
    for pair in pairs:
        proposal = phase5.propose_task(
            policy,
            pair.learner,
            state,
            greedy=False,
            temperature=1.25,
        )
        scalar = float(judge(pair.learner, pair.hidden, proposal.answer))
        feedback = phase5.apply_transactional_feedback(
            policy,
            pair.learner,
            proposal,
            scalar,
            state,
        )
        state = feedback.state
        scores.append(scalar)
        accepted += int(feedback.accepted)
        core_accepted += int(feedback.core_accepted)
        delta_norms.append(feedback.delta_norm)
        if _state_element_count(state) != incoming_elements:
            raise RuntimeError("online acquisition changed fixed state capacity")
    window = max(1, len(scores) // 4)
    return state, {
        "presentations": len(scores),
        "accepted_transactions": accepted,
        "core_accepted_transactions": core_accepted,
        "first_quarter_mean": sum(scores[:window]) / window,
        "last_quarter_mean": sum(scores[-window:]) / window,
        "trajectory_gain": (
            sum(scores[-window:]) / window - sum(scores[:window]) / window
        ),
        "mean_delta_norm": sum(delta_norms) / len(delta_norms),
        "state_element_count": incoming_elements,
    }


def _split_relational(
    pairs: Sequence[GeneratedRelationalProcedureTask],
) -> tuple[tuple[GeneratedRelationalProcedureTask, ...], tuple[GeneratedRelationalProcedureTask, ...]]:
    forward = tuple(pair for pair in pairs if not pair.hidden.reverse)
    reverse = tuple(pair for pair in pairs if pair.hidden.reverse)
    if not forward or not reverse:
        raise RuntimeError("relational stream is missing one declared procedure")
    return forward, reverse


def run(
    *,
    seed: int = 92_001,
    device: str | torch.device = "cpu",
    initial_checkpoint: str | Path,
    compiler_checkpoint: str | Path = phase5._PHASE4_CHECKPOINT,
    old_encounters_per_primitive: int = 8,
    old_query_cases: int = 8,
    relational_supports_per_procedure: int = 64,
    relational_queries_per_procedure: int = 40,
    adapter_meta_steps: int = 64,
    adapter_supports_per_procedure: int = 8,
    adapter_queries_per_procedure: int = 8,
    adapter_learning_rate: float = 8.0e-4,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Run one persistent-state, no-replay transfer experiment."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    target_device = torch.device(device)
    random.seed(seed)
    torch.manual_seed(seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    settings = phase5._PROFILES["composition"]
    compiler, compiler_record = phase5._load_phase4_compiler(compiler_checkpoint)
    policy = phase5.SkillMemoryPolicy(settings, compiler).to(
        device=target_device,
        dtype=torch.float32,
    )
    initialization = phase5._load_initial_policy_checkpoint(
        policy,
        initial_checkpoint,
        settings,
    )
    if not bool(policy.reversible_transition_mode.item()):
        raise RuntimeError("cross-family transfer requires an active reversible core")
    if initialization.get("source_stage") != "reversible_transition_acquisition":
        raise RuntimeError("initial checkpoint is not a reversible-acquisition result")
    policy.eval()
    policy.requires_grad_(False)
    retained_policy_fingerprint = reasoning_state_digest(policy)
    policy.public_fact_adapter = SharedPublicFactAdapter().to(
        device=target_device,
        dtype=torch.float32,
    )
    frozen_core_before = _frozen_core_fingerprint(policy)
    frozen_core_identity_before = _frozen_core_parameter_identity(policy)

    old_curriculum = make_skill_memory_composition_curriculum(
        seed + 1_000_003,
        encounters_per_primitive=old_encounters_per_primitive,
        cases_per_component_probe=old_query_cases,
        cases_per_composition=old_query_cases,
    )
    relational = make_relational_procedure_transfer_stream(
        seed + 2_000_003,
        supports_per_procedure=relational_supports_per_procedure,
        queries_per_procedure=relational_queries_per_procedure,
    )

    # The zero-initialized shared adapter is an exact identity.  This retained
    # family audit is discarded before training and is never used as a loss.
    phase5._seed_reproducible_stage(
        seed + 4_000_003,
        "cross-family-old-retention-audit",
        target_device,
    )
    pre_adapter_state, _ = _acquire_pairs(
        policy,
        policy.initial_state(1),
        old_curriculum.component_supports,
        score_skill_memory_answer,
    )
    old_pre_adapter = _summary(
        _score_pairs(
            policy,
            pre_adapter_state,
            old_curriculum.composition_queries,
            score_skill_memory_answer,
        )
    )

    adapter_training = _train_shared_public_fact_adapter(
        policy,
        seed=seed + 3_000_003,
        meta_steps=adapter_meta_steps,
        supports_per_procedure=adapter_supports_per_procedure,
        queries_per_procedure=adapter_queries_per_procedure,
        learning_rate=adapter_learning_rate,
    )
    policy.eval()

    phase5._seed_reproducible_stage(
        seed + 4_000_003,
        "cross-family-old-retention-audit",
        target_device,
    )
    state = policy.initial_state(1)
    initial_state_digest = procedural_skill_state_digest(state)
    initial_elements = _state_element_count(state)
    state, old_acquisition = _acquire_pairs(
        policy,
        state,
        old_curriculum.component_supports,
        score_skill_memory_answer,
    )
    old_state_digest = procedural_skill_state_digest(state)
    old_before_values = _score_pairs(
        policy,
        state,
        old_curriculum.composition_queries,
        score_skill_memory_answer,
    )

    relational_state_before = procedural_skill_state_digest(state)
    state, relational_acquisition = _acquire_pairs(
        policy,
        state,
        relational.supports,
        score_relational_procedure_answer,
    )
    relational_state_after = procedural_skill_state_digest(state)
    final_elements = _state_element_count(state)
    if final_elements != initial_elements:
        raise RuntimeError("cross-family stream changed fixed competence capacity")

    old_after_values = _score_pairs(
        policy,
        state,
        old_curriculum.composition_queries,
        score_skill_memory_answer,
    )
    reset_state = policy.initial_state(1)
    relational_forward, relational_reverse = _split_relational(relational.queries)

    def relational_report(
        pairs: Sequence[GeneratedRelationalProcedureTask],
    ) -> dict[str, Any]:
        full = _score_pairs(
            policy,
            state,
            pairs,
            score_relational_procedure_answer,
        )
        reset = _score_pairs(
            policy,
            reset_state,
            pairs,
            score_relational_procedure_answer,
        )
        removed = _score_pairs(
            policy,
            state,
            pairs,
            score_relational_procedure_answer,
            include_reversible_transition=False,
        )
        full_summary = _summary(full)
        reset_summary = _summary(reset)
        removed_summary = _summary(removed)
        return {
            "full": full_summary,
            "reset_state": reset_summary,
            "reversible_transition_removed": removed_summary,
            "acquired_state_gain": (
                float(full_summary["mean"]) - float(reset_summary["mean"])
            ),
            "reversible_transition_gain": (
                float(full_summary["mean"]) - float(removed_summary["mean"])
            ),
        }

    old_before = _summary(old_before_values)
    old_after = _summary(old_after_values)
    result: dict[str, Any] = {
        "report_version": _REPORT_VERSION,
        "seed": seed,
        "device": str(target_device),
        "initial_checkpoint": {
            "path": str(initial_checkpoint),
            "sha256": initialization["sha256"],
            "source_runner": initialization["source_runner"],
            "source_stage": initialization["source_stage"],
            "source_result_digest": initialization["result_digest"],
        },
        "compiler_checkpoint": compiler_record,
        "typed_public_fact_adapter": {
            "architecture": (
                "one learned permutation-equivariant residual self-attention encoder"
            ),
            "input_schema_binding": "public precedence_edges only",
            "opaque_family_task_or_skill_route": False,
            "output_head": False,
            "training": adapter_training,
        },
        "old_family": {
            "query_before_adapter_training": old_pre_adapter,
            "acquisition": old_acquisition,
            "query_before_new_family": old_before,
            "query_after_new_family": old_after,
            "adapter_training_retention_delta": (
                float(old_before["mean"]) - float(old_pre_adapter["mean"])
            ),
            "retention_delta": float(old_after["mean"]) - float(old_before["mean"]),
            "replayed_during_new_family": False,
        },
        "relational_family": {
            "public_representation": (
                "lossless displayed-node plus immediate-successor edge packing; "
                "no target rank or solver"
            ),
            "acquisition": relational_acquisition,
            "forward_path": relational_report(relational_forward),
            "reverse_path_composition": relational_report(relational_reverse),
            "support_source_count": len(relational.supports),
            "query_source_count": len(relational.queries),
            "support_query_disjoint": True,
        },
        "state": {
            "initial_digest": initial_state_digest,
            "after_old_family_digest": old_state_digest,
            "before_relational_digest": relational_state_before,
            "after_relational_digest": relational_state_after,
            "element_count_before": initial_elements,
            "element_count_after": final_elements,
            "constant_capacity": initial_elements == final_elements,
        },
        "integrity": {
            "retained_policy_fingerprint_before_adapter_attachment": (
                retained_policy_fingerprint
            ),
            "frozen_core_fingerprint_before": frozen_core_before,
            "frozen_core_fingerprint_after": _frozen_core_fingerprint(policy),
            "frozen_core_unchanged": (
                _frozen_core_fingerprint(policy) == frozen_core_before
            ),
            "frozen_core_parameter_identity_before": frozen_core_identity_before,
            "frozen_core_parameter_identity_after": (
                _frozen_core_parameter_identity(policy)
            ),
        },
        "claims": {
            "foundation_or_reversible_core_training": False,
            "typed_public_fact_adapter_training": True,
            "online_history_replay": False,
            "online_feedback_per_support": (
                "one sampled public permutation plus one scalar outcome"
            ),
            "query_feedback_or_writes": False,
            "deterministic_solver": False,
            "hidden_target_used_by_learner": False,
            "broad_cross_domain_transfer_proven": False,
            "purpose": "learned-observation cross-family transfer falsification",
        },
    }
    if not result["integrity"]["frozen_core_unchanged"] or (
        result["integrity"]["frozen_core_parameter_identity_before"]
        != result["integrity"]["frozen_core_parameter_identity_after"]
    ):
        raise RuntimeError("cross-family transfer changed the frozen Angler core")
    result["result_digest"] = "sha256:" + hashlib.sha256(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "runner": _REPORT_VERSION,
                "seed": seed,
                "base_checkpoint_sha256": initialization["sha256"],
                "base_result_digest": initialization["result_digest"],
                "adapter_class": "SharedPublicFactAdapter",
                "adapter_model": policy.public_fact_adapter.state_dict(),
                "adapter_training": adapter_training,
                "frozen_core_fingerprint": _frozen_core_fingerprint(policy),
                "result_digest": result["result_digest"],
            },
            checkpoint_path,
        )
        result["adapter_checkpoint"] = str(checkpoint_path)
        result["adapter_checkpoint_sha256"] = hashlib.sha256(
            checkpoint_path.read_bytes()
        ).hexdigest()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=92_001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument(
        "--compiler-checkpoint",
        default=str(phase5._PHASE4_CHECKPOINT),
    )
    parser.add_argument("--old-encounters-per-primitive", type=int, default=8)
    parser.add_argument("--old-query-cases", type=int, default=8)
    parser.add_argument("--relational-supports-per-procedure", type=int, default=64)
    parser.add_argument("--relational-queries-per-procedure", type=int, default=40)
    parser.add_argument("--adapter-meta-steps", type=int, default=64)
    parser.add_argument("--adapter-supports-per-procedure", type=int, default=8)
    parser.add_argument("--adapter-queries-per-procedure", type=int, default=8)
    parser.add_argument("--adapter-learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--checkpoint")
    parser.add_argument("--result-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        seed=args.seed,
        device=args.device,
        initial_checkpoint=args.initial_checkpoint,
        compiler_checkpoint=args.compiler_checkpoint,
        old_encounters_per_primitive=args.old_encounters_per_primitive,
        old_query_cases=args.old_query_cases,
        relational_supports_per_procedure=args.relational_supports_per_procedure,
        relational_queries_per_procedure=args.relational_queries_per_procedure,
        adapter_meta_steps=args.adapter_meta_steps,
        adapter_supports_per_procedure=args.adapter_supports_per_procedure,
        adapter_queries_per_procedure=args.adapter_queries_per_procedure,
        adapter_learning_rate=args.adapter_learning_rate,
        checkpoint=args.checkpoint,
    )
    encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)
    if args.result_json:
        destination = Path(args.result_json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
