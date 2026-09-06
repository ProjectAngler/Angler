"""V21-B: connect frozen V20/V19 representations to production behavior.

The representation donors and their learned fast-state initialization remain
frozen.  Only the pre-existing action, transition, backward-reasoning, and
STOP path is trained from public leave-one-package-out support traces.  This
module contains no task solver, plan repair, motif router, or query target
lookup.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Mapping, Sequence

import torch

from experiments.runners import phase6_software_pipeline_reconstruction as v12
from experiments.runners import phase6_v12_champion_paired_graph_context as v19
from experiments.runners import phase6_oml_relation_representation as v20


IDENTITY = "angler.oml-production-bridge.v21b"
SEED = 20260831
EPOCHS = 1
TRAIN_MECHANISMS = 64
DEVELOPMENT_MECHANISMS = 16
FINAL_MECHANISMS = 16
SUPPORTS_PER_MOTIF = 2
QUERIES_PER_MECHANISM = 1
MAXIMUM_STEPS = 4
LEARNING_RATE = 2.0e-3
GRADIENT_CLIP = 5.0

V19_CHECKPOINT_SHA256 = v20.SOURCE_CHECKPOINT_SHA256.lower()
V20_CHECKPOINT_SHA256 = (
    "d49e4caab64a264a11c675b295a8c453ac4475f078311eb7283a4f9a8817ef48"
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v19-checkpoint", required=True)
    parser.add_argument("--v20-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bridge_parameter_names(
    controller: v19.V12ChampionPairedGraphContextController,
) -> tuple[str, ...]:
    if type(controller) is not v19.V12ChampionPairedGraphContextController:
        raise TypeError("V21-B requires the exact V19/V20 controller")
    donor = set(v12._relation_matcher_parameter_names(controller)) | set(
        v19.MUTABLE_PARAMETER_NAMES
    )
    names = tuple(name for name, _ in controller.named_parameters() if name not in donor)
    named = dict(controller.named_parameters())
    if len(names) != 212 or sum(named[name].numel() for name in names) != 158_674:
        raise RuntimeError("V21-B production parameter identity changed")
    return names


def configure_bridge_trainability(
    controller: v19.V12ChampionPairedGraphContextController,
    *,
    train: bool,
) -> tuple[str, ...]:
    selected = set(bridge_parameter_names(controller))
    for name, parameter in controller.named_parameters():
        parameter.requires_grad_(train and name in selected)
        parameter.grad = None
    actual = {
        name for name, parameter in controller.named_parameters() if parameter.requires_grad
    }
    expected = selected if train else set()
    if actual != expected:
        raise RuntimeError("V21-B production ownership changed")
    controller.train(train)
    return tuple(name for name, _ in controller.named_parameters() if name in selected)


def _protected_state(
    controller: v19.V12ChampionPairedGraphContextController,
) -> dict[str, torch.Tensor]:
    selected = set(bridge_parameter_names(controller))
    return {
        name: value.detach().cpu().clone()
        for name, value in controller.state_dict().items()
        if name not in selected
    }


def _assert_protected_exact(
    controller: v19.V12ChampionPairedGraphContextController,
    before: Mapping[str, torch.Tensor],
) -> None:
    current = controller.state_dict()
    if set(before) - set(current):
        raise RuntimeError("V21-B protected tensor disappeared")
    changed = tuple(
        name
        for name, value in before.items()
        if not torch.equal(value, current[name].detach().cpu())
    )
    if changed:
        raise RuntimeError(f"V21-B changed frozen donor tensors: {changed[:3]}")


def _stream(partition: str, mechanism_index: int, epoch: int = 0):
    commitments = v12.software_pipeline_mechanism_partition(partition)
    seed = v12._experiment_seed(SEED, partition, epoch, mechanism_index)
    return v12.make_software_pipeline_stream(
        seed,
        supports_per_motif=SUPPORTS_PER_MOTIF,
        queries=QUERIES_PER_MECHANISM,
        maximum_steps=MAXIMUM_STEPS,
        mechanism_commitment=commitments[mechanism_index],
        mechanism_partition=partition,
    )


def acquire_supports(
    controller: v19.V12ChampionPairedGraphContextController,
    supports: Sequence[object],
) -> v19.V19SoftwareReconstructionState:
    state = controller.initial_state()
    for pair in supports:
        state = v19.acquire_v19_public_pipeline_traces(
            controller, pair.learner, state
        ).state
    return state


def fit_production_bridge(
    controller: v19.V12ChampionPairedGraphContextController,
) -> dict[str, object]:
    selected = configure_bridge_trainability(controller, train=True)
    named = dict(controller.named_parameters())
    parameters = tuple(named[name] for name in selected)
    protected = _protected_state(controller)
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=0.0)
    losses: list[float] = []
    gradients: list[float] = []
    started = time.perf_counter()
    for epoch in range(EPOCHS):
        for mechanism_index in range(TRAIN_MECHANISMS):
            stream = _stream("train", mechanism_index, epoch)
            fold_losses = []
            for heldout_index, heldout in enumerate(stream.supports):
                evidence = tuple(
                    pair
                    for index, pair in enumerate(stream.supports)
                    if index != heldout_index
                )
                state = acquire_supports(controller, evidence)
                terms = controller.public_heldout_production_losses(
                    heldout.learner,
                    state,
                    include_role_memory_causal_hinge=True,
                    detach_evidence_action_input=True,
                    use_legacy_evidence=False,
                )
                fold_losses.append(terms.mean())
            loss = torch.stack(fold_losses).mean()
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("V21-B production loss is non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP)
            if not bool(torch.isfinite(gradient).item()):
                raise RuntimeError("V21-B gradient norm is non-finite")
            optimizer.step()
            _assert_protected_exact(controller, protected)
            losses.append(float(loss.detach().item()))
            gradients.append(float(gradient.detach().item()))
    configure_bridge_trainability(controller, train=False)
    return {
        "epochs": EPOCHS,
        "mechanisms": TRAIN_MECHANISMS,
        "optimizer_steps": len(losses),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "loss_reduction": (losses[0] - losses[-1]) / max(abs(losses[0]), 1.0e-8),
        "mean_gradient_norm": sum(gradients) / len(gradients),
        "trainable_parameter_names": selected,
        "trainable_parameter_count": sum(named[name].numel() for name in selected),
        "donor_tensors_exact": True,
        "query_targets_used": 0,
        "hidden_fields_used": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }


def copy_bridge(
    source: v19.V12ChampionPairedGraphContextController,
    target: v19.V12ChampionPairedGraphContextController,
) -> None:
    source_named = dict(source.named_parameters())
    target_named = dict(target.named_parameters())
    names = bridge_parameter_names(source)
    if names != bridge_parameter_names(target):
        raise RuntimeError("V21-B control has a different bridge shape")
    with torch.no_grad():
        for name in names:
            target_named[name].copy_(source_named[name])
    configure_bridge_trainability(target, train=False)


def evaluate_partition(
    controller: v19.V12ChampionPairedGraphContextController,
    partition: str,
    mechanisms: int,
    *,
    include_role_memory: bool = True,
    include_backward_reasoning: bool = True,
    paired_graph_lesion: str | None = None,
) -> dict[str, object]:
    configure_bridge_trainability(controller, train=False)
    scores: list[float] = []
    plans: list[tuple[int, ...]] = []
    context = (
        controller.paired_graph_lesion(paired_graph_lesion)
        if paired_graph_lesion is not None
        else nullcontext()
    )
    with context, torch.no_grad():
        for mechanism_index in range(mechanisms):
            stream = _stream(partition, mechanism_index)
            state = acquire_supports(controller, stream.supports)
            snapshot = v19.snapshot_v19_reconstruction_state(state)
            for pair in stream.queries:
                query_state = v19.restore_v19_reconstruction_state(snapshot)
                rollout = v12.rollout_software_pipeline(
                    controller,
                    pair.learner,
                    query_state,
                    include_role_memory=include_role_memory,
                    include_backward_reasoning=include_backward_reasoning,
                )
                scores.append(float(v12.judge_software_pipeline_attempt(pair, rollout.pipeline)))
                plans.append(tuple(int(value) for value in rollout.selected_indices))
    return {
        "accuracy": sum(scores) / len(scores),
        "correct": int(sum(scores)),
        "queries": len(scores),
        "plans": plans,
    }


def save_checkpoint(
    path: str | Path,
    controller: v19.V12ChampionPairedGraphContextController,
) -> None:
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    names = set(bridge_parameter_names(controller))
    torch.save(
        {
            "version": "angler.oml-production-bridge.v21b.v1",
            "v19_checkpoint_sha256": V19_CHECKPOINT_SHA256,
            "v20_checkpoint_sha256": V20_CHECKPOINT_SHA256,
            "bridge_state": {
                name: value.detach().cpu()
                for name, value in controller.state_dict().items()
                if name in names
            },
        },
        temporary,
    )
    temporary.replace(target)


def run(args: argparse.Namespace) -> dict[str, object]:
    if _sha256(args.v19_checkpoint) != V19_CHECKPOINT_SHA256:
        raise RuntimeError("V21-B V19 checkpoint changed")
    if _sha256(args.v20_checkpoint) != V20_CHECKPOINT_SHA256:
        raise RuntimeError("V21-B V20 checkpoint changed")
    device = torch.device(args.device)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()
    system = v20.load_oml_checkpoint(
        args.v20_checkpoint, args.v19_checkpoint, device=device
    )
    controller = system.second_order_oml.controller
    source_control = copy.deepcopy(system.source_v19.controller)
    before = {
        partition: evaluate_partition(
            controller,
            partition,
            DEVELOPMENT_MECHANISMS if partition == "development" else FINAL_MECHANISMS,
        )
        for partition in ("development", "final")
    }
    training = fit_production_bridge(controller)
    copy_bridge(controller, source_control)
    panels = {}
    for partition, count in (
        ("development", DEVELOPMENT_MECHANISMS),
        ("final", FINAL_MECHANISMS),
    ):
        panels[partition] = {
            "full": evaluate_partition(controller, partition, count),
            "paired_graph_removed": evaluate_partition(
                controller, partition, count, paired_graph_lesion="zero_residual"
            ),
            "oml_removed": evaluate_partition(source_control, partition, count),
            "role_memory_removed": evaluate_partition(
                controller, partition, count, include_role_memory=False
            ),
            "backward_reasoning_removed": evaluate_partition(
                controller, partition, count, include_backward_reasoning=False
            ),
        }
    final = panels["final"]
    full = float(final["full"]["accuracy"])
    component_supported = (
        float(panels["development"]["full"]["accuracy"]) >= 0.5
        and full >= 0.5
        and full - float(before["final"]["accuracy"]) >= 0.25
        and full - float(final["paired_graph_removed"]["accuracy"]) >= 0.1
        and full - float(final["oml_removed"]["accuracy"]) >= 0.1
    )
    save_checkpoint(args.checkpoint, controller)
    result = {
        "identity": IDENTITY,
        "classification": (
            "OML_PRODUCTION_BRIDGE_SUPPORTED"
            if component_supported
            else "OML_PRODUCTION_BRIDGE_NOT_SUPPORTED"
        ),
        "component_supported": component_supported,
        "first_result_accepted_without_tuning": True,
        "training": training,
        "before_training": before,
        "panels": panels,
        "source": {
            "v19_checkpoint_sha256": V19_CHECKPOINT_SHA256,
            "v20_checkpoint_sha256": V20_CHECKPOINT_SHA256,
            "oml_arm": v20.ARM_SECOND_ORDER,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name()
                if device.type == "cuda"
                else "cpu"
            ),
            "torch_threads": torch.get_num_threads(),
            "peak_cuda_bytes": (
                torch.cuda.max_memory_allocated()
                if device.type == "cuda"
                else 0
            ),
        },
        "checkpoint_path": str(args.checkpoint),
        "deterministic_solver_used": False,
        "moving_origin_in_this_component": False,
        "cognee_in_this_component": False,
        "integration_note": (
            "V21-B isolates the previously missing representation-to-action bridge; "
            "Moving Origin and Cognee remain in the separate situated runtime lane."
        ),
    }
    output = Path(args.output)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
