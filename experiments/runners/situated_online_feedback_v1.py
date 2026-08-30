"""First replay-free Qwen-outcome adaptation test for situated selection."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
import time

import torch

from angler.memory import RecallBatch, SituatedRecall
from angler.reasoning import (
    LearnedSituatedMemoryReader,
    SituatedFeatureSpec,
    SituatedFeedbackPolicy,
    encode_situated_features,
    situated_outcome_loss,
)
from angler.runtime import (
    SituatedEvidenceSelection,
    build_qwen_prompt,
    encode_detached_segments,
    freeze_knowledge_model,
)
from experiments.runners.situated_qwen_generation_v1 import _generate, _sha256


IDENTITY = "angler.situated-online-feedback.v1"
SEED = 20260841
ACTIONS = ("trace", "stage", "invert", "seal")
FAMILIES = (
    "Kestrel dependency reconstruction",
    "Lumen interface migration",
    "Orchid regression recovery",
    "Harbor resource stabilization",
)
TARGETS = (1, 3, 2, 0)
SPEC = SituatedFeatureSpec(("feedback_epoch",), include_acquired_ordinal=False)
STREAM_COUNT = 96
PROBE_COUNT = 32
RANK = 32
MAXIMUM_RESIDUAL = 4.0
LEARNING_RATE = 3.0e-3
GRADIENT_CLIP = 2.0
RESPONSE_CONSTRAINT = (
    "The first word of your response must be the selected procedure. "
    "Return only one of: trace, stage, invert, seal."
)
_ACTION_PATTERN = re.compile(r"\b(?:trace|stage|invert|seal)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class FeedbackSample:
    identity: str
    family_index: int
    exposure_index: int
    query: str
    recall: RecallBatch
    target_action: int
    now: int


@dataclass(slots=True)
class EncodedSample:
    sample: FeedbackSample
    query: torch.Tensor
    candidates: torch.Tensor
    temporal: torch.Tensor
    mask: torch.Tensor
    base_scores: torch.Tensor


def parse_action(response: str) -> str | None:
    found = {match.group(0).lower() for match in _ACTION_PATTERN.finditer(response)}
    return next(iter(found)) if len(found) == 1 else None


def build_stream(*, seed: int = SEED) -> tuple[tuple[FeedbackSample, ...], tuple[FeedbackSample, ...]]:
    rng = random.Random(seed)
    schedule = (
        tuple(index % 2 for index in range(24))
        + tuple(2 + index % 2 for index in range(24))
        + tuple((0, 2, 1, 3)[index % 4] for index in range(48))
    )
    exposures = [0] * len(FAMILIES)

    def make(family_index: int, *, probe: bool) -> FeedbackSample:
        exposure = exposures[family_index]
        exposures[family_index] += 1
        order = list(range(len(ACTIONS)))
        rng.shuffle(order)
        items = []
        for position, action_index in enumerate(order):
            age = len(ACTIONS) - position - 1
            items.append(
                SituatedRecall(
                    artifact_ref=_ref(seed, family_index, exposure, action_index, "artifact"),
                    text=(
                        f"Reusable procedure proposal for a prior {FAMILIES[family_index]} "
                        f"case: {ACTIONS[action_index]}. This projection does not contain "
                        "the outcome for the current variant."
                    ),
                    source_ref=_ref(seed, family_index, exposure, action_index, "source"),
                    context=(("family", FAMILIES[family_index]),),
                    visibility="LEARNER_VISIBLE",
                    acquired_ordinal=position,
                    age=age,
                    landmark_relations=(("feedback_epoch", "UNKNOWN"),),
                    world_valid_from=None,
                    world_valid_until=None,
                    world_valid_at_query=None,
                    backend_score=0.75,
                    backend_ref=f"feedback-{family_index}-{exposure}-{action_index}",
                )
            )
        kind = "probe" if probe else "stream"
        identity = _ref(seed, family_index, exposure, 0, kind)
        query = (
            f"A fresh {FAMILIES[family_index]} case variant {exposure:02d} has arrived. "
            "Choose the locally effective reusable procedure from prior experience."
        )
        return FeedbackSample(
            identity=identity,
            family_index=family_index,
            exposure_index=exposure,
            query=query,
            recall=RecallBatch(tuple(items)),
            target_action=TARGETS[family_index],
            now=len(ACTIONS),
        )

    stream = tuple(make(family, probe=False) for family in schedule)
    probes = tuple(make(family, probe=True) for family in range(4) for _ in range(8))
    if len(stream) != STREAM_COUNT or len(probes) != PROBE_COUNT:
        raise RuntimeError("frozen stream size is inconsistent")
    identities = [sample.identity for sample in stream + probes]
    if len(set(identities)) != len(identities):
        raise RuntimeError("stream identities are not unique")
    return stream, probes


def _ref(seed: int, family: int, exposure: int, action: int, salt: str) -> str:
    payload = f"{salt}:{seed}:{family}:{exposure}:{action}".encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _tensor_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256(b"project-angler.situated-online-feedback.frozen.v1\x00")
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\x00")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def _encode_samples(
    samples: tuple[FeedbackSample, ...],
    embeddings: dict[str, torch.Tensor],
    reader: LearnedSituatedMemoryReader,
) -> tuple[EncodedSample, ...]:
    encoded = []
    device = next(reader.parameters()).device
    dtype = next(reader.parameters()).dtype
    for sample in samples:
        query = embeddings[sample.query].to(device=device, dtype=dtype).unsqueeze(0)
        candidates = torch.stack(
            [embeddings[item.text] for item in sample.recall.items]
        ).to(device=device, dtype=dtype).unsqueeze(0)
        temporal, mask = encode_situated_features(
            (sample.recall,),
            now=(sample.now,),
            spec=SPEC,
            device=device,
            dtype=dtype,
        )
        with torch.inference_mode():
            base_scores = reader(query, candidates, temporal, mask).scores
        encoded.append(EncodedSample(sample, query, candidates, temporal, mask, base_scores))
    return tuple(encoded)


def _selection(
    policy: SituatedFeedbackPolicy,
    row: EncodedSample,
):
    output = policy(row.query, row.candidates, row.temporal, row.mask, row.base_scores)
    index = int(output.weights.argmax(dim=1).item())
    item = row.sample.recall.items[index]
    action = next(action for action in ACTIONS if re.search(rf"\b{action}\b", item.text))
    selection = SituatedEvidenceSelection(
        artifact_ref=item.artifact_ref,
        text=item.text,
        candidate_index=index,
        attention=float(output.weights[0, index].detach().item()),
        attention_distribution=tuple(float(value) for value in output.weights[0].detach().tolist()),
    )
    return output, index, action, selection


def _probe_accuracy(policy: SituatedFeedbackPolicy, rows: tuple[EncodedSample, ...], families=None) -> float:
    selected_rows = [row for row in rows if families is None or row.sample.family_index in families]
    correct = 0
    with torch.inference_mode():
        for row in selected_rows:
            _, _, action, _ = _selection(policy, row)
            correct += int(action == ACTIONS[row.sample.target_action])
    return correct / len(selected_rows)


def _stage_metrics(records: list[dict[str, object]], start: int, end: int) -> dict[str, dict[str, float]]:
    result = {}
    for arm in ("live", "frozen", "shuffled"):
        window = records[start:end]
        result[arm] = {
            "qwen_accuracy": sum(bool(row[arm]["qwen_correct"]) for row in window) / len(window),
            "selection_accuracy": sum(bool(row[arm]["selection_correct"]) for row in window) / len(window),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/opt/angler/models/Qwen3-4B")
    parser.add_argument("--reader-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("online situated feedback requires CUDA")
    output_path = Path(args.output)
    if output_path.exists():
        raise RuntimeError("online situated feedback output already exists")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    stream, probes = build_stream()
    started = time.monotonic()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
    )
    freeze_knowledge_model(model)
    foundation_before = _tensor_digest(model)
    checkpoint_path = Path(args.reader_checkpoint)
    sealed = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reader = LearnedSituatedMemoryReader(
        content_width=int(sealed["content_width"]),
        temporal_width=SPEC.width,
        hidden_width=256,
        action_count=4,
    ).cuda()
    reader.load_state_dict(sealed["state_dict"], strict=True)
    reader.requires_grad_(False)
    reader.eval()
    reader_before = _tensor_digest(reader)

    texts = tuple(
        text
        for sample in stream + probes
        for text in (sample.query, *(item.text for item in sample.recall.items))
    )
    unique = tuple(dict.fromkeys(texts))
    features = encode_detached_segments(
        model,
        tokenizer,
        unique,
        batch_size=args.embedding_batch_size,
        storage_dtype=torch.bfloat16,
    )
    embeddings = {text: features[index].float() for index, text in enumerate(unique)}
    stream_rows = _encode_samples(stream, embeddings, reader)
    probe_rows = _encode_samples(probes, embeddings, reader)

    torch.manual_seed(SEED + 1)
    live = SituatedFeedbackPolicy(
        content_width=reader.content_width,
        temporal_width=SPEC.width,
        rank=RANK,
        maximum_residual=MAXIMUM_RESIDUAL,
    ).cuda()
    initial_state = live.capture_plastic_state()
    frozen = copy.deepcopy(live)
    shuffled = copy.deepcopy(live)
    live_optimizer = torch.optim.AdamW(live.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    shuffled_optimizer = torch.optim.AdamW(shuffled.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    initial_probe = _probe_accuracy(live, probe_rows)
    initial_digest = live.plastic_state_digest()
    records = []
    prior_shuffled_reward = torch.tensor([-1.0], device="cuda")
    stage_a_probe = None

    for step, row in enumerate(stream_rows):
        arm_values = {}
        prompts = []
        selections = {}
        for name, policy in (("live", live), ("frozen", frozen), ("shuffled", shuffled)):
            _, index, action, selection = _selection(policy, row)
            selections[name] = (index, action, selection)
            prompts.append(
                build_qwen_prompt(
                    row.sample.query,
                    selected=selection,
                    response_constraint=RESPONSE_CONSTRAINT,
                )
            )
        responses = _generate(model, tokenizer, prompts, batch_size=3, max_new_tokens=32)
        target = ACTIONS[row.sample.target_action]
        for (name, (_, action, selection)), response in zip(selections.items(), responses, strict=True):
            parsed = parse_action(response)
            arm_values[name] = {
                "selected_action": action,
                "selected_ref": selection.artifact_ref,
                "selection_correct": action == target,
                "response": response,
                "parsed": parsed,
                "qwen_correct": parsed == target,
            }

        live_optimizer.zero_grad(set_to_none=True)
        live_output, live_index, _, _ = _selection(live, row)
        live_reward = torch.tensor(
            [1.0 if arm_values["live"]["qwen_correct"] else -1.0], device="cuda"
        )
        live_loss = situated_outcome_loss(
            live_output,
            torch.tensor([live_index], device="cuda"),
            live_reward,
        )
        live_loss.backward()
        torch.nn.utils.clip_grad_norm_(live.parameters(), GRADIENT_CLIP)
        live_optimizer.step()

        shuffled_optimizer.zero_grad(set_to_none=True)
        shuffled_output, shuffled_index, _, _ = _selection(shuffled, row)
        shuffled_loss = situated_outcome_loss(
            shuffled_output,
            torch.tensor([shuffled_index], device="cuda"),
            prior_shuffled_reward,
        )
        shuffled_loss.backward()
        torch.nn.utils.clip_grad_norm_(shuffled.parameters(), GRADIENT_CLIP)
        shuffled_optimizer.step()
        prior_shuffled_reward = torch.tensor(
            [1.0 if arm_values["shuffled"]["qwen_correct"] else -1.0], device="cuda"
        )
        records.append(
            {
                "step": step,
                "sample_identity": row.sample.identity,
                "family_index": row.sample.family_index,
                "target": target,
                **arm_values,
            }
        )
        if step == 23:
            stage_a_probe = _probe_accuracy(live, probe_rows, families={0, 1})

    if stage_a_probe is None:
        raise RuntimeError("stage-A probe was not recorded")
    reset = copy.deepcopy(live)
    reset.restore_plastic_state(initial_state)
    if reset.plastic_state_digest() != initial_digest:
        raise RuntimeError("reset did not restore the initial plastic state")
    if frozen.plastic_state_digest() != initial_digest:
        raise RuntimeError("the frozen control changed")
    terminal_probe = {
        "live": _probe_accuracy(live, probe_rows),
        "frozen": _probe_accuracy(frozen, probe_rows),
        "shuffled": _probe_accuracy(shuffled, probe_rows),
        "reset": _probe_accuracy(reset, probe_rows),
        "live_early_families": _probe_accuracy(live, probe_rows, families={0, 1}),
        "stage_a_early_families": stage_a_probe,
    }
    stages = {
        "stage_a": _stage_metrics(records, 0, 24),
        "stage_b": _stage_metrics(records, 24, 48),
        "final_interleaved": _stage_metrics(records, 48, 96),
    }
    final_live = stages["final_interleaved"]["live"]["qwen_accuracy"]
    final_frozen = stages["final_interleaved"]["frozen"]["qwen_accuracy"]
    final_shuffled = stages["final_interleaved"]["shuffled"]["qwen_accuracy"]
    supported = (
        final_live >= 0.65
        and final_live - final_frozen >= 0.20
        and final_live - final_shuffled >= 0.20
        and terminal_probe["live"] >= 0.65
        and terminal_probe["live"] - terminal_probe["frozen"] >= 0.20
        and terminal_probe["live"] - terminal_probe["reset"] >= 0.20
        and terminal_probe["live_early_families"] >= stage_a_probe - 0.15
    )
    foundation_after = _tensor_digest(model)
    reader_after = _tensor_digest(reader)
    if foundation_after != foundation_before or reader_after != reader_before:
        raise RuntimeError("frozen Qwen or sealed reader changed")
    if live.plastic_parameter_count != frozen.plastic_parameter_count:
        raise RuntimeError("plastic state size changed across arms")
    if not all(torch.isfinite(parameter).all().item() for parameter in live.parameters()):
        raise RuntimeError("live plastic state is not finite")
    torch.cuda.synchronize()
    report = {
        "identity": IDENTITY,
        "classification": "ONLINE_SITUATED_FEEDBACK_SUPPORTED" if supported else "NOT_SUPPORTED",
        "metrics": {
            "initial_probe_selection_accuracy": initial_probe,
            "stages": stages,
            "terminal_probe": terminal_probe,
        },
        "thresholds": {
            "minimum_final_live_qwen_accuracy": 0.65,
            "minimum_live_gain_over_each_control": 0.20,
            "minimum_live_probe_accuracy": 0.65,
            "minimum_live_probe_gain_over_frozen": 0.20,
            "minimum_reset_loss": 0.20,
            "maximum_early_family_retention_loss": 0.15,
        },
        "inputs": {
            "reader_checkpoint_sha256": _sha256(checkpoint_path),
            "foundation_model": args.model,
            "stream_digest": hashlib.sha256("\n".join(sample.identity for sample in stream).encode()).hexdigest(),
            "probe_digest": hashlib.sha256("\n".join(sample.identity for sample in probes).encode()).hexdigest(),
        },
        "state": {
            "plastic_parameters": live.plastic_parameter_count,
            "initial_digest": initial_digest,
            "terminal_live_digest": live.plastic_state_digest(),
            "terminal_shuffled_digest": shuffled.plastic_state_digest(),
            "reset_digest": reset.plastic_state_digest(),
            "foundation_digest": foundation_after,
            "reader_digest": reader_after,
        },
        "counts": {"stream": len(stream), "probes": len(probes), "families": 4, "actions": 4},
        "records": records,
        "runtime": {
            "device": "cuda",
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": __import__("transformers").__version__,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "wall_seconds": time.monotonic() - started,
            "seed": SEED,
        },
        "limits": [
            "Four synthetic recurring software-procedure families, not arbitrary software repair.",
            "Outcome feedback is evaluator supplied and no answer label enters the learner.",
            "No replay, Qwen update, sealed-reader update, AGI, consciousness, or production claim.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"classification": report["classification"], "metrics": report["metrics"], "runtime": report["runtime"]}, sort_keys=True))


if __name__ == "__main__":
    main()
