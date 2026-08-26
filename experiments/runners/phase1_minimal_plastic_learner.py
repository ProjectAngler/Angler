"""Run the first local Qwen3-4B plastic-learning transaction."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from angler.learning.fttt_update import (  # noqa: E402
    LoraUpdateBudget,
    TeacherForcedReflectionBatch,
    propose_bounded_lora_update,
)
from angler.runtime import (  # noqa: E402
    adapter_tensor_digest,
    attach_single_adapter,
    build_causal_lm_lora_config,
    reload_adapter_local,
    save_adapter_local,
    validate_foundation_frozen,
)
from angler.worlds import (  # noqa: E402
    LearnerTask,
    OutcomeFeedback,
    generate_relational_task,
    make_held_out_variant,
    verify_final_answer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/opt/angler/models/Qwen3-4B")
    parser.add_argument(
        "--adapter-dir",
        default="/opt/angler/project/work/phase1-minimal-plastic-candidate",
    )
    parser.add_argument("--parent-adapter")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--adaptation-tasks", type=int, default=4)
    parser.add_argument("--held-out-tasks", type=int, default=4)
    parser.add_argument("--item-count", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--consolidation-presentations", type=int, default=3)
    parser.add_argument("--retry-temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--submission-tokens", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--max-adapter-delta-norm", type=float, default=2.0)
    parser.add_argument("--include-records", action="store_true")
    return parser.parse_args()


def load_foundation(model_path: str) -> torch.nn.Module:
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )


def generation_inputs(tokenizer: Any, messages: Sequence[dict[str, str]]) -> Any:
    rendered = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(rendered, return_tensors="pt", add_special_tokens=False)


def generate_response(
    model: torch.nn.Module,
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
) -> str:
    prior_training = model.training
    model.eval()
    inputs = generation_inputs(tokenizer, messages).to(model.device)
    with torch.inference_mode():
        sampling: dict[str, Any] = {"do_sample": False}
        if temperature > 0.0:
            sampling = {
                "do_sample": True,
                "temperature": temperature,
                "top_p": 0.95,
            }
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            cache_implementation="dynamic",
            pad_token_id=tokenizer.eos_token_id,
            **sampling,
        )
    new_ids = output_ids[0, inputs.input_ids.shape[1] :]
    response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    model.train(prior_training)
    return response


def outcome_feedback(task: LearnerTask, outcome: OutcomeFeedback) -> str:
    if outcome.code == "INVALID_FINAL_ANSWER":
        return (
            "Outcome feedback: the submission was not exactly one comma-separated "
            "sequence containing every visible symbol once. Submit a new final "
            "comma-separated sequence only."
        )
    violated = [
        (
            index,
            task.constraints[index - 1].earlier,
            task.constraints[index - 1].later,
        )
        for index in outcome.violated_visible_constraints
    ]
    details = "; ".join(
        f"{index}: {earlier} must precede {later}"
        for index, earlier, later in violated
    )
    return (
        "Outcome feedback: the submitted sequence violated these already-visible "
        f"constraints: {details}. No answer or solution method is provided. "
        "Analyze how to correct the violations, but wait to submit the final "
        "sequence until asked."
    )


def final_submission_request() -> str:
    return (
        "Using your analysis, submit only the final comma-separated sequence. "
        "Do not include explanation or formatting."
    )


def text_digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def extract_submitted_answer(task: LearnerTask, response: str) -> str:
    """Extract a final sequence from model text without consulting hidden state."""

    lines = [line.strip() for line in response.splitlines() if line.strip()]
    for reverse_index, line in enumerate(reversed(lines)):
        lowered = line.lower()
        if "symbols" in lowered or "constraints" in lowered:
            continue
        is_final_region = reverse_index < 3 or "answer" in lowered or "final" in lowered
        if not is_final_region:
            continue
        candidates = [line]
        if ":" in line:
            candidates.append(line.rsplit(":", 1)[1])
        for candidate in candidates:
            cleaned = candidate.strip().strip("`*_#- ")
            parts = tuple(part.strip().strip("`*_. ") for part in cleaned.split(","))
            if len(parts) == len(task.symbols) and set(parts) == set(task.symbols):
                return ", ".join(parts)
    return response.strip()


def make_teacher_batch(
    tokenizer: Any,
    prefix_messages: Sequence[dict[str, str]],
    accepted_response: str,
    *,
    device: torch.device,
) -> TeacherForcedReflectionBatch:
    prefix = generation_inputs(tokenizer, prefix_messages)
    full_messages = [
        *prefix_messages,
        {"role": "assistant", "content": accepted_response},
    ]
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    full = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    prefix_length = int(prefix.input_ids.shape[1])
    if full.input_ids.shape[1] <= prefix_length:
        raise RuntimeError("accepted response produced no supervised tokens")
    if not torch.equal(full.input_ids[:, :prefix_length], prefix.input_ids):
        raise RuntimeError("chat-template prefix changed when response was appended")

    labels = torch.full_like(full.input_ids, -100)
    labels[:, prefix_length:] = full.input_ids[:, prefix_length:]
    return TeacherForcedReflectionBatch(
        input_ids=full.input_ids.to(device),
        reflection_labels=labels.to(device),
        attention_mask=full.attention_mask.to(device),
    )


def attempt_with_feedback(
    model: torch.nn.Module,
    tokenizer: Any,
    task: LearnerTask,
    *,
    max_retries: int,
    max_new_tokens: int,
    submission_tokens: int,
    retry_temperature: float,
) -> tuple[tuple[TeacherForcedReflectionBatch, ...], dict[str, Any]]:
    messages: list[dict[str, str]] = [{"role": "user", "content": task.prompt}]
    attempts: list[dict[str, Any]] = []

    for attempt_index in range(max_retries + 1):
        temperature = 0.0 if attempt_index == 0 else retry_temperature
        analysis_prefix_messages = tuple(messages)
        analysis = generate_response(
            model,
            tokenizer,
            analysis_prefix_messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        messages.extend(
            (
                {"role": "assistant", "content": analysis},
                {"role": "user", "content": final_submission_request()},
            )
        )
        prefix_messages = tuple(messages)
        submission = generate_response(
            model,
            tokenizer,
            prefix_messages,
            max_new_tokens=submission_tokens,
            temperature=temperature,
        )
        submitted_answer = extract_submitted_answer(task, submission)
        outcome = verify_final_answer(task, submitted_answer)
        attempts.append(
            {
                "attempt": attempt_index + 1,
                "analysis_digest": text_digest(analysis),
                "analysis_characters": len(analysis),
                "response": submission,
                "submitted_answer": submitted_answer,
                "outcome": asdict(outcome),
            }
        )
        if outcome.correct:
            accepted_batches = (
                make_teacher_batch(
                    tokenizer,
                    analysis_prefix_messages,
                    analysis,
                    device=model.device,
                ),
                make_teacher_batch(
                    tokenizer,
                    prefix_messages,
                    submission,
                    device=model.device,
                ),
            )
            return accepted_batches, {
                "task_id": task.instance_id,
                "accepted": True,
                "feedback_rounds": attempt_index,
                "attempts": attempts,
            }

        messages.extend(
            (
                {"role": "assistant", "content": submission},
                {"role": "user", "content": outcome_feedback(task, outcome)},
            )
        )

    return (), {
        "task_id": task.instance_id,
        "accepted": False,
        "feedback_rounds": max_retries,
        "attempts": attempts,
    }


def collect_clean_presentations(
    model: torch.nn.Module,
    tokenizer: Any,
    task: LearnerTask,
    *,
    presentations: int,
    max_new_tokens: int,
    submission_tokens: int,
    temperature: float,
) -> tuple[tuple[TeacherForcedReflectionBatch, ...], dict[str, Any]]:
    """Re-present one task in fresh contexts and retain only verified model traces."""

    batches: list[TeacherForcedReflectionBatch] = []
    records: list[dict[str, Any]] = []
    for presentation_index in range(presentations):
        analysis_prefix = ({"role": "user", "content": task.prompt},)
        analysis = generate_response(
            model,
            tokenizer,
            analysis_prefix,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        submission_prefix = (
            *analysis_prefix,
            {"role": "assistant", "content": analysis},
            {"role": "user", "content": final_submission_request()},
        )
        submission = generate_response(
            model,
            tokenizer,
            submission_prefix,
            max_new_tokens=submission_tokens,
            temperature=temperature,
        )
        submitted_answer = extract_submitted_answer(task, submission)
        outcome = verify_final_answer(task, submitted_answer)
        records.append(
            {
                "presentation": presentation_index + 1,
                "analysis_digest": text_digest(analysis),
                "analysis_characters": len(analysis),
                "response": submission,
                "submitted_answer": submitted_answer,
                "outcome": asdict(outcome),
            }
        )
        if outcome.correct:
            batches.extend(
                (
                    make_teacher_batch(
                        tokenizer,
                        analysis_prefix,
                        analysis,
                        device=model.device,
                    ),
                    make_teacher_batch(
                        tokenizer,
                        submission_prefix,
                        submission,
                        device=model.device,
                    ),
                )
            )

    return tuple(batches), {
        "requested": presentations,
        "accepted": len(batches) // 2,
        "records": records,
    }


def evaluate_tasks(
    model: torch.nn.Module,
    tokenizer: Any,
    tasks: Sequence[LearnerTask],
    *,
    max_new_tokens: int,
    submission_tokens: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    correct = 0
    valid = 0
    for task in tasks:
        messages: list[dict[str, str]] = [
            {"role": "user", "content": task.prompt}
        ]
        analysis = generate_response(
            model,
            tokenizer,
            tuple(messages),
            max_new_tokens=max_new_tokens,
        )
        messages.extend(
            (
                {"role": "assistant", "content": analysis},
                {"role": "user", "content": final_submission_request()},
            )
        )
        submission = generate_response(
            model,
            tokenizer,
            tuple(messages),
            max_new_tokens=submission_tokens,
        )
        submitted_answer = extract_submitted_answer(task, submission)
        outcome = verify_final_answer(task, submitted_answer)
        correct += int(outcome.correct)
        valid += int(outcome.disposition == "VALID_RESULT")
        records.append(
            {
                "task_id": task.instance_id,
                "analysis_digest": text_digest(analysis),
                "analysis_characters": len(analysis),
                "response": submission,
                "submitted_answer": submitted_answer,
                "outcome": asdict(outcome),
            }
        )
    return {
        "correct": correct,
        "valid_submissions": valid,
        "total": len(tasks),
        "accuracy": correct / len(tasks) if tasks else 0.0,
        "records": records,
    }


def summarize_initial_adaptation(experiences: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize each task's first presentation before any weight update."""

    initial_outcomes = [record["attempts"][0]["outcome"] for record in experiences]
    correct = sum(int(outcome["correct"]) for outcome in initial_outcomes)
    valid = sum(
        int(outcome["disposition"] == "VALID_RESULT")
        for outcome in initial_outcomes
    )
    return {
        "correct": correct,
        "valid_submissions": valid,
        "total": len(initial_outcomes),
        "accuracy": correct / len(initial_outcomes) if initial_outcomes else 0.0,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the 4B plastic-learning run")
    if args.adaptation_tasks <= 0 or args.held_out_tasks <= 0:
        raise ValueError("task counts must be positive")
    if args.update_epochs <= 0:
        raise ValueError("update_epochs must be positive")
    if args.consolidation_presentations <= 0:
        raise ValueError("consolidation_presentations must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    candidate_output = Path(args.adapter_dir).expanduser().resolve()
    if candidate_output.exists():
        raise FileExistsError(
            f"candidate output already exists; choose a fresh lineage path: {candidate_output}"
        )
    parent_adapter_path = (
        Path(args.parent_adapter).expanduser().resolve()
        if args.parent_adapter
        else None
    )
    if parent_adapter_path == candidate_output:
        raise ValueError("parent adapter and candidate output must be different paths")

    print("loading frozen foundation", file=sys.stderr, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    foundation = load_foundation(args.model)
    if parent_adapter_path is not None:
        model = reload_adapter_local(
            foundation,
            parent_adapter_path,
            trainable_adapter=True,
        )
    else:
        parent_adapter_path = None
        model = attach_single_adapter(
            foundation,
            build_causal_lm_lora_config(
                target_modules=("q_proj", "v_proj"),
                rank=8,
                alpha=16,
                dropout=0.0,
            ),
        )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.config.use_cache = False
    inventory = validate_foundation_frozen(model)
    parent_digest = adapter_tensor_digest(model)

    source_tasks = [
        generate_relational_task(args.seed + index * 101, item_count=args.item_count)
        for index in range(max(args.adaptation_tasks, args.held_out_tasks))
    ]
    adaptation = [task.learner for task in source_tasks[: args.adaptation_tasks]]
    held_out = [
        make_held_out_variant(task, seed=args.seed + 100_000 + index * 103).learner
        for index, task in enumerate(source_tasks[: args.held_out_tasks])
    ]

    print("measuring the current parent state", file=sys.stderr, flush=True)
    parent_transfer = evaluate_tasks(
        model,
        tokenizer,
        held_out,
        max_new_tokens=args.max_new_tokens,
        submission_tokens=args.submission_tokens,
    )

    print("collecting verifier-accepted model revisions", file=sys.stderr, flush=True)
    batches: list[TeacherForcedReflectionBatch] = []
    experiences: list[dict[str, Any]] = []
    for task in adaptation:
        accepted_batches, record = attempt_with_feedback(
            model,
            tokenizer,
            task,
            max_retries=args.max_retries,
            max_new_tokens=args.max_new_tokens,
            submission_tokens=args.submission_tokens,
            retry_temperature=args.retry_temperature,
        )
        experiences.append(record)
        if accepted_batches:
            batches.extend(accepted_batches)
        if accepted_batches and record["feedback_rounds"] > 0:
            clean_batches, consolidation = collect_clean_presentations(
                model,
                tokenizer,
                task,
                presentations=args.consolidation_presentations,
                max_new_tokens=args.max_new_tokens,
                submission_tokens=args.submission_tokens,
                temperature=args.retry_temperature,
            )
            batches.extend(clean_batches)
            record["consolidation"] = consolidation

    if not batches:
        raise RuntimeError(
            "the model produced no verifier-accepted revision after outcome feedback"
        )

    adaptation_before = summarize_initial_adaptation(experiences)

    training_batches = tuple(
        batch
        for _ in range(args.update_epochs)
        for batch in batches
    )
    total_input_tokens = sum(
        int(batch.attention_mask.sum().item())
        if batch.attention_mask is not None
        else int(batch.input_ids.numel())
        for batch in training_batches
    )
    total_supervised_tokens = sum(
        int(batch.reflection_labels[:, 1:].ne(-100).sum().item())
        for batch in training_batches
    )
    budget = LoraUpdateBudget(
        max_steps=len(training_batches),
        max_input_tokens=total_input_tokens,
        max_supervised_tokens=total_supervised_tokens,
        learning_rate=args.learning_rate,
        max_gradient_norm=args.max_gradient_norm,
        max_adapter_delta_norm=args.max_adapter_delta_norm,
    )

    print("proposing bounded neural-state update", file=sys.stderr, flush=True)
    candidate = propose_bounded_lora_update(model, training_batches, budget)
    with candidate:
        print("re-presenting learned tasks after the update", file=sys.stderr, flush=True)
        adaptation_after = evaluate_tasks(
            model,
            tokenizer,
            adaptation,
            max_new_tokens=args.max_new_tokens,
            submission_tokens=args.submission_tokens,
        )
        print("measuring candidate on renamed/reordered holdouts", file=sys.stderr, flush=True)
        candidate_evaluation = evaluate_tasks(
            model,
            tokenizer,
            held_out,
            max_new_tokens=args.max_new_tokens,
            submission_tokens=args.submission_tokens,
        )

        parent_competence = adaptation_before["correct"] + parent_transfer["correct"]
        candidate_competence = (
            adaptation_after["correct"] + candidate_evaluation["correct"]
        )
        retained = candidate_competence > parent_competence
        if retained:
            adapter_path = save_adapter_local(model, candidate_output)
            finalized = candidate.retain()
        else:
            finalized = candidate.reject()
            adapter_path = None

    torch.cuda.synchronize()
    finished = time.perf_counter()
    if args.include_records:
        reported_experiences = experiences
        reported_parent_transfer = parent_transfer
        reported_candidate = candidate_evaluation
    else:
        reported_experiences = [
            {
                "task_id": record["task_id"],
                "accepted": record["accepted"],
                "feedback_rounds": record["feedback_rounds"],
                "attempt_count": len(record["attempts"]),
                "final_outcome": record["attempts"][-1]["outcome"],
                "clean_presentations": {
                    "requested": record.get("consolidation", {}).get("requested", 0),
                    "accepted": record.get("consolidation", {}).get("accepted", 0),
                },
            }
            for record in experiences
        ]
        reported_parent_transfer = {
            key: value for key, value in parent_transfer.items() if key != "records"
        }
        reported_candidate = {
            key: value
            for key, value in candidate_evaluation.items()
            if key != "records"
        }
    initial_by_task = {
        record["task_id"]: record["attempts"][0]["outcome"]
        for record in experiences
    }
    corrected_task_ids = {
        record["task_id"]
        for record in experiences
        if record["accepted"] and record["feedback_rounds"] > 0
    }
    accepted_task_ids = {
        record["task_id"] for record in experiences if record["accepted"]
    }
    adaptation_task_changes = [
        {
            "task_id": record["task_id"],
            "received_learning_exposure": record["task_id"] in accepted_task_ids,
            "received_corrective_exposure": record["task_id"] in corrected_task_ids,
            "correct_before_update": initial_by_task[record["task_id"]]["correct"],
            "correct_after_update": record["outcome"]["correct"],
        }
        for record in adaptation_after["records"]
    ]
    result = {
        "experiment": "ANGLER-PHASE1-MINIMAL-PLASTIC-LEARNER",
        "status": "CANDIDATE_RETAINED" if retained else "CANDIDATE_REJECTED_ROLLED_BACK",
        "model_path": str(Path(args.model).resolve()),
        "seed": args.seed,
        "world": {
            "family": "angler.relational-order@1.0.0",
            "item_count": args.item_count,
            "adaptation_tasks": len(adaptation),
            "held_out_tasks": len(held_out),
        },
        "plastic_state": {
            "foundation_parameters": inventory.foundation_numel,
            "trainable_foundation_parameters": inventory.trainable_foundation_numel,
            "adapter_parameters": inventory.lora_numel,
            "trainable_adapter_parameters": inventory.trainable_lora_numel,
            "parent_digest": parent_digest,
            "parent_path": (
                str(parent_adapter_path) if parent_adapter_path is not None else None
            ),
            "final_digest": finalized.final_adapter_digest,
            "saved_path": str(adapter_path) if adapter_path is not None else None,
        },
        "experience": reported_experiences,
        "training_schedule": {
            "accepted_revision_batches": len(batches),
            "accepted_episode_traces": sum(
                int(record["accepted"])
                for record in experiences
            ),
            "corrected_tasks": sum(
                int(record["accepted"] and record["feedback_rounds"] > 0)
                for record in experiences
            ),
            "retention_anchor_traces": sum(
                int(record["accepted"] and record["feedback_rounds"] == 0)
                for record in experiences
            ),
            "fresh_consolidation_presentations_per_corrected_task": (
                args.consolidation_presentations
            ),
            "accepted_clean_presentations": sum(
                record.get("consolidation", {}).get("accepted", 0)
                for record in experiences
            ),
            "learning_replays_per_accepted_trace": args.update_epochs,
            "batches_per_presentation": 2,
            "optimizer_steps": len(training_batches),
        },
        "presented_task_adaptation": {
            "before_update": adaptation_before,
            "after_update": {
                key: value
                for key, value in adaptation_after.items()
                if key != "records"
            },
            "task_changes": adaptation_task_changes,
        },
        "parent_state_transfer": reported_parent_transfer,
        "candidate_held_out": reported_candidate,
        "improvement_decision": {
            "metric": "presented_correct_plus_paired_transfer_correct",
            "parent": parent_competence,
            "candidate": candidate_competence,
            "improvement": candidate_competence - parent_competence,
        },
        "update": asdict(candidate.receipt),
        "decision": asdict(finalized),
        "wall_seconds": round(finished - started, 3),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "hidden_solutions_exposed_to_model_or_updater": False,
        "local_files_only": True,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
