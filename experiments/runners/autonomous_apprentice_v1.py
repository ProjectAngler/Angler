"""First frozen dual-GPU evaluation of autonomous software apprenticeship."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import torch

from angler.memory import MovingOriginIndex, SituatedMemory
from angler.reasoning import SituatedFeatureSpec
from angler.runtime import (
    ActionTraceCompetenceLearner,
    AtomicCompetenceStore,
    AutonomousApprentice,
    AutonomousTaskContract,
    ConversationJournal,
    HybridChunkCogneeProjectionBackend,
    LocalQwenIO,
    SituatedEpisodeRecorder,
    foundation_tensor_digest,
    freeze_knowledge_model,
)


IDENTITY = "angler.autonomous-apprentice.v1-first-result"
SEED = 20260831
ARMS = ("full", "state_reset", "memory_disabled", "moving_origin_removed")


@dataclass(frozen=True, slots=True)
class TaskTemplate:
    task_id: str
    mechanism: str
    phase: str
    request: str
    initial_source: str
    verifier_source: str


TASKS = (
    TaskTemplate(
        "slug-a",
        "canonical_slug",
        "acquisition",
        "Implement normalize_label(text) in solution.py. Strip outer whitespace, lowercase the text, replace each run of non-alphanumeric characters with one hyphen, and remove leading or trailing hyphens.",
        "def normalize_label(text):\n    return text\n",
        "from solution import normalize_label as f; assert f('  Hello,   WORLD!  ') == 'hello-world'; assert f('--A__B--') == 'a-b'; assert f('***') == ''",
    ),
    TaskTemplate(
        "unique-a",
        "stable_unique",
        "acquisition",
        "Implement stable_unique(items) in solution.py. Return a list containing the first occurrence of each hashable item while preserving input order.",
        "def stable_unique(items):\n    return list(items)\n",
        "from solution import stable_unique as f; assert f([3,1,3,2,1]) == [3,1,2]; assert f(['a','a','b']) == ['a','b']; assert f([]) == []",
    ),
    TaskTemplate(
        "clamp-a",
        "inclusive_clamp",
        "acquisition",
        "Implement clamp_number(value, lower, upper) in solution.py. Return value limited inclusively to [lower, upper], and raise ValueError when lower is greater than upper.",
        "def clamp_number(value, lower, upper):\n    return value\n",
        "from solution import clamp_number as f; assert f(-2,0,5)==0; assert f(8,0,5)==5; assert f(3,0,5)==3;\ntry: f(1,2,0)\nexcept ValueError: pass\nelse: raise AssertionError('bounds')",
    ),
    TaskTemplate(
        "chunk-a",
        "bounded_chunks",
        "acquisition",
        "Implement chunked(items, size) in solution.py. Return consecutive list chunks of at most size items, preserve order, and raise ValueError when size is not positive.",
        "def chunked(items, size):\n    return [list(items)]\n",
        "from solution import chunked as f; assert f([1,2,3,4,5],2)==[[1,2],[3,4],[5]]; assert f([],3)==[];\ntry: f([1],0)\nexcept ValueError: pass\nelse: raise AssertionError('size')",
    ),
    TaskTemplate(
        "slug-b",
        "canonical_slug",
        "transfer",
        "Implement canonical_key(value) in solution.py. Trim outer whitespace, lowercase it, collapse every run of characters that are not letters or digits to one hyphen, then trim hyphens from both ends.",
        "def canonical_key(value):\n    return value.lower()\n",
        "from solution import canonical_key as f; assert f('  Red / GREEN + Blue ')=='red-green-blue'; assert f('X...Y')=='x-y'; assert f('___')==''",
    ),
    TaskTemplate(
        "unique-b",
        "stable_unique",
        "transfer",
        "Implement ordered_distinct(values) in solution.py. Produce a list with duplicates removed while retaining only each value's earliest position. Inputs are hashable.",
        "def ordered_distinct(values):\n    return sorted(set(values))\n",
        "from solution import ordered_distinct as f; assert f(['z','a','z','b','a'])==['z','a','b']; assert f([2,2,1])==[2,1]; assert f(())==[]",
    ),
    TaskTemplate(
        "clamp-b",
        "inclusive_clamp",
        "transfer",
        "Implement bound_value(number, minimum, maximum) in solution.py. Clamp number inclusively to the declared bounds and reject reversed bounds with ValueError.",
        "def bound_value(number, minimum, maximum):\n    return minimum\n",
        "from solution import bound_value as f; assert f(5,1,4)==4; assert f(-1,1,4)==1; assert f(2,1,4)==2;\ntry: f(0,3,2)\nexcept ValueError: pass\nelse: raise AssertionError('bounds')",
    ),
    TaskTemplate(
        "chunk-b",
        "bounded_chunks",
        "transfer",
        "Implement grouped(sequence, width) in solution.py. Return order-preserving consecutive list groups no larger than width, return an empty list for empty input, and raise ValueError unless width is positive.",
        "def grouped(sequence, width):\n    return []\n",
        "from solution import grouped as f; assert f('abcde',3)==[['a','b','c'],['d','e']]; assert f([],2)==[];\ntry: f([1],-1)\nexcept ValueError: pass\nelse: raise AssertionError('width')",
    ),
)


class QwenJsonPolicy:
    """Extract one JSON action object without adding task logic."""

    def __init__(self, qwen: LocalQwenIO) -> None:
        self.qwen = qwen
        self.calls = 0
        self.raw_hashes: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        raw = self.qwen.generate(prompt).strip()
        self.raw_hashes.append(hashlib.sha256(raw.encode()).hexdigest())
        try:
            value = json.loads(raw)
            if isinstance(value, dict):
                return json.dumps(value, separators=(",", ":"))
        except json.JSONDecodeError:
            pass
        decoder = json.JSONDecoder()
        for index, character in enumerate(raw):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(raw[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return json.dumps(value, separators=(",", ":"))
        return raw


class NullRecorder:
    async def record(self, episode) -> None:
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configure_cognee(storage_root: Path) -> Any:
    if os.getenv("TELEMETRY_DISABLED") != "1":
        raise RuntimeError("TELEMETRY_DISABLED=1 is required before Cognee import")
    import cognee

    storage_root.mkdir(parents=True, exist_ok=True)
    cognee.config.system_root_directory(str(storage_root / "system"))
    cognee.config.data_root_directory(str(storage_root / "data"))
    cognee.config.set_llm_provider("ollama")
    cognee.config.set_llm_model("llama3.2:3b")
    cognee.config.set_llm_endpoint("http://127.0.0.1:11434")
    cognee.config.set_llm_api_key("ollama")
    cognee.config.set_embedding_provider("fastembed")
    cognee.config.set_embedding_model("BAAI/bge-small-en-v1.5")
    cognee.config.set_embedding_dimensions(384)
    return cognee


def _load_qwen(model_path: str) -> LocalQwenIO:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    )
    freeze_knowledge_model(model)
    return LocalQwenIO(
        model,
        tokenizer,
        embedding_batch_size=16,
        max_new_tokens=512,
        enable_thinking=False,
    )


def _task_contract(template: TaskTemplate, root: Path) -> AutonomousTaskContract:
    return AutonomousTaskContract(
        task_id=template.task_id,
        request=template.request,
        success_description="the private objective Python verifier exits successfully",
        workspace_root=root,
        readable_paths=("solution.py",),
        writable_paths=("solution.py",),
        verifier_command=(sys.executable, "-c", template.verifier_source),
        allow_mutation=True,
        max_steps=10,
        max_verifier_runs=4,
        verifier_timeout_seconds=20.0,
        max_file_bytes=32_768,
        max_observation_chars=8_192,
    )


async def _run_arm(
    arm: str,
    state_root: Path,
    checkpoint: Path,
    qwen: LocalQwenIO,
) -> dict[str, Any]:
    arm_root = state_root / arm
    journal = ConversationJournal(arm_root / "journal.jsonl")
    backend = HybridChunkCogneeProjectionBackend(
        f"angler_autonomous_apprentice_v1_{arm}",
        local_models_configured=True,
    )
    memory = SituatedMemory(backend, origin=MovingOriginIndex())
    learner = ActionTraceCompetenceLearner.from_checkpoint(
        checkpoint,
        memory,
        qwen,
        feature_spec=SituatedFeatureSpec(),
        device="cuda:1",
        dtype=torch.float32,
        use_moving_origin=arm != "moving_origin_removed",
    )
    initial_snapshot = learner.capture_state()
    initial_digest = learner.state_digest()
    policy = QwenJsonPolicy(qwen)
    recorder = NullRecorder() if arm == "memory_disabled" else SituatedEpisodeRecorder(journal, memory)
    apprentice = AutonomousApprentice(
        policy,
        learner,
        recorder,
        AtomicCompetenceStore(arm_root / "competence.pt"),
        experience=(lambda request: "") if arm == "memory_disabled" else learner.experience,
    )
    rows = []
    restart_verified = False
    for index, template in enumerate(TASKS):
        task_root = arm_root / "tasks" / template.task_id
        task_root.mkdir(parents=True, exist_ok=False)
        (task_root / "solution.py").write_text(template.initial_source, encoding="utf-8")
        before = learner.state_digest()
        started = time.monotonic()
        result = await apprentice.run(_task_contract(template, task_root))
        elapsed = time.monotonic() - started
        rows.append(
            {
                "task_id": template.task_id,
                "mechanism": template.mechanism,
                "phase": template.phase,
                "status": result.status,
                "action_receipts": len(result.receipts),
                "verifier_dispositions": [item.disposition for item in result.verifier_receipts],
                "verifier_runs": len(result.verifier_receipts),
                "steps": len(result.receipts) + len(result.verifier_receipts),
                "state_before": before,
                "state_after": learner.state_digest(),
                "wall_seconds": elapsed,
                "source_sha256": _sha256(task_root / "solution.py"),
            }
        )
        if arm == "state_reset":
            learner.restore_state(initial_snapshot)
        if arm == "full" and index == 3:
            snapshot = learner.capture_state()
            expected = learner.state_digest()
            replacement = ActionTraceCompetenceLearner.from_checkpoint(
                checkpoint,
                memory,
                qwen,
                feature_spec=SituatedFeatureSpec(),
                device="cuda:1",
                dtype=torch.float32,
            )
            replacement.restore_state(snapshot)
            restart_verified = replacement.state_digest() == expected
            learner = replacement
            apprentice = AutonomousApprentice(
                policy,
                learner,
                recorder,
                AtomicCompetenceStore(arm_root / "competence.pt"),
                experience=learner.experience,
            )
    transfer = [row for row in rows if row["phase"] == "transfer"]
    successful = [row for row in rows if row["status"] == "SUCCEEDED"]
    transfer_success = [row for row in transfer if row["status"] == "SUCCEEDED"]
    return {
        "arm": arm,
        "tasks": rows,
        "success_rate": len(successful) / len(rows),
        "transfer_success_rate": len(transfer_success) / len(transfer),
        "transfer_mean_steps": (
            statistics.mean(row["steps"] for row in transfer_success)
            if transfer_success
            else None
        ),
        "retry_to_success": any(
            "FAIL" in row["verifier_dispositions"]
            and row["verifier_dispositions"][-1] == "PASS"
            for row in rows
        ),
        "initial_state_digest": initial_digest,
        "terminal_state_digest": learner.state_digest(),
        "state_changed": learner.state_digest() != initial_digest,
        "journal_records": len(journal.records()),
        "origin_now": memory.origin.now,
        "policy_calls": policy.calls,
        "policy_raw_hashes": policy.raw_hashes,
        "restart_verified": restart_verified if arm == "full" else None,
    }


def _strictly_better(full: dict[str, Any], control: dict[str, Any]) -> bool:
    if full["transfer_success_rate"] > control["transfer_success_rate"]:
        return True
    if full["transfer_success_rate"] < control["transfer_success_rate"]:
        return False
    full_steps = full["transfer_mean_steps"]
    control_steps = control["transfer_mean_steps"]
    return (
        full_steps is not None
        and control_steps is not None
        and full_steps + 0.5 <= control_steps
    )


async def evaluate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("autonomous evaluation requires both workstation GPUs")
    result_path = Path(args.result)
    state_root = Path(args.state_root)
    if result_path.exists() or state_root.exists():
        raise RuntimeError("first-result output or state root already exists")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    _configure_cognee(state_root / "cognee")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.set_device(1)
    torch.cuda.reset_peak_memory_stats(1)
    started = time.monotonic()
    qwen = _load_qwen(args.model)
    foundation_before = foundation_tensor_digest(qwen.model)
    arm_results = []
    for arm in ARMS:
        arm_results.append(await _run_arm(arm, state_root, checkpoint, qwen))
    foundation_after = foundation_tensor_digest(qwen.model)
    if foundation_before != foundation_after:
        raise RuntimeError("frozen Qwen foundation changed")
    by_arm = {row["arm"]: row for row in arm_results}
    full = by_arm["full"]
    supported = (
        full["success_rate"] >= 0.75
        and full["retry_to_success"]
        and full["state_changed"]
        and full["restart_verified"]
        and full["journal_records"] > 0
        and _strictly_better(full, by_arm["state_reset"])
        and _strictly_better(full, by_arm["memory_disabled"])
    )
    payload = {
        "identity": IDENTITY,
        "classification": "BOUNDED_AUTONOMOUS_APPRENTICESHIP_SUPPORTED" if supported else "NOT_SUPPORTED",
        "frozen": {
            "seed": SEED,
            "task_count": len(TASKS),
            "mechanisms": sorted({task.mechanism for task in TASKS}),
            "arms": list(ARMS),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "model": args.model,
            "foundation_digest": foundation_before,
        },
        "arms": arm_results,
        "gate": {
            "full_success_at_least_0_75": full["success_rate"] >= 0.75,
            "failed_attempt_recovers": full["retry_to_success"],
            "persistent_state_changes": full["state_changed"],
            "restart_replays_exact_state": full["restart_verified"],
            "objective_episodes_persist": full["journal_records"] > 0,
            "better_than_state_reset": _strictly_better(full, by_arm["state_reset"]),
            "better_than_memory_disabled": _strictly_better(full, by_arm["memory_disabled"]),
        },
        "runtime": {
            "wall_seconds": time.monotonic() - started,
            "torch": torch.__version__,
            "python": sys.version,
            "qwen_device": "cuda:0",
            "angler_device": "cuda:1",
            "peak_cuda_bytes": {
                "cuda:0": torch.cuda.max_memory_allocated(0),
                "cuda:1": torch.cuda.max_memory_allocated(1),
            },
        },
        "nonclaims": [
            "The tasks are bounded synthetic software mechanisms, not arbitrary real projects.",
            "Objective verifier outcomes replace manual grading; they do not prove open-ended self-improvement.",
            "Qwen and the trained procedural core remain frozen; only plastic state and episodic memory change.",
            "No AGI, consciousness, cross-domain transfer, deployment, or unrestricted tool-use claim.",
        ],
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/opt/angler/models/Qwen3-4B")
    parser.add_argument("--checkpoint", default="/opt/angler/results/compositional-procedure-v6-action-match.pt")
    parser.add_argument("--state-root", default="/opt/angler/state/autonomous-apprentice-v1/evaluation-first")
    parser.add_argument("--result", default="/opt/angler/results/autonomous-apprentice-v1-evaluation.json")
    args = parser.parse_args()
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
