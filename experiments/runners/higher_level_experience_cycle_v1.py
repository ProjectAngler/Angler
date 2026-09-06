"""Run the first fully assembled higher-level Angler successor cycle.

This is one integration witness, not a component benchmark.  Both frozen models,
Cognee, Moving Origin, semantic-plus-utility memory, objective feedback,
reflection, refinement, consolidation, situated world/self/focus state, and the
append-only trace participate in the same live transaction.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any

from angler.cognition.contracts import (
    CognitiveEpisode,
    CognitiveMemoryKind,
    CognitiveMemoryRecord,
    CognitiveRelation,
    EpistemicStatus,
    ProspectiveCommitment,
    RelationType,
)
from angler.memory.cognee_acquisition_adapter import CogneeAcquisitionAdapter
from angler.memory.cognee_subprocess_bindings import CogneeSubprocessBindings
from angler.memory.cognee_worker_protocol import CogneeWorkerScope
from angler.memory.cognitive_acquisition import CognitiveAcquisition, CognitiveGraphProjectionV2
from angler.memory.moving_origin import MovingOriginIndex
from angler.runtime.higher_level_experience_cycle import (
    ConsequenceVector,
    ExecutionReceipt,
    MemoryCandidate,
    StructuredExperience,
    TraceLedger,
    UtilityMemory,
    canonical_bytes,
    content_ref,
)


IDENTITY = "angler.higher-level-experience-cycle.v1-r7-smoke"
ROOT = Path("/opt/angler/src/angler")
RESULT_ROOT = Path("/opt/angler/results/higher-level-experience-cycle-v1-r7")
STATE_ROOT = Path("/opt/angler/state/project-angler/higher-level-experience-cycle-v1-r7")
RESULT_PATH = RESULT_ROOT / "result.json"
MODEL_17B = Path("/opt/angler/models/Qwen3-1.7B")
MODEL_14B = Path("/opt/angler/models/Qwen3-14B-NVFP4")
GPU_5080 = "GPU-df4bb978-e75f-08a0-6660-2b9ed69ee8ca"
GPU_5070 = "GPU-d9dd1ae0-f65d-ef22-f924-2c3e9c976c1e"
TRT_IMAGE = "nvcr.io/nvidia/tensorrt-llm/release:1.2.1"
TASK = (
    "Four people must cross a bridge at night using one torch. Their crossing "
    "times are 1, 2, 7, and 10 minutes. At most two people cross at once, a pair "
    "moves at the slower person's speed, and every crossing must carry the torch. "
    "Find the minimum total time. Explain the schedule and end with `FINAL: N`, "
    "where N is the integer number of minutes."
)
EXPECTED_FINAL = 17  # Held only by the post-receipt objective verifier.
MAX_WALL_SECONDS = 1800.0
MIN_FREE_MIB = 2048


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _preserve_stage(stage: str, value: object) -> None:
    with (RESULT_ROOT / "stage-evidence.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(_json({"stage": stage, "value": value}) + "\n")


def _gpu_free_mib() -> dict[str, int]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        fields[0].strip(): int(fields[1].strip())
        for fields in (line.split(",") for line in completed.stdout.splitlines())
    }


def _extract_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start < 0:
        raise ValueError("model output contains no JSON object")
    depth, quoted, escaped = 0, False, False
    for index in range(start, len(text)):
        character = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                value = json.loads(text[start : index + 1])
                if type(value) is not dict:
                    raise ValueError("model JSON output is not an object")
                return value
    raise ValueError("model JSON object is incomplete")


def _extract_consolidation(text: str) -> dict[str, Any]:
    keys = "LESSON|WORLD|SELF|FOCUS|OPEN"
    matches = re.findall(
        rf"(?:^|\s)({keys})=(.*?)(?=\s(?:{keys})=|$)",
        text.strip(),
        flags=re.DOTALL,
    )
    fields = {key: value.strip(" |\n\t") for key, value in matches}
    expected = {"LESSON", "WORLD", "SELF", "FOCUS", "OPEN"}
    if set(fields) != expected or any(not value for value in fields.values()):
        raise ValueError("consolidation line schema differs")
    unfinished = [] if fields["OPEN"].upper() == "NONE" else [fields["OPEN"]]
    return {
        "lesson": fields["LESSON"],
        "world_model": fields["WORLD"],
        "self_model": fields["SELF"],
        "focus": fields["FOCUS"],
        "unfinished_patterns": unfinished,
    }


def _experience_worker() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(20260902)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_17B, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_17B,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("ANGLER_WORKER=" + _json({"ready": True, "device": str(next(model.parameters()).device)}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        messages = [
            {"role": "system", "content": request["system"]},
            {"role": "user", "content": request["user"]},
        ]
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        ).to("cuda:0")
        input_tokens = encoded["input_ids"]
        if input_tokens.shape[-1] > 512:
            raise RuntimeError("experience-model input exceeded 512 tokens")
        started = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=int(request.get("max_new_tokens", 256)),
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        output = tokenizer.decode(generated[0, input_tokens.shape[-1] :], skip_special_tokens=True)
        result = {
            "id": request["id"],
            "text": output,
            "input_tokens": int(input_tokens.shape[-1]),
            "output_tokens": int(generated.shape[-1] - input_tokens.shape[-1]),
            "seconds": time.monotonic() - started,
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
        }
        print("ANGLER_WORKER=" + _json(result), flush=True)
    return 0


CORTEX_WORKER_CODE = r'''
import json, sys, time
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig

llm = LLM(
    model="/model",
    max_batch_size=1,
    max_seq_len=12544,
    max_num_tokens=2048,
    enable_chunked_prefill=True,
    kv_cache_config=KvCacheConfig(
        enable_block_reuse=False,
        enable_partial_reuse=False,
        max_tokens=12544,
    ),
    cuda_graph_config={"batch_sizes": [1], "max_batch_size": 1},
)
print("ANGLER_CORTEX=" + json.dumps({"ready": True}, sort_keys=True), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    started = time.monotonic()
    output = llm.generate(
        request["prompt"],
        SamplingParams(max_tokens=request.get("max_tokens", 256), temperature=0.0, seed=20260902),
        use_tqdm=False,
    )
    text = output.outputs[0].text
    result = {
        "id": request["id"],
        "text": text,
        "seconds": time.monotonic() - started,
        "output_tokens": len(output.outputs[0].token_ids),
    }
    print("ANGLER_CORTEX=" + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
'''


CORTEX_ONESHOT_CODE = r'''
import base64, json, sys, time, torch
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig
prompt = base64.b64decode(sys.argv[1]).decode("utf-8")
request_id = sys.argv[2]
llm = LLM(
    model="/model", max_batch_size=1, max_seq_len=12544, max_num_tokens=2048,
    enable_chunked_prefill=True,
    kv_cache_config=KvCacheConfig(enable_block_reuse=False, enable_partial_reuse=False, max_tokens=12544),
    cuda_graph_config={"batch_sizes": [1], "max_batch_size": 1},
)
started = time.monotonic()
output = llm.generate(prompt, SamplingParams(max_tokens=256, temperature=0.0, seed=20260902), use_tqdm=False)
free_bytes, total_bytes = torch.cuda.mem_get_info()
result = {
    "id": request_id, "text": output.outputs[0].text,
    "seconds": time.monotonic() - started,
    "output_tokens": len(output.outputs[0].token_ids),
    "settled_free_mib": int(free_bytes // (1024 * 1024)),
    "total_mib": int(total_bytes // (1024 * 1024)),
}
print("ANGLER_CORTEX=" + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
'''


def _run_cortex_once(prompt: str, stage: str) -> dict[str, Any]:
    encoded_prompt = base64.b64encode(prompt.encode("utf-8")).decode("ascii")
    completed = subprocess.run(
        [
            "/usr/bin/sudo", "-n", "/usr/bin/docker", "run", "--rm",
            "--network", "none", "--ipc", "host", "--gpus", "device=0",
            "-e", "CUDA_VISIBLE_DEVICES=0", "-e", "TLLM_LOG_LEVEL=ERROR",
            "-e", "PYTHONWARNINGS=ignore", "-v", f"{MODEL_14B}:/model:ro",
            TRT_IMAGE, "python", "-u", "-c", CORTEX_ONESHOT_CODE,
            encoded_prompt, stage,
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    (RESULT_ROOT / f"cortex-{stage}-worker.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{stage} cortex process exited with {completed.returncode}")
    marker = "ANGLER_CORTEX="
    lines = [line for line in completed.stdout.splitlines() if line.startswith(marker)]
    if len(lines) != 1:
        raise RuntimeError(f"{stage} cortex result marker differs")
    result = json.loads(lines[0][len(marker) :])
    if type(result) is not dict or result.get("id") != stage:
        raise RuntimeError(f"{stage} cortex result identity differs")
    return result


class JsonLineWorker:
    def __init__(self, process: subprocess.Popen[str], prefix: str, stderr_stream: Any) -> None:
        self.process, self.prefix, self.stderr_stream = process, prefix, stderr_stream

    @classmethod
    def experience(cls) -> "JsonLineWorker":
        environment = {
            **os.environ,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": GPU_5070,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": str(ROOT / "src"),
        }
        log = (RESULT_ROOT / "experience-worker.stderr.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            ["/opt/angler/venvs/angler/bin/python", "-u", str(Path(__file__).resolve()), "--experience-worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            bufsize=1,
            env=environment,
        )
        worker = cls(process, "ANGLER_WORKER=", log)
        ready = worker._read(420.0)
        if ready.get("ready") is not True or ready.get("device") != "cuda:0":
            raise RuntimeError("experience worker readiness differs")
        return worker

    @classmethod
    def cortex(cls, stage: str) -> "JsonLineWorker":
        if stage not in {"initial", "refinement"}:
            raise ValueError("cortex stage differs")
        log = (RESULT_ROOT / f"cortex-{stage}-worker.stderr.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            [
                "/usr/bin/sudo", "-n", "/usr/bin/docker", "run", "-i", "--rm",
                "--network", "none", "--ipc", "host", "--gpus", "device=0",
                "-e", "CUDA_VISIBLE_DEVICES=0", "-e", "TLLM_LOG_LEVEL=ERROR",
                "-e", "PYTHONWARNINGS=ignore", "-v", f"{MODEL_14B}:/model:ro",
                TRT_IMAGE, "python", "-u", "-c", CORTEX_WORKER_CODE,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            bufsize=1,
        )
        worker = cls(process, "ANGLER_CORTEX=", log)
        ready = worker._read(600.0)
        if ready.get("ready") is not True:
            raise RuntimeError("cortex worker readiness differs")
        return worker

    def _read(self, timeout: float) -> dict[str, Any]:
        assert self.process.stdout is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"worker exited with {self.process.returncode}")
            # Both workers flush one bounded result line; model initialization is
            # bounded by the outer live-run timer.
            line = self.process.stdout.readline()
            if not line:
                continue
            if line.startswith(self.prefix):
                value = json.loads(line[len(self.prefix) :])
                if type(value) is not dict:
                    raise RuntimeError("worker response is not an object")
                return value
        raise TimeoutError("worker response timed out")

    def request(self, payload: dict[str, Any], timeout: float = 300.0) -> dict[str, Any]:
        if self.process.poll() is not None:
            raise RuntimeError("worker is not running")
        assert self.process.stdin is not None
        self.process.stdin.write(_json(payload) + "\n")
        self.process.stdin.flush()
        result = self._read(timeout)
        if result.get("id") != payload.get("id"):
            raise RuntimeError("worker response identity differs")
        return result

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.stderr_stream.close()


class GpuProbe:
    def __init__(self) -> None:
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            command = [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            stamp = time.time()
            if completed.returncode == 0:
                for line in completed.stdout.splitlines():
                    fields = [item.strip() for item in line.split(",")]
                    if len(fields) == 8:
                        self.samples.append({
                            "time": stamp, "index": int(fields[0]), "uuid": fields[1], "name": fields[2],
                            "total_mib": int(fields[3]), "used_mib": int(fields[4]), "free_mib": int(fields[5]),
                            "utilization_percent": int(fields[6]), "power_w": float(fields[7]),
                        })
            self._stop.wait(0.25)

    def finish(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=2)
        summary: dict[str, Any] = {}
        for index in (0, 1):
            rows = [item for item in self.samples if item["index"] == index]
            if rows:
                summary[str(index)] = {
                    "uuid": rows[0]["uuid"], "name": rows[0]["name"],
                    "samples": len(rows), "peak_used_mib": max(item["used_mib"] for item in rows),
                    "minimum_free_mib": min(item["free_mib"] for item in rows),
                    "peak_utilization_percent": max(item["utilization_percent"] for item in rows),
                    "peak_power_w": max(item["power_w"] for item in rows),
                }
        return summary


def _worker_prompt(kind: str, payload: dict[str, Any]) -> tuple[str, str]:
    if kind == "experience":
        system = (
            "You are a frozen structured-experience generator, not the public answerer and not an outcome judge. "
            "Use the request, retrieved memories, and temporal context to propose how another model should reason. "
            "Return only one JSON object with exactly these keys: interpretation, process_action, strategy, "
            "predicted_consequence, checks, uncertainty, world_model, self_model, focus, unfinished_patterns. "
            "process_action is one of DECOMPOSE, RETRIEVE, PREDICT, TRY, VERIFY, REVISE, ASK, ACT, STOP; "
            "checks and unfinished_patterns are arrays; uncertainty is 0 through 1. Use at most 18 words per string, "
            "exactly two checks, and at most two unfinished patterns so the complete object fits in 256 tokens. "
            "Do not solve the task or state its final answer."
        )
    elif kind == "reflection":
        system = (
            "You are a frozen experience-reflection model. Given an already executed public response and objective "
            "feedback, identify what reasoning process was useful or weak. Return only JSON with keys analysis, "
            "revision, retained_principle. Use at most 16 words in each value so the complete object fits in 192 tokens. "
            "Do not invent hidden verifier targets or claim feelings."
        )
    elif kind == "consolidation":
        system = (
            "You are a frozen consolidation model. Convert the complete attempt-feedback-reflection-refinement history "
            "into reusable, fallible memory and changed situated state. Return exactly one line in this format: "
            "LESSON=<text> || WORLD=<text> || SELF=<text> || FOCUS=<text> || OPEN=<text or NONE>. "
            "Use at most eight words in each field. Do not add JSON, lists, explanations, consciousness, or feeling claims."
        )
    else:
        raise ValueError("unknown experience operation")
    return system, _json(payload)


def _chat_prompt(system: str, user: str) -> str:
    return (
        "<|im_start|>system\n" + system + "<|im_end|>\n"
        "<|im_start|>user\n" + user + "<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _verify(receipt: ExecutionReceipt, seconds: float) -> tuple[ConsequenceVector, dict[str, Any]]:
    matches = re.findall(r"FINAL:\s*(-?\d+)", receipt.response, flags=re.IGNORECASE)
    parsed = int(matches[-1]) if matches else None
    accepted = parsed == EXPECTED_FINAL
    diagnostics = {
        "accepted": accepted,
        "format_present": parsed is not None,
        "explanation_present": len(receipt.response.split()) >= 20,
        "parsed_final": parsed,
        "expected_value_disclosed_to_models": False,
    }
    consequence = ConsequenceVector(
        objective_progress=1.0 if accepted else (0.35 if parsed is not None else 0.0),
        constraint_satisfaction=1.0 if parsed is not None else 0.0,
        prediction_error=0.0 if accepted else 0.8,
        information_gain=0.35 if accepted else 0.8,
        evidence_quality=0.8 if diagnostics["explanation_present"] else 0.2,
        reuse_value=0.6,
        cost=min(1.0, seconds / 30.0),
        safety=1.0,
        human_feedback=0.0,
    )
    return consequence, diagnostics


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _canonical_acquisition(
    *,
    content: str,
    ordinal: int,
    predecessor_acquisition_ref: str | None,
    prior_record_ref: str | None,
    outcome: str,
) -> CognitiveAcquisition:
    label = f"{IDENTITY}:{ordinal}"
    support = (_digest(label + ":support"),)
    parent = _digest(label + ":parent")
    proposals = (
        "Preserve the observed experience.",
        "Do not preserve the observed experience.",
    )
    commitment = ProspectiveCommitment(
        parent_event_ref=None,
        task_id=label,
        candidate_index=0,
        candidate_trace=proposals[0],
        predicted_score=0.0,
        uncertainty=0.5,
        horizon=1,
        competence_state_digest=parent,
    )
    episode = CognitiveEpisode(
        task_id=label,
        request="Preserve this fallible private-research experience with provenance.",
        recalled_refs=support,
        proposals=proposals,
        selected_index=0,
        commitment=commitment,
        response=content,
        observations=("Canonical experience record constructed after observation.",),
        outcome="failure" if outcome == "not_accepted" else "success",
        feedback_text="Objective disposition is recorded by the enclosing cycle.",
        feedback_source_ref=_digest(label + ":feedback"),
        parent_state_digest=parent,
        child_state_digest=_digest(label + ":child"),
        model_ref=_digest("qwen3-14b-nvfp4"),
        encoder_ref=_digest("qwen3-1.7b-structured-experience"),
        supporting_evidence_refs=support,
    )
    relations = (
        ()
        if prior_record_ref is None
        else (CognitiveRelation(RelationType.DERIVED_FROM, prior_record_ref),)
    )
    record = CognitiveMemoryRecord(
        kind=CognitiveMemoryKind.EPISODIC,
        epistemic_status=EpistemicStatus.OBSERVED,
        content=content,
        provenance_refs=tuple(sorted((episode.episode_ref, _digest(label + ":evidence")))),
        visibility="LEARNER_VISIBLE",
        producer_id="ANG-WORK-RUNTIME-HIGHER-LEVEL-EXPERIENCE-CYCLE-V1-001",
        producer_checkpoint_ref=_digest("higher-level-cycle-r1-runner"),
        competence_ref=_digest("higher-level-cycle-r1-frozen-models"),
        acquired_ordinal=ordinal,
        world_valid_from=ordinal,
        relations=relations,
    )
    return CognitiveAcquisition.from_source(
        episode,
        ordinal=ordinal,
        predecessor_acquisition_ref=predecessor_acquisition_ref,
        record=record,
    )


async def _run_live() -> dict[str, Any]:
    started_wall = time.time()
    started = time.monotonic()
    trace = TraceLedger(maximum=32)
    utility = UtilityMemory(alpha=0.25)
    origin = MovingOriginIndex()
    model_inputs: list[dict[str, Any]] = []
    experience_outputs: list[dict[str, Any]] = []
    cortex_outputs: list[dict[str, Any]] = []
    canonical_memory: dict[str, str] = {}
    workers: list[JsonLineWorker] = []
    scope = CogneeWorkerScope(
        dataset_name="higher-level-experience-cycle-v1-r7",
        tenant_name="higher-level-experience-cycle-v1-r7-tenant",
        node_set_name="higher-level-experience-cycle-v1-r7-records",
        state_root=str(STATE_ROOT / "cognee"),
    )
    cognee: CogneeSubprocessBindings | None = None
    try:
        cognee = await CogneeSubprocessBindings.start(scope=scope, timeout_seconds=120.0)
        os.environ.setdefault("TELEMETRY_DISABLED", "1")
        cognee_adapter = CogneeAcquisitionAdapter(
            bindings=cognee,
            tenant_id=cognee.tenant_id,
            dataset_id=cognee.dataset_id,
            dataset_name=cognee.dataset_name,
            node_set_name=cognee.node_set_name,
            local_embeddings_configured=True,
            external_embedding_calls_authorized=False,
            telemetry_authorized=False,
        )
        # Match the already-qualified live acquisition path: explicitly bind a
        # fresh empty namespace before the first canonical projection.
        await cognee_adapter.forget_namespace()
        seeds = (
            "For constrained search, make the state transitions explicit, compare complete candidate schedules, and verify the final constraint rather than trusting familiarity.",
            "When an objective check rejects an answer, preserve the attempt, distinguish reasoning failure from output-format failure, and revise the weakest assumption.",
            "A useful reasoning trace states its prediction before action and turns both success and failure into fallible reusable guidance with provenance.",
        )
        previous_record_ref: str | None = None
        predecessor_acquisition_ref: str | None = None
        projected = []
        for ordinal, content in enumerate(seeds):
            acquisition = _canonical_acquisition(
                content=content,
                ordinal=ordinal,
                predecessor_acquisition_ref=predecessor_acquisition_ref,
                prior_record_ref=previous_record_ref,
                outcome="seeded_observation",
            )
            projection = CognitiveGraphProjectionV2.from_acquisition(acquisition)
            backend_ref = await cognee_adapter.project(projection)
            record_ref = acquisition.record.record_ref
            projection_ref = projection.projection_ref
            origin.append(record_ref, projection_ref)
            canonical_memory[record_ref] = content
            projected.append({"record_ref": record_ref, "projection_ref": projection_ref, "backend_ref": backend_ref})
            previous_record_ref = record_ref
            predecessor_acquisition_ref = acquisition.acquisition_ref
        trace.append("COGNEE_SEED_PROJECTION", (), projected)
        _preserve_stage("cognee_seed_projection", projected)

        raw_search = await cognee.search_references(TASK, limit=4)
        entries = raw_search[0]["search_result"]
        semantic = tuple(
            MemoryCandidate(
                record_ref=entry["payload"]["record_ref"],
                content=canonical_memory[entry["payload"]["record_ref"]],
                semantic_distance=float(entry["score"]),
                acquired_ordinal=origin.position(entry["payload"]["record_ref"]).acquired_ordinal,
            )
            for entry in entries
        )
        trace.append("SEMANTIC_RECALL", (content_ref(TASK),), [asdict(item) for item in semantic])
        selected = utility.rank(semantic)
        trace.append("UTILITY_RERANK", tuple(item.record_ref for item in semantic), [asdict(item) for item in selected])
        temporal = {
            "now": origin.now,
            "last_event_ref": origin.recent(1).event_refs[0] if origin.size else None,
            "positions": {item.record_ref: asdict(origin.position(item.record_ref)) for item in selected},
        }
        trace.append("MOVING_ORIGIN_CONTEXT", tuple(item.record_ref for item in selected), temporal)

        experience_worker = JsonLineWorker.experience()
        workers.append(experience_worker)
        experience_payload = {
            "request": TASK,
            "retrieved_memories": [
                {"content": item.content}
                for item in selected
            ],
            "temporal_context": {
                "now": temporal["now"],
                "last_event_available": temporal["last_event_ref"] is not None,
            },
        }
        system, user = _worker_prompt("experience", experience_payload)
        model_inputs.append({"model": "Qwen3-1.7B", "stage": "experience", "system": system, "user": user})
        raw_experience = experience_worker.request({"id": "experience", "system": system, "user": user, "max_new_tokens": 256})
        experience_outputs.append(raw_experience)
        _preserve_stage("experience_raw", raw_experience)
        experience_mapping = _extract_json(raw_experience["text"])
        experience = StructuredExperience.from_mapping(experience_mapping)
        trace.append("STRUCTURED_EXPERIENCE", tuple(item.record_ref for item in selected), asdict(experience))

        cortex_user = _json({
            "request": TASK,
            "retrieved_memories": [{"record_ref": item.record_ref, "content": item.content} for item in selected],
            "temporal_context": temporal,
            "structured_experience": asdict(experience),
        })
        cortex_system = (
            "You are the frozen public reasoning cortex inside an attributed research cycle. Solve the user's request. "
            "Retrieved memories and structured experience are fallible process guidance, never answer authority. "
            "Do not claim feelings or consciousness. Follow the requested FINAL format and keep the entire answer under 120 words."
        )
        initial_prompt = _chat_prompt(cortex_system, cortex_user)
        model_inputs.append({"model": "Qwen3-14B-NVFP4", "stage": "initial", "prompt": initial_prompt})
        raw_initial = _run_cortex_once(initial_prompt, "initial")
        cortex_outputs.append(raw_initial)
        _preserve_stage("initial_cortex_raw", raw_initial)
        initial_receipt = ExecutionReceipt(content_ref(TASK), raw_initial["text"], "COMPLETED")
        trace.append("INITIAL_CORTEX_RECEIPT", (content_ref(TASK), content_ref(asdict(experience))), asdict(initial_receipt))

        initial_consequence, initial_diagnostics = _verify(initial_receipt, float(raw_initial["seconds"]))
        trace.append("INITIAL_OBJECTIVE_FEEDBACK", (initial_receipt.receipt_ref,), {"consequence": asdict(initial_consequence), "diagnostics": initial_diagnostics})

        reflection_payload = {
            "request": TASK,
            "strategy": experience.strategy,
            "prediction": experience.predicted_consequence,
            "public_response": initial_receipt.response,
            "objective_feedback": {
                "accepted": initial_diagnostics["accepted"],
                "format_present": initial_diagnostics["format_present"],
                "objective_progress": initial_consequence.objective_progress,
                "prediction_error": initial_consequence.prediction_error,
            },
        }
        system, user = _worker_prompt("reflection", reflection_payload)
        model_inputs.append({"model": "Qwen3-1.7B", "stage": "reflection", "system": system, "user": user})
        raw_reflection = experience_worker.request({"id": "reflection", "system": system, "user": user, "max_new_tokens": 192})
        experience_outputs.append(raw_reflection)
        _preserve_stage("reflection_raw", raw_reflection)
        reflection = _extract_json(raw_reflection["text"])
        trace.append("REFLECTION", (initial_receipt.receipt_ref, content_ref(asdict(initial_consequence))), reflection)

        refinement_user = _json({
            "request": TASK,
            "prior_response": initial_receipt.response,
            "objective_feedback_without_hidden_target": {k: v for k, v in initial_diagnostics.items() if k != "parsed_final"},
            "reflection": reflection,
            "structured_experience": asdict(experience),
        })
        refinement_system = (
            "You are the same frozen public reasoning cortex performing one attributable refinement. "
            "Use only the public request, prior response, target-free objective feedback, and reflection. "
            "Re-solve and return a complete response under 120 words ending in FINAL: N."
        )
        refinement_prompt = _chat_prompt(refinement_system, refinement_user)
        model_inputs.append({"model": "Qwen3-14B-NVFP4", "stage": "refinement", "prompt": refinement_prompt})
        raw_refinement = _run_cortex_once(refinement_prompt, "refinement")
        cortex_outputs.append(raw_refinement)
        _preserve_stage("refinement_cortex_raw", raw_refinement)
        refinement_receipt = ExecutionReceipt(content_ref({"task": TASK, "stage": "refinement"}), raw_refinement["text"], "COMPLETED")
        trace.append("REFINEMENT_CORTEX_RECEIPT", (initial_receipt.receipt_ref, content_ref(reflection)), asdict(refinement_receipt))
        final_consequence, final_diagnostics = _verify(refinement_receipt, float(raw_refinement["seconds"]))
        trace.append("FINAL_OBJECTIVE_FEEDBACK", (refinement_receipt.receipt_ref,), {"consequence": asdict(final_consequence), "diagnostics": final_diagnostics})

        consolidation_payload = {
            "request": TASK,
            "initial_strategy": experience.strategy,
            "initial_feedback": {
                "accepted": initial_diagnostics["accepted"],
                "objective_progress": initial_consequence.objective_progress,
            },
            "reflection": reflection,
            "refined_response": refinement_receipt.response,
            "final_feedback": {
                "accepted": final_diagnostics["accepted"],
                "objective_progress": final_consequence.objective_progress,
                "prediction_error": final_consequence.prediction_error,
            },
        }
        system, user = _worker_prompt("consolidation", consolidation_payload)
        model_inputs.append({"model": "Qwen3-1.7B", "stage": "consolidation", "system": system, "user": user})
        raw_consolidation = experience_worker.request({"id": "consolidation", "system": system, "user": user, "max_new_tokens": 96})
        experience_outputs.append(raw_consolidation)
        _preserve_stage("consolidation_raw", raw_consolidation)
        consolidation = _extract_consolidation(raw_consolidation["text"])
        required = {"lesson", "world_model", "self_model", "focus", "unfinished_patterns"}
        if set(consolidation) != required or type(consolidation["unfinished_patterns"]) is not list:
            raise ValueError("consolidation schema differs")
        consolidation_acquisition = _canonical_acquisition(
            content=_json(consolidation),
            ordinal=origin.size,
            predecessor_acquisition_ref=predecessor_acquisition_ref,
            prior_record_ref=previous_record_ref,
            outcome="accepted" if final_diagnostics["accepted"] else "not_accepted",
        )
        consolidation_projection = CognitiveGraphProjectionV2.from_acquisition(
            consolidation_acquisition
        )
        consolidation_backend_ref = await cognee_adapter.project(
            consolidation_projection
        )
        consolidation_ref = consolidation_acquisition.record.record_ref
        consolidation_projection_ref = consolidation_projection.projection_ref
        origin.append(consolidation_ref, consolidation_projection_ref)
        canonical_memory[consolidation_ref] = _json(consolidation)
        trace.append("COGNEE_CONSOLIDATION", (refinement_receipt.receipt_ref,), {
            "record_ref": consolidation_ref, "projection_ref": consolidation_projection_ref,
            "backend_ref": consolidation_backend_ref, "content": consolidation,
        })
        utility.update(tuple(item.record_ref for item in selected), final_consequence)
        trace.append("UTILITY_UPDATE", tuple(item.record_ref for item in selected), utility.snapshot())
        next_temporal = {
            "now": origin.now,
            "event_ref": consolidation_ref,
            "position": asdict(origin.position(consolidation_ref)),
        }
        trace.append("MOVING_ORIGIN_ADVANCE", (consolidation_ref,), next_temporal)

        changed_state = {
            "world_model": consolidation["world_model"],
            "self_model": consolidation["self_model"],
            "focus": consolidation["focus"],
            "unfinished_patterns": consolidation["unfinished_patterns"],
            "memory_head": consolidation_ref,
            "temporal_now": origin.now,
            "utility": utility.snapshot(),
        }
        trace.append("CHANGED_SHARED_STATE", (consolidation_ref,), changed_state)
        current_free = _gpu_free_mib()
        settled_gpu_free_mib = {
            "0": min(int(raw_initial["settled_free_mib"]), int(raw_refinement["settled_free_mib"])),
            "1": current_free["1"],
        }
        _preserve_stage("settled_gpu_free_mib", settled_gpu_free_mib)
        hidden_literal = str(EXPECTED_FINAL)
        pre_receipt_inputs = model_inputs[:2]
        # The task necessarily includes the raw times and can independently imply
        # 17; leakage evidence therefore checks for an explicit expected/verifier
        # field, not an unavoidable numeral substring.
        leakage = any("expected_value" in _json(item) or "verifier_target" in _json(item) for item in pre_receipt_inputs)
        return {
            "identity": IDENTITY,
            "status": "PASS",
            "claim_scope": "assembled technical integration only",
            "started_unix": started_wall,
            "wall_seconds": time.monotonic() - started,
            "task": TASK,
            "expected_final_private_to_verifier": EXPECTED_FINAL,
            "components": {
                "cortex": {"model": "nvidia/Qwen3-14B-NVFP4", "revision": "bc39319a4dc265d9bbb9a9731bc52c4988d9ece7", "runtime": "TensorRT-LLM 1.2.1", "physical_gpu": 0},
                "experience_model": {"model": "Qwen/Qwen3-1.7B", "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", "frozen": True, "physical_gpu": 1},
                "cognee": {"version": "1.5.3", "dataset_id": cognee.dataset_id, "scope": asdict(scope)},
                "moving_origin": "angler.moving-origin.v1",
                "utility": "MemRL-style post-semantic learned utility",
                "learning_flow": "attempt-feedback-reflection-refinement-consolidation",
                "trace": "Agent-Lightning-style append-only attribution",
                "seam": {"use": "mechanism-inspired private research evaluation", "repository_commit": "66a8d7fdf5b6ae0e835d972de28ea544c448ad7f", "source_copied": False},
            },
            "semantic_candidates": [asdict(item) for item in semantic],
            "selected_memories": [asdict(item) for item in selected],
            "temporal_before": temporal,
            "structured_experience": asdict(experience),
            "initial_receipt": asdict(initial_receipt),
            "initial_feedback": {"consequence": asdict(initial_consequence), "diagnostics": initial_diagnostics},
            "reflection": reflection,
            "refinement_receipt": asdict(refinement_receipt),
            "final_feedback": {"consequence": asdict(final_consequence), "diagnostics": final_diagnostics},
            "consolidation": consolidation,
            "changed_state": changed_state,
            "settled_gpu_free_mib": settled_gpu_free_mib,
            "moving_origin_snapshot": origin.snapshot(),
            "model_inputs": model_inputs,
            "experience_model_outputs": experience_outputs,
            "cortex_outputs": cortex_outputs,
            "pre_receipt_hidden_target_field_leakage": leakage,
            "trace": {"ledger_ref": trace.ledger_ref, "spans": [asdict(span) | {"span_ref": span.span_ref} for span in trace.spans]},
            "nonclaims": ["reasoning improvement", "continual learning", "generalization", "autonomy", "identity", "feelings", "consciousness", "AGI", "system readiness"],
        }
    finally:
        for worker in reversed(workers):
            worker.close()
        if cognee is not None:
            await cognee.close()


def _preflight() -> dict[str, Any]:
    if not MODEL_17B.is_dir() or not MODEL_14B.is_dir():
        raise RuntimeError("a frozen local model path is missing")
    if not (ROOT / "experiments/manifests/higher-level-experience-cycle-v1-r7.json").is_file():
        raise RuntimeError("frozen manifest is missing")
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    rows = [line.split(",") for line in completed.stdout.splitlines()]
    if len(rows) != 2 or rows[0][1].strip() != GPU_5080 or rows[1][1].strip() != GPU_5070:
        raise RuntimeError("physical GPU order or identity differs")
    if any(int(row[2].strip()) < MIN_FREE_MIB for row in rows):
        raise RuntimeError("a GPU lacks minimum launch headroom")
    return {
        "identity": IDENTITY,
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "gpu_rows": [[item.strip() for item in row] for row in rows],
        "result_absent": not RESULT_PATH.exists(),
        "state_absent": not STATE_ROOT.exists(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experience-worker", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--live", action="store_true")
    arguments = parser.parse_args()
    if arguments.experience_worker:
        return _experience_worker()
    if arguments.preflight:
        print(json.dumps(_preflight(), indent=2, sort_keys=True))
        return 0
    if not arguments.live:
        parser.error("choose --preflight or --live")
    RESULT_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    if RESULT_PATH.exists() or STATE_ROOT.exists():
        raise RuntimeError("live identity has existing result or state; do not rerun it")
    probe = GpuProbe()
    probe.start()
    started = time.monotonic()
    try:
        result = asyncio.run(asyncio.wait_for(_run_live(), timeout=MAX_WALL_SECONDS))
    except BaseException as exc:
        result = {
            "identity": IDENTITY,
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "wall_seconds": time.monotonic() - started,
        }
    finally:
        gpu_summary = probe.finish()
    result["gpu_summary"] = gpu_summary
    settled = result.get("settled_gpu_free_mib", {})
    result["resource_gate_pass"] = (
        type(settled) is dict
        and set(settled) == {"0", "1"}
        and all(type(value) is int and value >= MIN_FREE_MIB for value in settled.values())
    )
    if result.get("status") == "PASS" and not result["resource_gate_pass"]:
        result["status"] = "FAIL"
        result["error"] = "GPU headroom gate failed"
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    RESULT_PATH.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
