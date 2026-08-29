from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import unittest
from unittest import mock


os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

from experiments.evaluators import phase6_v19_paired_graph_context_recovery as recovery
from experiments.runners import phase6_software_pipeline_reconstruction as v12
from experiments.runners import phase6_v12_champion_paired_graph_context as v19


_SOURCE_CHECKPOINT = Path(
    "/opt/angler/results/phase6-software-pipeline-reconstruction-v12-conflict.pt"
)
_TERMINAL_CHECKPOINT = Path(
    "/opt/angler/results/"
    "phase6-software-pipeline-reconstruction-v19-paired-graph-context.pt"
)
_TERMINAL_CHECKPOINT_SHA256 = (
    "10BB6BAC9BD83F7F4EE0ABF2846CE4133D2133790C2B55113C9044930D2EBC7F"
)
_TERMINAL_SYSTEM_DIGEST = (
    "sha256:99712cfbc24140703203561f3ca42d904752aae92c8ec8d637128f7fe93bebc6"
)
_TERMINAL_MUTABLE_DIGEST = (
    "sha256:9cb6c11f5ff05fe75737227599094378cdacc9d914a3d558548780b26f7735ed"
)
_TERMINAL_OPTIMIZER_DIGEST = (
    "sha256:662fd334ecf56f0120e1b3023598099d7929289821df4a72b54a2ab74e83a388"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _inputs(*, stored_count: int = 3) -> tuple[torch.Tensor, ...]:
    code = torch.linspace(-0.4, 0.4, 32)
    query_codes = torch.stack((code, code.clone()))
    graph = torch.zeros((5, 5), dtype=torch.bool)
    graph[0, 1] = True
    graph[1, 3] = True
    graph[3, 4] = True
    query_graphs = torch.stack((graph, graph.clone()))
    mask = torch.tensor((True, True, False, True, True))
    query_masks = torch.stack((mask, mask.clone()))
    stored_contexts = torch.stack(
        tuple(torch.roll(code, shifts=index) for index in range(stored_count))
    )
    stored_graphs = torch.stack(
        tuple(graph.clone() for _ in range(stored_count))
    )
    stored_masks = torch.stack(tuple(mask.clone() for _ in range(stored_count)))
    return (
        query_codes,
        query_graphs,
        query_masks,
        stored_contexts,
        stored_graphs,
        stored_masks,
    )


class InjectedEvaluationError(RuntimeError):
    pass


class Phase6V19PairedGraphContextRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._threads = torch.get_num_threads()
        torch.set_num_threads(1)
        if not _SOURCE_CHECKPOINT.is_file():
            raise RuntimeError("the frozen terminal V12 checkpoint is required")
        cls.system = v19.load_v12_champion_paired_graph_context_source(
            _SOURCE_CHECKPOINT,
            device="cpu",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls._threads)

    def _recover_with_logits(
        self,
        raw_logits: torch.Tensor,
        inputs: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[dict[str, object], torch.Tensor]:
        if inputs is None:
            inputs = _inputs(stored_count=raw_logits.shape[1])
        observed: dict[str, torch.Tensor] = {}

        def frozen_evaluation(
            system: v19.V12ChampionPairedGraphContextSystem,
        ) -> dict[str, object]:
            with system.controller.paired_graph_lesion("zero_residual"):
                observed["logits"] = system.controller._paired_graph_context_logits(
                    *inputs
                )
            return {"sentinel": "frozen-evaluation"}

        with mock.patch.object(
            self.system.controller,
            "_context_pair_logits",
            return_value=raw_logits,
        ), mock.patch.object(
            v19,
            "evaluate_v12_champion_paired_graph_context",
            side_effect=frozen_evaluation,
        ) as evaluator:
            result = recovery.evaluate_v19_paired_graph_context_recovery(self.system)
        self.assertEqual(evaluator.call_count, 1)
        return result, observed["logits"]

    def assertWrapperAbsent(self) -> None:
        controller = self.system.controller
        self.assertNotIn("_paired_graph_context_logits", controller.__dict__)
        self.assertNotIn("_v19_evaluation_recovery_wrapper_active", controller.__dict__)
        self.assertIs(
            controller._paired_graph_context_logits.__func__,
            v19.V12ChampionPairedGraphContextController._paired_graph_context_logits,
        )

    def test_protocol_frozen_inputs_and_public_api(self) -> None:
        root = Path(__file__).resolve().parents[3]
        expected = {
            "experiments/runners/phase6_v12_champion_paired_graph_context.py": (
                "54A8E2E510424E485DE34A2975A82C927D22C87B5576EFE00537545158ECE5BE"
            ),
            "tests/unit/experiments/test_phase6_v12_champion_paired_graph_context.py": (
                "C0D1DBBDE81B628D8D9CCFA751DCB9CFE951B3809860BE5298494C103D1E12BD"
            ),
            ".angler_v19_once.py": (
                "099381C7AE58F1FBEEFCEC31B0FE1D53DA591D9D51D4E53549FA534F8D5D3123"
            ),
            "docs/blueprints/branches/learning/work/"
            "ANG-WORK-LEARNING-SOFTWARE-PIPELINE-RECONSTRUCTION-V19-EVAL-RECOVERY-001.md": (
                "4A068443C2FA8A7481576154575FDED9D08CD5ED4064FDE0DAA003A73F2B4A57"
            ),
        }
        self.assertEqual(
            {relative: _sha256(root / relative) for relative in expected},
            expected,
        )
        self.assertEqual(
            recovery.PROTOCOL_ID,
            "phase6.public-v12-champion-paired-graph-context-eval-recovery.v19r1",
        )
        self.assertEqual(
            recovery.__all__,
            ["PROTOCOL_ID", "evaluate_v19_paired_graph_context_recovery"],
        )

    def test_original_zero_residual_duplicate_regression_and_projection(self) -> None:
        inputs = _inputs()
        raw = torch.tensor(
            ((0.25, -0.125, 0.5), (0.25, -0.125, 0.5)),
            dtype=torch.float32,
        )
        raw[1, 0] = torch.nextafter(raw[0, 0], torch.tensor(float("inf")))
        with mock.patch.object(
            self.system.controller,
            "_context_pair_logits",
            return_value=raw,
        ):
            with self.system.controller.paired_graph_lesion("zero_residual"):
                original = self.system.controller._paired_graph_context_logits(*inputs)
        self.assertIs(original, raw)
        self.assertFalse(torch.equal(original[0], original[1]))

        result, projected = self._recover_with_logits(raw, inputs)
        self.assertEqual(result["evaluation"], {"sentinel": "frozen-evaluation"})
        self.assertTrue(torch.equal(projected[0], raw[0]))
        self.assertTrue(torch.equal(projected[1], raw[0]))
        self.assertTrue(torch.equal(raw[1], original[1]))
        audit = result["projection_audit"]
        self.assertEqual(
            (
                audit["zero_residual_calls"],
                audit["duplicate_groups"],
                audit["duplicate_rows_projected"],
            ),
            (1, 1, 1),
        )
        self.assertGreater(audit["maximum_raw_duplicate_logit_difference"], 0.0)
        self.assertLessEqual(
            audit["maximum_raw_duplicate_logit_difference"],
            audit["raw_duplicate_logit_difference_ceiling"],
        )
        self.assertTrue(audit["wrapper_restored"])
        self.assertWrapperAbsent()

    def test_exact_duplicate_and_overlimit_fail_closed(self) -> None:
        exact = torch.tensor(((0.1, 0.2), (0.1, 0.2)), dtype=torch.float32)
        result, projected = self._recover_with_logits(exact)
        self.assertTrue(torch.equal(projected[0], projected[1]))
        self.assertEqual(
            result["projection_audit"][
                "maximum_raw_duplicate_logit_difference"
            ],
            0.0,
        )
        self.assertWrapperAbsent()

        overlimit = exact.clone()
        overlimit[1, 1] += 2.0e-6
        with self.assertRaisesRegex(RuntimeError, "difference ceiling"):
            self._recover_with_logits(overlimit)
        self.assertWrapperAbsent()

    def test_code_graph_or_mask_difference_prevents_grouping(self) -> None:
        raw = torch.tensor(((0.3, -0.2), (-0.4, 0.7)), dtype=torch.float32)
        base = _inputs(stored_count=2)
        variants: dict[str, tuple[torch.Tensor, ...]] = {}
        code_variant = list(value.clone() for value in base)
        code_variant[0][1, 0] = torch.nextafter(
            code_variant[0][1, 0], torch.tensor(float("inf"))
        )
        variants["code"] = tuple(code_variant)
        graph_variant = list(value.clone() for value in base)
        graph_variant[1][1, 4, 0] = ~graph_variant[1][1, 4, 0]
        variants["graph"] = tuple(graph_variant)
        mask_variant = list(value.clone() for value in base)
        mask_variant[2][1, 2] = ~mask_variant[2][1, 2]
        variants["mask"] = tuple(mask_variant)

        for label, inputs in variants.items():
            with self.subTest(label=label):
                result, observed = self._recover_with_logits(raw, inputs)
                self.assertIs(observed, raw)
                self.assertEqual(
                    result["projection_audit"]["duplicate_rows_projected"], 0
                )
                self.assertEqual(
                    result["projection_audit"]["representative_rows"], 2
                )
                self.assertWrapperAbsent()

    def test_all_nonzero_lesions_delegate_exact_frozen_outputs(self) -> None:
        inputs = _inputs(stored_count=1)
        raw = torch.tensor(((0.2,), (0.2,)), dtype=torch.float32)
        observed: list[torch.Tensor] = []

        def frozen_evaluation(
            system: v19.V12ChampionPairedGraphContextSystem,
        ) -> dict[str, object]:
            observed.append(system.controller._paired_graph_context_logits(*inputs))
            for lesion in (
                "uniform_cross_graph_attention",
                "mismatch_zero",
            ):
                with system.controller.paired_graph_lesion(lesion):
                    observed.append(
                        system.controller._paired_graph_context_logits(*inputs)
                    )
            return {"sentinel": True}

        with mock.patch.object(
            self.system.controller,
            "_context_pair_logits",
            return_value=raw,
        ), mock.patch.object(
            v19,
            "evaluate_v12_champion_paired_graph_context",
            side_effect=frozen_evaluation,
        ):
            result = recovery.evaluate_v19_paired_graph_context_recovery(self.system)
        self.assertTrue(all(value is raw for value in observed))
        self.assertEqual(
            result["projection_audit"]["delegated_nonzero_lesion_calls"], 3
        )
        self.assertEqual(result["projection_audit"]["zero_residual_calls"], 0)
        self.assertWrapperAbsent()

    def test_wrapper_is_non_nested_and_restored_after_success_and_error(self) -> None:
        nested_rejected: list[bool] = []

        def successful(
            system: v19.V12ChampionPairedGraphContextSystem,
        ) -> dict[str, object]:
            self.assertIn("_paired_graph_context_logits", system.controller.__dict__)
            with self.assertRaisesRegex(RuntimeError, "nested|pre-existing"):
                recovery.evaluate_v19_paired_graph_context_recovery(system)
            nested_rejected.append(True)
            return {"sentinel": "success"}

        with mock.patch.object(
            v19,
            "evaluate_v12_champion_paired_graph_context",
            side_effect=successful,
        ):
            recovery.evaluate_v19_paired_graph_context_recovery(self.system)
        self.assertEqual(nested_rejected, [True])
        self.assertWrapperAbsent()

        def failing(_system: v19.V12ChampionPairedGraphContextSystem) -> object:
            self.assertIn(
                "_paired_graph_context_logits", _system.controller.__dict__
            )
            raise InjectedEvaluationError("injected evaluator failure")

        with mock.patch.object(
            v19,
            "evaluate_v12_champion_paired_graph_context",
            side_effect=failing,
        ):
            with self.assertRaisesRegex(
                InjectedEvaluationError, "injected evaluator failure"
            ):
                recovery.evaluate_v19_paired_graph_context_recovery(self.system)
        self.assertWrapperAbsent()

    def test_production_zero_lesion_remains_exact_v12_after_recovery(self) -> None:
        with mock.patch.object(
            v19,
            "evaluate_v12_champion_paired_graph_context",
            return_value={"sentinel": "bounded"},
        ):
            recovery.evaluate_v19_paired_graph_context_recovery(self.system)
        self.assertWrapperAbsent()

        base, _, _ = v12.load_public_relation_conflict_checkpoint(_SOURCE_CHECKPOINT)
        plan = v19.v12_champion_paired_graph_context_plan()
        stream = v12._relation_credit_panel_streams(
            plan["commitments"], plan["panel_seed_pairs"][0]
        )[0]
        base_state = base.initial_state()
        v19_state = self.system.controller.initial_state()
        for pair in stream.supports[:3]:
            base_state = v12.acquire_public_pipeline_traces(
                base, pair.learner, base_state
            ).state
            v19_state = v19.acquire_v19_public_pipeline_traces(
                self.system.controller, pair.learner, v19_state
            ).state
        task = stream.supports[3].learner
        common_encoding = self.system.controller.encode_task(task)
        expected = base.score_actions(task, base_state, encoding=common_encoding)
        with self.system.controller.paired_graph_lesion("zero_residual"):
            actual = self.system.controller.score_actions(
                task,
                v19_state,
                encoding=common_encoding,
            )
        for field in v12.SoftwareStepScores.__dataclass_fields__:
            self.assertTrue(
                torch.equal(getattr(expected, field), getattr(actual, field)),
                field,
            )

    def test_real_same_contract_stream_closes_original_zero_lesion_gap(self) -> None:
        system = v19.load_v12_champion_paired_graph_context_checkpoint(
            _TERMINAL_CHECKPOINT,
            device="cpu",
        )
        plan = v19.v12_champion_paired_graph_context_plan()
        stream = v12._relation_credit_panel_streams(
            plan["commitments"], plan["panel_seed_pairs"][0]
        )[0]
        with system.controller.paired_graph_lesion("zero_residual"):
            with self.assertRaisesRegex(
                RuntimeError, "same-contract V19 query alternatives changed context"
            ):
                v19.public_paired_graph_credit_rows(system.controller, stream)

        observed: dict[str, object] = {}

        def bounded(
            candidate: v19.V12ChampionPairedGraphContextSystem,
        ) -> dict[str, object]:
            with candidate.controller.paired_graph_lesion("zero_residual"):
                rows = v19.public_paired_graph_credit_rows(
                    candidate.controller,
                    stream,
                )
            observed["rows"] = rows
            return {"bounded_rows": len(rows)}

        with mock.patch.object(
            v19,
            "evaluate_v12_champion_paired_graph_context",
            side_effect=bounded,
        ):
            result = recovery.evaluate_v19_paired_graph_context_recovery(system)
        self.assertEqual(result["evaluation"], {"bounded_rows": 4})
        self.assertEqual(len(observed["rows"]), 4)
        self.assertGreater(
            result["projection_audit"]["duplicate_rows_projected"], 0
        )
        self.assertGreater(
            result["projection_audit"][
                "maximum_raw_duplicate_logit_difference"
            ],
            0.0,
        )
        self.assertNotIn("_paired_graph_context_logits", system.controller.__dict__)

    def test_terminal_checkpoint_and_system_are_unchanged_by_bounded_wiring(self) -> None:
        self.assertTrue(_TERMINAL_CHECKPOINT.is_file())
        self.assertEqual(_sha256(_TERMINAL_CHECKPOINT), _TERMINAL_CHECKPOINT_SHA256)
        system = v19.load_v12_champion_paired_graph_context_checkpoint(
            _TERMINAL_CHECKPOINT,
            device="cpu",
        )
        before = {
            "system": v19.paired_graph_system_digest(system),
            "mutable": v19.paired_graph_mutable_digest(system.controller),
            "optimizer": v19.paired_graph_optimizer_digest(system.optimizer_state),
            "updates": system.context_updates,
        }
        self.assertEqual(
            before,
            {
                "system": _TERMINAL_SYSTEM_DIGEST,
                "mutable": _TERMINAL_MUTABLE_DIGEST,
                "optimizer": _TERMINAL_OPTIMIZER_DIGEST,
                "updates": 512,
            },
        )
        inputs = _inputs(stored_count=2)
        raw = torch.tensor(((0.1, -0.3), (0.1, -0.3)), dtype=torch.float32)

        def bounded(
            candidate: v19.V12ChampionPairedGraphContextSystem,
        ) -> dict[str, object]:
            with candidate.controller.paired_graph_lesion("zero_residual"):
                candidate.controller._paired_graph_context_logits(*inputs)
            return {"bounded": True}

        with mock.patch.object(
            system.controller,
            "_context_pair_logits",
            return_value=raw,
        ), mock.patch.object(
            v19,
            "evaluate_v12_champion_paired_graph_context",
            side_effect=bounded,
        ):
            recovery.evaluate_v19_paired_graph_context_recovery(system)
        after = {
            "system": v19.paired_graph_system_digest(system),
            "mutable": v19.paired_graph_mutable_digest(system.controller),
            "optimizer": v19.paired_graph_optimizer_digest(system.optimizer_state),
            "updates": system.context_updates,
        }
        self.assertEqual(after, before)
        self.assertEqual(_sha256(_TERMINAL_CHECKPOINT), _TERMINAL_CHECKPOINT_SHA256)
        self.assertNotIn("_paired_graph_context_logits", system.controller.__dict__)

    def test_sources_have_zero_training_calls_and_one_causal_evaluation_call(self) -> None:
        root = Path(__file__).resolve().parents[3]
        evaluator_path = root / (
            "experiments/evaluators/phase6_v19_paired_graph_context_recovery.py"
        )
        harness_path = root / ".angler_v19_eval_recovery_r1_once.py"
        self.assertTrue(harness_path.is_file())

        def calls(path: Path) -> list[str]:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            result = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name):
                    result.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    result.append(node.func.attr)
            return result

        evaluator_calls = calls(evaluator_path)
        harness_calls = calls(harness_path)
        for observed in (evaluator_calls, harness_calls):
            self.assertFalse(any(name.startswith("fit") for name in observed))
            self.assertNotIn("step", observed)
        self.assertEqual(
            evaluator_calls.count("evaluate_v12_champion_paired_graph_context"),
            1,
        )
        self.assertEqual(
            harness_calls.count("evaluate_v19_paired_graph_context_recovery"),
            1,
        )


if __name__ == "__main__":
    unittest.main()
