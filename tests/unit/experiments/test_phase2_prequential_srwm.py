from __future__ import annotations

import unittest

import torch

from angler.reasoning import self_referential_state_digest
from experiments.evaluators.latent_order_suite import evaluator_programs
from experiments.runners.phase2_prequential_srwm import (
    _PROFILES,
    _UniqueSeedStream,
    _balanced_flags,
    _classify_effect,
    _generate_batch,
    _make_model,
    _run_online_sequence,
)


class PrequentialSrwmRunnerTests(unittest.TestCase):
    def test_balanced_flags_and_status_require_causal_evidence(self) -> None:
        self.assertEqual(_balanced_flags(8).count(False), 4)
        self.assertEqual(_balanced_flags(8).count(True), 4)
        with self.assertRaisesRegex(ValueError, "positive and even"):
            _balanced_flags(3)

        status, checks = _classify_effect(
            {
                "macro_worst_flag_gain": 0.2,
                "positive_gain_mechanisms": 4,
                "macro_worst_flag_causal_advantage": 0.0,
                "causal_control_mechanisms": 0,
                "macro_worst_flag_reacquisition_savings": 0.1,
                "positive_reacquisition_mechanisms": 4,
            }
        )
        self.assertEqual(status, "NO_CAUSAL_ADAPTIVE_EFFECT_OBSERVED")
        self.assertTrue(checks["online_gain"])
        self.assertFalse(checks["matched_causal_controls"])

    def test_matched_branches_share_rng_and_no_write_preserves_state(self) -> None:
        torch.manual_seed(4401)
        device = torch.device("cpu")
        model = _make_model(_PROFILES["smoke"], device)
        model.eval().requires_grad_(False)
        state = model.initial_state(1)
        before = self_referential_state_digest(state)
        flags = _balanced_flags(4)
        stream = _UniqueSeedStream(4401)
        program = evaluator_programs()[0]
        tasks = _generate_batch(
            (program,) * 4,
            stream.take(4),
            stream,
            public_flags=flags,
        )

        with torch.no_grad():
            ordinary_state, ordinary = _run_online_sequence(
                model,
                state,
                tasks,
                device,
                mode="ordinary",
                sampling_seed=9981,
            )
            no_write_state, no_write = _run_online_sequence(
                model,
                state,
                tasks,
                device,
                mode="no_write",
                sampling_seed=9981,
            )
            repeated_state, repeated = _run_online_sequence(
                model,
                state,
                tasks,
                device,
                mode="ordinary",
                sampling_seed=9981,
            )

        self.assertEqual(
            ordinary["trajectory"][0]["score_before_write"],
            no_write["trajectory"][0]["score_before_write"],
        )
        self.assertEqual(ordinary, repeated)
        self.assertEqual(
            self_referential_state_digest(ordinary_state),
            self_referential_state_digest(repeated_state),
        )
        self.assertNotEqual(
            self_referential_state_digest(ordinary_state),
            before,
        )
        self.assertEqual(
            self_referential_state_digest(no_write_state),
            before,
        )


if __name__ == "__main__":
    unittest.main()
