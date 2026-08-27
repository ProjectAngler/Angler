"""Semantics checks for phase-4 immutable feature memoization."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from angler.procedures import trunk as operator_trunk  # noqa: E402
from angler.procedures.alignment import AliasTable  # noqa: E402
from experiments.runners import phase4_causal_operator_compiler as runner  # noqa: E402
from experiments.runners.causal_operator_experience import (  # noqa: E402
    build_causal_operator_experience,
)


class Phase4FeatureCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        experience = build_causal_operator_experience(
            seed=42_017,
            traces_per_domain=40,
        )
        selected = runner._selected_candidates(experience)
        operator = selected["tokens"].operator
        context = runner._make_context(
            "tokens",
            selected["tokens"],
            operator,
            AliasTable(),
        )
        cls.operator = operator
        cls.examples = runner._experience_examples(context)

    def test_hash_cache_is_exact_detached_and_mutation_isolated(self) -> None:
        operator_trunk._cached_hash_row.cache_clear()
        encoder = operator_trunk.FrozenHashTextEncoder(64)
        text = "immutable relational feature"

        first = encoder.encode_texts((text,))
        after_first = operator_trunk._cached_hash_row.cache_info()
        second = encoder.encode_texts((text,))
        after_second = operator_trunk._cached_hash_row.cache_info()

        self.assertTrue(torch.equal(first, second))
        self.assertEqual(after_first.misses, 1)
        self.assertEqual(after_second.misses, 1)
        self.assertEqual(after_second.hits, 1)
        self.assertIsNone(second.grad_fn)
        first.add_(1000.0)
        third = encoder.encode_texts((text,))
        self.assertTrue(torch.equal(second, third))
        self.assertFalse(torch.equal(first, third))
        self.assertNotIn("_cached_hash_row", encoder.state_dict())

    def test_cached_training_has_identical_losses_and_parameter_updates(self) -> None:
        profile = runner.PROFILES["smoke"]
        device = torch.device("cpu")
        baseline = runner._make_learner(profile, seed=73_001, device=device)
        cached = runner._make_learner(profile, seed=73_001, device=device)
        baseline_optimizer = torch.optim.AdamW(
            baseline.parameters(),
            lr=profile.learning_rate,
            weight_decay=1e-4,
        )
        cached_optimizer = torch.optim.AdamW(
            cached.parameters(),
            lr=profile.learning_rate,
            weight_decay=1e-4,
        )
        common = dict(
            new_examples=self.examples,
            replay_examples=(),
            steps=3,
            batch_size=4,
            replay_ratio=0.0,
            seed=73_002,
        )

        uncached_hash_row = operator_trunk._cached_hash_row.__wrapped__
        with patch.object(
            operator_trunk,
            "_cached_hash_row",
            side_effect=uncached_hash_row,
        ):
            baseline_receipt = runner._train_steps(
                baseline,
                baseline_optimizer,
                **common,
            )

        operator_trunk._cached_hash_row.cache_clear()
        cached_receipt = runner._train_steps(
            cached,
            cached_optimizer,
            **common,
        )
        cache_info = operator_trunk._cached_hash_row.cache_info()

        for field in (
            "steps",
            "examples",
            "replay_examples_available",
            "replay_ratio",
            "first_loss",
            "last_loss",
            "tail_mean_loss",
        ):
            self.assertEqual(cached_receipt[field], baseline_receipt[field])
        self.assertEqual(
            tuple(name for name, _ in baseline.named_parameters()),
            tuple(name for name, _ in cached.named_parameters()),
        )
        for (_, expected), (_, observed) in zip(
            baseline.named_parameters(),
            cached.named_parameters(),
            strict=True,
        ):
            self.assertTrue(torch.equal(observed, expected))
        self.assertGreater(cache_info.hits, 0)
        self.assertGreater(cache_info.misses, 0)


if __name__ == "__main__":
    unittest.main()
