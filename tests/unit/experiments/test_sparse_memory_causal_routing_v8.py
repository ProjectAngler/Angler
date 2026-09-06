from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from angler.reasoning import SparseMemoryCausalRoutingCore
from experiments.corpora.causal_neuromodulated_apprenticeship_v6 import (
    FinalPartitionSealedError,
)
from experiments.runners import content_addressed_causal_memory_v7 as v7
from experiments.runners import sparse_memory_causal_routing_v8 as v8


class SparseMemoryCausalRoutingV8RunnerTests(unittest.TestCase):
    def test_protocol_is_inherited_without_threshold_or_schedule_change(self) -> None:
        self.assertIs(v8._train_core, v7._train_core)
        self.assertIs(v8._evaluate_all, v7._evaluate_all)
        self.assertIs(v8._development_gate, v7._development_gate)
        self.assertEqual(v8.RANK, v7.RANK)
        self.assertEqual(v8.MEMORY_SLOTS, v7.MEMORY_SLOTS)
        self.assertEqual(v8.TEMPORAL_WIDTH, v7.TEMPORAL_WIDTH)
        self.assertEqual(len(v7.v6._pair_schedule()), 256)

    def test_only_outcome_blind_semantic_parameters_are_loaded_from_v7(self) -> None:
        torch.manual_seed(201)
        donor = SparseMemoryCausalRoutingCore(content_width=6, temporal_width=8)
        donor_state = donor.state_dict()
        payload = {"identity": v7.IDENTITY, "core_state_dict": donor_state}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v7.pt"
            torch.save(payload, path)
            torch.manual_seed(202)
            recipient = SparseMemoryCausalRoutingCore(content_width=6, temporal_width=8)
            outcome_before = recipient.outcome_embedding.weight.detach().clone()
            record = v8._load_semantic_donor(recipient, path)

        self.assertEqual(len(record["mapping"]), 7)
        for name in record["mapping"]:
            self.assertTrue(torch.equal(recipient.state_dict()[name], donor_state[name]))
        self.assertTrue(torch.equal(recipient.outcome_embedding.weight, outcome_before))

    def test_final_partition_is_still_sealed_by_default(self) -> None:
        corpus = v8.build_causal_neuromodulated_apprenticeship_v6()
        with self.assertRaises(FinalPartitionSealedError):
            _ = corpus.final


if __name__ == "__main__":
    unittest.main()
