import re
import unittest

import torch

from angler.reasoning import EdgeAwareActionTraceDecoder, StructuredActionTraceMatcherDecoder
from experiments.runners import trifecta_correspondence_composition_v8 as v8
from experiments.runners import trifecta_edge_procedure_v9 as v9


class TrifectaEdgeProcedureV9Tests(unittest.TestCase):
    def _example(self):
        from experiments.runners import scaled_procedural_software_v1 as v1

        _, _, _, final = v1.prepare_corpus()
        return final[0].queries[0]

    def test_public_edges_are_rename_invariant(self) -> None:
        example = self._example()
        original = v9.public_edge_features(example.query_text, example.action_texts)

        def rename(text: str) -> str:
            return re.sub(
                r"([ST])(\d+)",
                lambda match: f"{match.group(1)}{int(match.group(2)) + 100}",
                text,
            )

        renamed = v9.public_edge_features(
            rename(example.query_text),
            tuple(rename(text) for text in example.action_texts),
        )
        self.assertTrue(torch.equal(original, renamed))

    def test_public_edges_are_candidate_permutation_equivariant(self) -> None:
        example = self._example()
        original = v9.public_edge_features(example.query_text, example.action_texts)
        order = torch.tensor([2, 0, 5, 1, 4, 3])
        permuted = v9.public_edge_features(
            example.query_text,
            tuple(example.action_texts[index] for index in order.tolist()),
        )
        expected = original.index_select(1, order).index_select(2, order)
        self.assertTrue(torch.equal(permuted, expected))

    def test_v8_decoder_migration_adds_only_declared_parameters(self) -> None:
        config = {
            "content_width": 16,
            "procedure_width": 12,
            "hidden_width": 24,
            "heads": 4,
            "maximum_actions": 6,
            "maximum_steps": 4,
        }
        parent = StructuredActionTraceMatcherDecoder(**config)
        successor = EdgeAwareActionTraceDecoder(
            **config,
            relation_count=v9.RELATION_COUNT,
            message_rounds=v9.MESSAGE_ROUNDS,
        )
        missing = v9.migrate_v8_decoder_state(successor, parent.state_dict())
        self.assertTrue(missing)
        self.assertTrue(
            all(
                any(name.startswith(prefix) for prefix in v9.NEW_DECODER_PREFIXES)
                for name in missing
            )
        )
        for name, value in parent.state_dict().items():
            self.assertTrue(torch.equal(successor.state_dict()[name], value))


if __name__ == "__main__":
    unittest.main()
