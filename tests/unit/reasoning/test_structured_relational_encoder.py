from __future__ import annotations

import unittest

import torch

from angler.reasoning import (
    SemanticRelationalFusion,
    StructuredActionTraceMatcherDecoder,
    StructuredActionTraceProceduralCore,
    StructuredRelationalEncoder,
    build_relational_incidence,
    parse_public_relations,
)


QUERY = "origin=S0; goal=S2; forbidden=none; budget=4\ncomponents:"
ACTIONS = (
    "A in=T0 out=T1 reads=S0 writes=S1 graph=0>1,1>2",
    "B in=T1 out=T2 reads=S1 writes=S2 graph=0>2,2>1",
)


class StructuredRelationalEncoderTests(unittest.TestCase):
    def test_parser_and_incidence_preserve_only_declared_relations(self) -> None:
        graph = parse_public_relations(QUERY, ACTIONS)
        tensors = build_relational_incidence(graph)
        self.assertEqual(tuple(tensors.candidate_entity_indices.tolist()), (0, 1))
        self.assertEqual(tensors.fact_features.ndim, 3)
        self.assertEqual(tensors.entity_features.ndim, 3)
        self.assertEqual(tensors.mention_features.ndim, 3)
        self.assertTrue(bool(tensors.fact_mask.all()))
        self.assertTrue(bool(tensors.entity_mask.all()))
        self.assertTrue(bool(tensors.mention_mask.all()))

    def test_component_local_topology_nodes_do_not_alias(self) -> None:
        graph = parse_public_relations(QUERY, ACTIONS)
        tensors = build_relational_incidence(graph)
        # 2 candidates + 3 states + 3 types + 6 candidate-local topology nodes.
        self.assertEqual(tensors.entity_features.shape[1], 14)

    def test_renaming_symbols_leaves_incidence_bytes_equal(self) -> None:
        renamed_query = "origin=S7; goal=S9; forbidden=none; budget=4\ncomponents:"
        renamed_actions = (
            "A in=T5 out=T6 reads=S7 writes=S8 graph=5>6,6>7",
            "B in=T6 out=T7 reads=S8 writes=S9 graph=5>7,7>6",
        )
        left = build_relational_incidence(parse_public_relations(QUERY, ACTIONS))
        right = build_relational_incidence(parse_public_relations(renamed_query, renamed_actions))
        for name in (
            "fact_features",
            "entity_features",
            "mention_features",
            "mention_fact_indices",
            "mention_entity_indices",
            "candidate_entity_indices",
        ):
            self.assertTrue(torch.equal(getattr(left, name), getattr(right, name)), name)

    def test_topology_lesion_changes_only_topology_edge_facts(self) -> None:
        graph = parse_public_relations(QUERY, ACTIONS)
        full = build_relational_incidence(graph)
        lesioned = build_relational_incidence(graph, include_topology_edges=False)
        self.assertEqual(full.entity_features.shape, lesioned.entity_features.shape)
        self.assertGreater(full.fact_features.shape[1], lesioned.fact_features.shape[1])

    def test_candidate_permutation_is_equivariant(self) -> None:
        torch.manual_seed(11)
        encoder = StructuredRelationalEncoder(width=32, reasoning_steps=2, workspace_slots=2)
        encoder.eval()
        original = build_relational_incidence(parse_public_relations(QUERY, ACTIONS))
        permuted = build_relational_incidence(parse_public_relations(QUERY, tuple(reversed(ACTIONS))))
        with torch.inference_mode():
            left, _ = encoder(original)
            right, _ = encoder(permuted)
        self.assertTrue(torch.allclose(left[:, 0], right[:, 1], atol=1.0e-5, rtol=1.0e-5))
        self.assertTrue(torch.allclose(left[:, 1], right[:, 0], atol=1.0e-5, rtol=1.0e-5))

    def test_encoder_and_zero_residual_fusion_are_differentiable(self) -> None:
        torch.manual_seed(7)
        tensors = build_relational_incidence(parse_public_relations(QUERY, ACTIONS))
        encoder = StructuredRelationalEncoder(width=32, reasoning_steps=2, workspace_slots=2)
        fusion = SemanticRelationalFusion(
            semantic_width=64,
            relational_width=32,
            hidden_width=48,
        )
        candidates, workspace = encoder(tensors)
        semantic = torch.randn(1, 2, 64)
        fused = fusion(semantic, candidates)
        self.assertTrue(torch.equal(fused, semantic))
        self.assertEqual(workspace.shape, (1, 2, 32))
        fused.square().mean().backward()
        self.assertIsNotNone(fusion.relational_residual[-1].weight.grad)
        self.assertGreater(float(fusion.relational_residual[-1].weight.grad.norm()), 0.0)

    def test_invalid_grammar_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            parse_public_relations(QUERY, ("A answer=correct",))

    def test_structured_successor_preserves_v6_parameter_topology(self) -> None:
        from angler.reasoning import ProceduralCoreConfig

        config = ProceduralCoreConfig(
            tier="test",
            content_width=64,
            temporal_width=8,
            model_width=32,
            heads=4,
            depth=1,
            feedforward_multiplier=2,
            procedure_tokens=2,
            plastic_slots=4,
            dropout=0.0,
        )
        core = StructuredActionTraceProceduralCore(config, maximum_trace_steps=3)
        decoder = StructuredActionTraceMatcherDecoder(
            procedure_width=32,
            content_width=64,
            hidden_width=32,
            heads=4,
            maximum_actions=3,
            maximum_steps=3,
        )
        self.assertIn("action_trace_projection.1.weight", core.state_dict())
        self.assertIn("matcher_update.2.weight", decoder.state_dict())


if __name__ == "__main__":
    unittest.main()
