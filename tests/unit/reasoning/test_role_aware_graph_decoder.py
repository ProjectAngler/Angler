import re
import unittest

import torch

from angler.reasoning import (
    RoleAwareGraphProcedureDecoder,
    build_public_candidate_roles,
    parse_public_relations,
)


class RoleAwareGraphProcedureDecoderTests(unittest.TestCase):
    def test_public_roles_are_rename_invariant(self) -> None:
        query = "origin=S0; goal=S2; forbidden=S9; budget=4"
        actions = (
            "A in=T0 out=T1 reads=S0 writes=S1 graph=0>1",
            "B in=T1 out=T2 reads=S1 writes=S2 graph=0>1",
        )
        first = build_public_candidate_roles(
            parse_public_relations(query, actions), maximum_actions=6
        )
        def rename(value: str) -> str:
            return re.sub(
                r"([ST])(\d+)",
                lambda match: f"{match.group(1)}{int(match.group(2)) + 10}",
                value,
            )
        second = build_public_candidate_roles(
            parse_public_relations(
                rename(query),
                tuple(rename(value) for value in actions),
            ),
            maximum_actions=6,
        )
        self.assertTrue(torch.equal(first, second))

    def test_roles_have_finite_causal_gradient_path(self) -> None:
        decoder = RoleAwareGraphProcedureDecoder(
            content_width=16,
            procedure_width=12,
            hidden_width=24,
            heads=4,
            maximum_actions=6,
            maximum_steps=4,
            relation_count=8,
            message_rounds=2,
            role_width=12,
        )
        common = dict(
            procedure_slots=torch.randn(1, 3, 12),
            action_features=torch.randn(1, 6, 16),
            action_mask=torch.ones(1, 6, dtype=torch.bool),
            edge_features=torch.randn(1, 6, 6, 8),
            donor_scores=torch.randn(1, 6),
            memory_values=torch.randn(4, 12),
            memory_mask=torch.ones(4, dtype=torch.bool),
            role_features=torch.randn(1, 6, 12),
        )
        with_roles = decoder(**common, include_roles=True).logits
        without_roles = decoder(**common, include_roles=False).logits
        self.assertFalse(torch.equal(with_roles, without_roles))
        with_roles[torch.isfinite(with_roles)].square().mean().backward()
        gradients = [
            parameter.grad
            for name, parameter in decoder.named_parameters()
            if name.startswith("role_")
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(value is not None and torch.isfinite(value).all() for value in gradients))


if __name__ == "__main__":
    unittest.main()
