import unittest

import torch

from experiments.runners import phase6_software_pipeline_reconstruction as v12
from experiments.runners import phase6_v12_champion_paired_graph_context as v19
from experiments.runners import phase6_oml_relation_representation as v20
from experiments.runners import phase6_oml_production_bridge_v21b as v21b


class OMLProductionBridgeV21BTests(unittest.TestCase):
    def test_parameter_ownership_excludes_donor_representation(self) -> None:
        controller = v19.V12ChampionPairedGraphContextController(
            v12.SOFTWARE_PIPELINE_PROFILES["smoke"]
        )
        names = set(v21b.configure_bridge_trainability(controller, train=True))
        report = v20._validate_parameter_partition(controller)
        self.assertTrue(names)
        self.assertEqual(len(names), 212)
        self.assertTrue(any(name.startswith("backward_reasoner.") for name in names))
        self.assertTrue(any(name.startswith("stop_head.") for name in names))
        self.assertFalse(names & set(report["rln_parameter_names"]))
        self.assertFalse(any(name.startswith("paired_graph_") for name in names))
        self.assertFalse(any(name.startswith("evidence_context_encoder.") for name in names))
        v21b.configure_bridge_trainability(controller, train=False)
        self.assertFalse(any(value.requires_grad for value in controller.parameters()))

    def test_v19_acquisition_retains_raw_graph_state(self) -> None:
        controller = v19.V12ChampionPairedGraphContextController(
            v12.SOFTWARE_PIPELINE_PROFILES["smoke"]
        )
        stream = v21b._stream("train", 0)
        state = v21b.acquire_supports(controller, stream.supports)
        self.assertIs(type(state), v19.V19SoftwareReconstructionState)
        self.assertTrue(bool(state.context_trace_graph_masks.any().item()))
        snapshot = v19.snapshot_v19_reconstruction_state(state)
        restored = v19.restore_v19_reconstruction_state(snapshot)
        self.assertTrue(torch.equal(state.context_trace_graphs, restored.context_trace_graphs))

    def test_one_public_fold_reaches_only_bridge_parameters(self) -> None:
        controller = v19.V12ChampionPairedGraphContextController(
            v12.SOFTWARE_PIPELINE_PROFILES["smoke"]
        )
        selected = set(v21b.configure_bridge_trainability(controller, train=True))
        protected = v21b._protected_state(controller)
        stream = v21b._stream("train", 0)
        state = v21b.acquire_supports(controller, stream.supports[1:])
        loss = controller.public_heldout_production_losses(
            stream.supports[0].learner,
            state,
            include_role_memory_causal_hinge=True,
            detach_evidence_action_input=True,
            use_legacy_evidence=False,
        ).mean()
        loss.backward()
        named = dict(controller.named_parameters())
        reached = {
            name
            for name in selected
            if named[name].grad is not None
            and bool(torch.isfinite(named[name].grad).all().item())
        }
        self.assertTrue(reached)
        self.assertFalse(
            any(
                parameter.grad is not None
                for name, parameter in controller.named_parameters()
                if name not in selected
            )
        )
        v21b._assert_protected_exact(controller, protected)


if __name__ == "__main__":
    unittest.main()
