from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
import inspect
import itertools
import json
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for location in (ROOT, SRC):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))


from angler.procedures.records import ActionSchema, Record, State  # noqa: E402
from experiments.evaluators import (  # noqa: E402
    software_pipeline_reconstruction_suite as suite,
)
from experiments.evaluators.software_pipeline_reconstruction_suite import (  # noqa: E402
    CommittedSoftwarePipeline,
    commit_software_pipeline,
    judge_software_pipeline_attempt,
    make_software_pipeline_control_stream,
    make_software_pipeline_stream,
    software_pipeline_mechanism_partition,
)


def _actions_for_solution(
    pair: suite.GeneratedSoftwarePipelineTask,
    solution: suite._HiddenSoftwarePipelineSolution | None = None,
    *,
    interleave: bool = False,
) -> tuple:
    if solution is None:
        solution = pair.hidden
    public_actions = {
        action.digest: action for action in pair.learner.grounded_candidates
    }
    by_role = {
        (implementation.motif, implementation.stage): public_actions[
            implementation.action_digest
        ]
        for implementation in solution.implementations
        if not implementation.distractor
    }
    motifs = solution.required_motifs
    if interleave and len(motifs) == 2:
        return (
            by_role[(motifs[0], 0)],
            by_role[(motifs[1], 0)],
            by_role[(motifs[0], 1)],
            by_role[(motifs[1], 1)],
        )
    return tuple(
        by_role[(motif, stage)]
        for motif in motifs
        for stage in (0, 1)
    )


def _all_stage_preserving_actions(
    pair: suite.GeneratedSoftwarePipelineTask,
) -> tuple[tuple, ...]:
    public_actions = {
        action.digest: action for action in pair.learner.grounded_candidates
    }
    by_role = {
        (implementation.motif, implementation.stage): public_actions[
            implementation.action_digest
        ]
        for implementation in pair.hidden.implementations
        if not implementation.distractor
    }
    left, right = pair.hidden.required_motifs
    rows = []
    for left_positions in itertools.combinations(range(4), 2):
        sequence = [None] * 4
        right_positions = tuple(
            index for index in range(4) if index not in left_positions
        )
        for position, stage in zip(left_positions, (0, 1), strict=True):
            sequence[position] = by_role[(left, stage)]
        for position, stage in zip(right_positions, (0, 1), strict=True):
            sequence[position] = by_role[(right, stage)]
        rows.append(tuple(sequence))
    return tuple(rows)


def _commit_exact(
    pair: suite.GeneratedSoftwarePipelineTask,
    solution: suite._HiddenSoftwarePipelineSolution | None = None,
    *,
    interleave: bool = False,
) -> CommittedSoftwarePipeline:
    actions = _actions_for_solution(pair, solution, interleave=interleave)
    return commit_software_pipeline(
        pair.learner,
        actions,
        stopped=len(actions) < pair.learner.max_steps,
    )


def _public_contract_for_action(task, action):
    return next(
        value for value in task.components if value.schema == action.schema
    )


def _public_successor(task, state, action) -> State:
    contract = _public_contract_for_action(task, action)
    records = set(state.records)
    records.update(
        Record(f"{state.namespace}.holds", (token,))
        for token in contract.state_writes
    )
    return State.from_records(
        state.namespace,
        sorted(records, key=lambda value: (value.predicate, value.arguments)),
    )


def _relation_count(contract) -> int:
    return sum(
        record.predicate.endswith(".relates") for record in contract.incidence
    )


def _count_and_type_chain_baseline(stream, pair):
    """The invalidated count/type heuristic, retained as a regression.

    It observes only public support actions, records their incidence counts,
    follows public input/output type chains, and breaks any remaining tie by a
    fresh opaque action digest.  It never reads evaluator-private state.
    """

    selected_counts: dict[int, list[int]] = {}
    for support in stream.supports:
        transitions = support.learner.observations[0].transitions
        first = _public_contract_for_action(
            support.learner, transitions[0].action
        )
        second = _public_contract_for_action(
            support.learner, transitions[1].action
        )
        selected_counts.setdefault(_relation_count(first), []).append(
            _relation_count(second)
        )

    task = pair.learner
    contracts = {
        action: _public_contract_for_action(task, action)
        for action in task.grounded_candidates
    }
    output_types = {value.output_type for value in contracts.values()}
    first_actions = sorted(
        (
            action
            for action, contract in contracts.items()
            if contract.input_type not in output_types
        ),
        key=lambda value: value.digest,
    )
    actions = []
    for first_action in first_actions:
        first_contract = contracts[first_action]
        alternatives = sorted(
            (
                action
                for action, contract in contracts.items()
                if contract.input_type == first_contract.output_type
            ),
            key=lambda value: value.digest,
        )
        expected = selected_counts.get(_relation_count(first_contract), [])
        matched = [
            action
            for action in alternatives
            if _relation_count(contracts[action]) in expected
        ]
        actions.extend((first_action, (matched or alternatives)[0]))
    return commit_software_pipeline(
        task,
        actions,
        stopped=len(actions) < task.max_steps,
    )


def _topology_signature(contract):
    edges = [
        (record.arguments[1], record.arguments[2])
        for record in contract.incidence
        if record.predicate.endswith(".relates")
    ]
    nodes = sorted({value for edge in edges for value in edge})
    labels = {
        node: (
            sum(target == node for _, target in edges),
            sum(source == node for source, _ in edges),
        )
        for node in nodes
    }
    for _ in range(len(nodes)):
        rows = {
            node: (
                labels[node],
                tuple(sorted(labels[source] for source, target in edges if target == node)),
                tuple(sorted(labels[target] for source, target in edges if source == node)),
            )
            for node in nodes
        }
        vocabulary = {
            value: index
            for index, value in enumerate(sorted(set(rows.values()), key=repr))
        }
        labels = {node: vocabulary[value] for node, value in rows.items()}
    return (
        tuple(sorted(labels.values())),
        tuple(sorted((labels[source], labels[target]) for source, target in edges)),
    )


def _wl_signature_retrieval_baseline(stream, pair):
    """Prior exact-WL episodic lookup retained as an adversarial regression."""

    observed = set()
    for support in stream.supports:
        for transition in support.learner.observations[0].transitions:
            observed.add(
                _topology_signature(
                    _public_contract_for_action(
                        support.learner,
                        transition.action,
                    )
                )
            )
    task = pair.learner
    contracts = {
        action: _public_contract_for_action(task, action)
        for action in task.grounded_candidates
    }
    output_types = {value.output_type for value in contracts.values()}
    first_actions = sorted(
        (
            action
            for action, contract in contracts.items()
            if contract.input_type not in output_types
        ),
        key=lambda value: value.digest,
    )
    actions = []
    for first_action in first_actions:
        first_contract = contracts[first_action]
        alternatives = sorted(
            (
                action
                for action, contract in contracts.items()
                if contract.input_type == first_contract.output_type
            ),
            key=lambda value: value.digest,
        )
        matches = [
            action
            for action in alternatives
            if _topology_signature(contracts[action]) in observed
        ]
        actions.extend((first_action, (matches or alternatives)[0]))
    return commit_software_pipeline(
        task,
        actions,
        stopped=len(actions) < task.max_steps,
    )


def _candidate_graphlet_feature(contract):
    """Public candidate-only degree/walk feature from the red-team attack."""

    edges = [
        (record.arguments[1], record.arguments[2])
        for record in contract.incidence
        if record.predicate.endswith(".relates")
    ]
    nodes = sorted({value for edge in edges for value in edge})
    indices = {value: index for index, value in enumerate(nodes)}
    adjacency = [[0 for _ in nodes] for _ in nodes]
    for source, target in edges:
        adjacency[indices[source]][indices[target]] = 1
    categories = [
        (
            sum(row[index] for row in adjacency),
            sum(adjacency[index]),
        )
        for index in range(len(nodes))
    ]
    category_values = sorted(set(categories))
    power = [row[:] for row in adjacency]
    feature = []
    incoming = [
        [source for source in range(len(nodes)) if adjacency[source][target]]
        for target in range(len(nodes))
    ]
    for _ in range(8):
        total = max(1, sum(sum(row) for row in power))
        for left_category in category_values:
            for right_category in category_values:
                mass = sum(
                    power[left][right]
                    for left in range(len(nodes))
                    for right in range(len(nodes))
                    if categories[left] == left_category
                    and categories[right] == right_category
                )
                feature.append(mass / total)
        local = sorted(
            (
                sum(power[index]),
                sum(row[index] for row in power),
            )
            for index in range(len(nodes))
        )
        scale = max(1, *(value for pair in local for value in pair))
        feature.extend(value / scale for pair in local for value in pair)
        power = [
            [
                sum(power[source][middle] for middle in incoming[target])
                for target in range(len(nodes))
            ]
            for source in range(len(nodes))
        ]
    return tuple(feature)


def _candidate_graphlet_knn_baseline(stream, pair):
    support_examples = []
    for support in stream.supports:
        task = support.learner
        selected = {
            transition.action for trace in task.observations for transition in trace.transitions
        }
        produced = {component.output_type for component in task.components}
        for action in task.grounded_candidates:
            component = _public_contract_for_action(task, action)
            if component.input_type in produced:
                support_examples.append(
                    (_candidate_graphlet_feature(component), action in selected)
                )
    positives = [feature for feature, selected in support_examples if selected]
    negatives = [feature for feature, selected in support_examples if not selected]
    if not positives or not negatives:
        raise RuntimeError("graphlet baseline requires both support labels")

    def distance(left, right):
        return sum((a - b) ** 2 for a, b in zip(left, right, strict=True))

    task = pair.learner
    contracts = {
        action: _public_contract_for_action(task, action)
        for action in task.grounded_candidates
    }
    output_types = {value.output_type for value in contracts.values()}
    roots = sorted(
        (
            action
            for action, contract in contracts.items()
            if contract.input_type not in output_types
        ),
        key=lambda value: value.digest,
    )
    actions = []
    for root in roots:
        alternatives = sorted(
            (
                action
                for action, contract in contracts.items()
                if contract.input_type == contracts[root].output_type
            ),
            key=lambda value: value.digest,
        )
        scored = []
        for action in alternatives:
            feature = _candidate_graphlet_feature(contracts[action])
            score = min(distance(feature, value) for value in positives) - min(
                distance(feature, value) for value in negatives
            )
            scored.append((score, action.digest, action))
        actions.extend((root, min(scored)[2]))
    return commit_software_pipeline(
        task,
        actions,
        stopped=len(actions) < task.max_steps,
    )


def _public_graph_relation(contract):
    edges = {
        (record.arguments[1], record.arguments[2])
        for record in contract.incidence
        if record.predicate.endswith(".relates")
    }
    nodes = sorted({value for edge in edges for value in edge})
    incoming = {
        node: {source for source, target in edges if target == node} for node in nodes
    }
    outgoing = {
        node: {target for source, target in edges if source == node} for node in nodes
    }
    sources = {node for node in nodes if len(outgoing[node]) == 2}
    destinations = {node for node in nodes if len(incoming[node]) == 2}
    cycle_successor = {}
    for node in nodes:
        candidates = outgoing[node]
        if node in sources:
            candidates = candidates - destinations
        if len(candidates) != 1:
            raise RuntimeError("public cycle relation is ambiguous")
        cycle_successor[node] = next(iter(candidates))
    cycle = [nodes[0]]
    while cycle_successor[cycle[-1]] != cycle[0]:
        cycle.append(cycle_successor[cycle[-1]])
        if len(cycle) > len(nodes):
            raise RuntimeError("public cycle relation does not close")
    if len(cycle) != len(nodes):
        raise RuntimeError("public cycle relation omits nodes")
    ordered_sources = [node for node in cycle if node in sources]
    source_positions = [cycle.index(node) for node in ordered_sources]
    gaps = tuple(
        sorted(
            (
                source_positions[(index + 1) % len(source_positions)]
                - source_positions[index]
            )
            % len(cycle)
            for index in range(len(source_positions))
        )
    )
    mapping = tuple(
        next(iter(outgoing[source] & destinations)) for source in ordered_sources
    )
    return gaps, mapping


def _joint_relation_oracle(stream, pair):
    learned = {}
    for support in stream.supports:
        task = support.learner
        output_types = {component.output_type for component in task.components}
        root = next(
            component
            for component in task.components
            if component.input_type not in output_types
        )
        selected_action = task.observations[0].transitions[1].action
        selected = _public_contract_for_action(task, selected_action)
        key, base_mapping = _public_graph_relation(root)
        _, selected_mapping = _public_graph_relation(selected)
        if selected_mapping == base_mapping[1:] + base_mapping[:1]:
            relation = 1
        elif selected_mapping == base_mapping[-1:] + base_mapping[:-1]:
            relation = -1
        else:
            raise RuntimeError("support does not expose a declared graph relation")
        learned[key] = relation

    task = pair.learner
    contracts = {
        action: _public_contract_for_action(task, action)
        for action in task.grounded_candidates
    }
    output_types = {component.output_type for component in contracts.values()}
    roots = sorted(
        (
            action
            for action, component in contracts.items()
            if component.input_type not in output_types
        ),
        key=lambda value: value.digest,
    )
    actions = []
    for root_action in roots:
        root = contracts[root_action]
        key, base_mapping = _public_graph_relation(root)
        expected = learned[key]
        alternatives = []
        for action, component in contracts.items():
            if component.input_type != root.output_type:
                continue
            _, mapping = _public_graph_relation(component)
            if mapping == base_mapping[1:] + base_mapping[:1]:
                relation = 1
            elif mapping == base_mapping[-1:] + base_mapping[:-1]:
                relation = -1
            else:
                raise RuntimeError("query twin is not a declared graph relation")
            alternatives.append((relation, action))
        selected = next(action for relation, action in alternatives if relation == expected)
        actions.extend((root_action, selected))
    return commit_software_pipeline(
        task,
        actions,
        stopped=len(actions) < task.max_steps,
    )


class SoftwarePipelinePartitionTests(unittest.TestCase):
    def test_train_and_development_commitments_are_deterministic_and_disjoint(
        self,
    ) -> None:
        train = software_pipeline_mechanism_partition("train")
        development = software_pipeline_mechanism_partition("development")

        self.assertEqual(len(train), 64)
        self.assertEqual(len(development), 16)
        self.assertEqual(len(set(train)), 64)
        self.assertEqual(len(set(development)), 16)
        self.assertTrue(set(train).isdisjoint(development))
        self.assertEqual(train, software_pipeline_mechanism_partition("train"))
        self.assertEqual(
            development,
            software_pipeline_mechanism_partition("development"),
        )
        train_semantics = {
            (value.motifs, value.variants, value.presentation_variant)
            for value in suite._semantic_partition("train")
        }
        development_semantics = {
            (value.motifs, value.variants, value.presentation_variant)
            for value in suite._semantic_partition("development")
        }
        self.assertEqual(len(train_semantics), 64)
        self.assertEqual(len(development_semantics), 16)
        self.assertTrue(train_semantics.isdisjoint(development_semantics))
        self.assertGreaterEqual(
            len(
                {
                    value.motifs
                    for value in suite._semantic_partition("development")
                }
            ),
            6,
        )

    def test_development_generation_opens_only_requested_partition(self) -> None:
        original = suite._semantic_partition
        opened: list[str] = []

        def recording(partition: str):
            opened.append(partition)
            return original(partition)

        with mock.patch.object(suite, "_semantic_partition", side_effect=recording):
            stream = make_software_pipeline_stream(
                61_101,
                mechanism_partition="development",
            )

        self.assertEqual(stream.mechanism_partition, "development")
        self.assertEqual(set(opened), {"development"})

    def test_commitment_cannot_cross_partition_boundary(self) -> None:
        train_commitment = software_pipeline_mechanism_partition("train")[0]
        with self.assertRaisesRegex(ValueError, "outside"):
            make_software_pipeline_stream(
                61_103,
                mechanism_commitment=train_commitment,
                mechanism_partition="development",
            )


class SoftwarePipelineGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.commitment = software_pipeline_mechanism_partition("development")[3]
        self.stream = make_software_pipeline_stream(
            61_201,
            surface_seed=71_201,
            supports_per_motif=2,
            queries=3,
            mechanism_commitment=self.commitment,
            mechanism_partition="development",
        )

    def test_stream_replays_exactly_and_rerender_changes_only_surface(self) -> None:
        replay = make_software_pipeline_stream(
            61_201,
            surface_seed=71_201,
            supports_per_motif=2,
            queries=3,
            mechanism_commitment=self.commitment,
            mechanism_partition="development",
        )
        rerendered = make_software_pipeline_stream(
            61_201,
            surface_seed=71_202,
            supports_per_motif=2,
            queries=3,
            mechanism_commitment=self.commitment,
            mechanism_partition="development",
        )

        self.assertEqual(self.stream, replay)
        self.assertEqual(
            self.stream.mechanism_commitment,
            rerendered.mechanism_commitment,
        )
        for left, right in zip(
            (*self.stream.supports, *self.stream.queries),
            (*rerendered.supports, *rerendered.queries),
            strict=True,
        ):
            self.assertNotEqual(left.learner.to_canonical(), right.learner.to_canonical())
            self.assertEqual(left.hidden.required_motifs, right.hidden.required_motifs)
            self.assertEqual(left.hidden.integration_inputs, right.hidden.integration_inputs)
            self.assertEqual(
                tuple(
                    (
                        value.motif,
                        value.stage,
                        value.variant,
                        value.distractor,
                    )
                    for value in left.hidden.implementations
                ),
                tuple(
                    (
                        value.motif,
                        value.stage,
                        value.variant,
                        value.distractor,
                    )
                    for value in right.hidden.implementations
                ),
            )

    def test_every_package_has_fresh_opaque_identity_and_zero_address_overlap(
        self,
    ) -> None:
        pairs = (*self.stream.supports, *self.stream.queries)
        namespaces = [pair.learner.origin.namespace for pair in pairs]
        self.assertEqual(len(set(namespaces)), len(namespaces))

        state_sets = [
            {state.digest for state in pair.learner.states} for pair in pairs
        ]
        component_sets = [
            {component.digest for component in pair.learner.components}
            for pair in pairs
        ]
        action_sets = [
            {action.digest for action in pair.learner.grounded_candidates}
            for pair in pairs
        ]
        address_sets = [
            {
                suite._pair_address(state.digest, action.digest)
                for state in pair.learner.states
                for action in pair.learner.grounded_candidates
            }
            for pair in pairs
        ]
        for values in (state_sets, component_sets, action_sets, address_sets):
            for index, left in enumerate(values):
                self.assertTrue(
                    all(left.isdisjoint(right) for right in values[index + 1 :])
                )

        support_count = len(self.stream.supports)
        for values in (state_sets, component_sets, action_sets, address_sets):
            support_union = set().union(*values[:support_count])
            query_union = set().union(*values[support_count:])
            self.assertTrue(support_union.isdisjoint(query_union))

    def test_supports_are_separate_and_queries_are_novel_compositions(self) -> None:
        support_motifs = [pair.hidden.required_motifs for pair in self.stream.supports]
        query_motifs = {pair.hidden.required_motifs for pair in self.stream.queries}

        self.assertTrue(all(len(value) == 1 for value in support_motifs))
        self.assertEqual(len(query_motifs), 1)
        composed = next(iter(query_motifs))
        self.assertEqual(len(composed), 2)
        self.assertEqual({value[0] for value in support_motifs}, set(composed))
        self.assertNotIn(composed, support_motifs)
        self.assertTrue(all(pair.learner.observations for pair in self.stream.supports))
        self.assertTrue(all(not pair.learner.observations for pair in self.stream.queries))

    def test_public_projection_is_typed_bounded_and_private_free(self) -> None:
        self.assertEqual(
            {field.name for field in fields(suite.PublicSoftwarePipelineTask)},
            {
                "components",
                "grounded_candidates",
                "states",
                "observations",
                "origin",
                "required_output",
                "max_steps",
            },
        )
        for pair in (*self.stream.supports, *self.stream.queries):
            task = pair.learner
            self.assertIn(len(task.components), (3, 6))
            self.assertEqual(len(task.components), len(task.grounded_candidates))
            self.assertIn(len(task.states), (3, 9))
            self.assertIn(task.max_steps, (3, 4))
            self.assertIn(task.origin, task.states)
            self.assertTrue(task.required_output.exact)
            self.assertTrue(
                any(
                    state.records == task.required_output.required
                    for state in task.states
                )
            )
            public_json = json.dumps(task.to_canonical(), sort_keys=True)
            for forbidden in (
                "mechanism_commitment",
                "mechanism_partition",
                "package_commitment",
                "integration_inputs",
                "required_motifs",
                "implementations",
                "distractor",
                "semantic_index",
                "reference_pipeline",
                "failure_location",
            ):
                self.assertNotIn(forbidden, public_json)

    def test_queries_expose_complete_variant_blind_public_progress_closure(
        self,
    ) -> None:
        self.assertTrue(
            all(len(pair.learner.states) == 3 for pair in self.stream.supports)
        )
        for pair in self.stream.queries:
            task = pair.learner
            self.assertEqual(len(task.states), 9)
            self.assertEqual(task.observations, ())
            declared = set(task.states)
            for state in task.states:
                held = {
                    record.arguments[0]
                    for record in state.records
                    if record.predicate.endswith(".holds")
                }
                for action in task.grounded_candidates:
                    contract = _public_contract_for_action(task, action)
                    if set(contract.state_reads) <= held:
                        self.assertIn(
                            _public_successor(task, state, action),
                            declared,
                        )

            completions: dict[str, list] = {}
            for action in task.grounded_candidates:
                contract = _public_contract_for_action(task, action)
                completions.setdefault(contract.input_type, []).append(action)
            for twins in completions.values():
                if len(twins) != 2:
                    continue
                left, right = twins
                left_contract = _public_contract_for_action(task, left)
                right_contract = _public_contract_for_action(task, right)
                self.assertEqual(left_contract.state_reads, right_contract.state_reads)
                self.assertEqual(left_contract.state_writes, right_contract.state_writes)
                for state in task.states:
                    held = {
                        record.arguments[0]
                        for record in state.records
                        if record.predicate.endswith(".holds")
                    }
                    if set(left_contract.state_reads) <= held:
                        self.assertEqual(
                            _public_successor(task, state, left),
                            _public_successor(task, state, right),
                        )

    def test_counterfactual_twins_are_type_and_goal_compatible(self) -> None:
        for pair in self.stream.queries:
            contracts = {
                value.schema.digest: value for value in pair.learner.components
            }
            by_motif: dict[int, list] = {}
            for implementation in pair.hidden.implementations:
                if implementation.stage == 1:
                    by_motif.setdefault(implementation.motif, []).append(
                        contracts[
                            next(
                                action.schema.digest
                                for action in pair.learner.grounded_candidates
                                if action.digest == implementation.action_digest
                            )
                        ]
                    )
            for alternatives in by_motif.values():
                self.assertEqual(len(alternatives), 2)
                left, right = alternatives
                self.assertEqual(left.input_type, right.input_type)
                self.assertEqual(left.output_type, right.output_type)
                self.assertEqual(left.error_type, right.error_type)
                self.assertEqual(left.state_reads, right.state_reads)
                self.assertEqual(left.state_writes, right.state_writes)
                self.assertNotEqual(left.incidence, right.incidence)
                self.assertTrue(
                    set(left.state_writes).issubset(
                        {
                            record.arguments[0]
                            for record in pair.learner.required_output.required
                        }
                    )
                )

    def test_anonymous_topology_is_stable_under_rerender_and_not_labelled(self) -> None:
        rerendered = make_software_pipeline_stream(
            61_201,
            surface_seed=71_299,
            supports_per_motif=2,
            queries=3,
            mechanism_commitment=self.commitment,
            mechanism_partition="development",
        )

        def topology_counts(pair):
            counts = []
            implementations = {
                value.action_digest: value for value in pair.hidden.implementations
            }
            for action in pair.learner.grounded_candidates:
                contract = next(
                    value
                    for value in pair.learner.components
                    if value.schema == action.schema
                )
                relation_count = sum(
                    record.predicate.endswith(".relates")
                    for record in contract.incidence
                )
                private = implementations[action.digest]
                counts.append(
                    (private.motif, private.stage, private.distractor, relation_count)
                )
            return sorted(counts)

        for left, right in zip(
            (*self.stream.supports, *self.stream.queries),
            (*rerendered.supports, *rerendered.queries),
            strict=True,
        ):
            self.assertEqual(topology_counts(left), topology_counts(right))
            projection = json.dumps(left.learner.to_canonical(), sort_keys=True)
            self.assertNotIn("shape_", projection)
            self.assertNotIn("data_flow", projection)
            self.assertNotIn("execution_order", projection)
            self.assertNotIn("branch_routing", projection)
            self.assertNotIn("stale_invalidation", projection)
            self.assertNotIn("error_propagation", projection)

    def test_component_scalar_aggregates_do_not_encode_structural_role(self) -> None:
        expected = {
            (
                0,
                1,
                1,
                27,
                (2, 2, 2, 2, 2, *([3] * 22)),
                22,
            )
        }
        for pair in (*self.stream.supports, *self.stream.queries):
            signatures = {
                (
                    len(component.schema.parameters),
                    len(component.state_reads),
                    len(component.state_writes),
                    len(component.incidence),
                    tuple(
                        sorted(
                            len(record.arguments)
                            for record in component.incidence
                        )
                    ),
                    _relation_count(component),
                )
                for component in pair.learner.components
            }
            self.assertEqual(signatures, expected)

    def test_graph_transforms_share_nodes_and_preserve_degree_aggregates(self) -> None:
        for pair in (*self.stream.supports, *self.stream.queries):
            by_motif: dict[int, list] = {}
            for implementation in pair.hidden.implementations:
                action = next(
                    value
                    for value in pair.learner.grounded_candidates
                    if value.digest == implementation.action_digest
                )
                by_motif.setdefault(implementation.motif, []).append(
                    _public_contract_for_action(pair.learner, action)
                )
            for contracts in by_motif.values():
                self.assertEqual(len(contracts), 3)
                node_sets = []
                degree_profiles = []
                signatures = []
                for contract in contracts:
                    edges = [
                        (record.arguments[1], record.arguments[2])
                        for record in contract.incidence
                        if record.predicate.endswith(".relates")
                    ]
                    nodes = {value for edge in edges for value in edge}
                    node_sets.append(nodes)
                    degree_profiles.append(
                        tuple(
                            sorted(
                                (
                                    sum(target == node for _, target in edges),
                                    sum(source == node for source, _ in edges),
                                )
                                for node in nodes
                            )
                        )
                    )
                    signatures.append(_topology_signature(contract))
                    self.assertEqual(len(nodes), 18)
                    self.assertEqual(len(edges), 22)
                self.assertEqual(node_sets[0], node_sets[1])
                self.assertEqual(node_sets[0], node_sets[2])
                self.assertEqual(degree_profiles[0], degree_profiles[1])
                self.assertEqual(degree_profiles[0], degree_profiles[2])
                self.assertEqual(len(set(signatures)), 3)

    def test_support_and_query_wl_signatures_have_zero_overlap(self) -> None:
        for supports_per_motif in (1, 4):
            stream = make_software_pipeline_stream(
                61_211 + supports_per_motif,
                surface_seed=71_211 + supports_per_motif,
                supports_per_motif=supports_per_motif,
                queries=4,
                mechanism_commitment=self.commitment,
                mechanism_partition="development",
            )
            support_keys = {
                _topology_signature(component)
                for pair in stream.supports
                for component in pair.learner.components
            }
            query_keys = {
                _topology_signature(component)
                for pair in stream.queries
                for component in pair.learner.components
            }
            self.assertEqual(
                len(support_keys),
                sum(len(pair.learner.components) for pair in stream.supports),
            )
            self.assertEqual(
                len(query_keys),
                sum(len(pair.learner.components) for pair in stream.queries),
            )
            self.assertTrue(support_keys.isdisjoint(query_keys))

    def test_forward_and_backward_candidate_marginals_are_exactly_exchangeable(self) -> None:
        sources = (0, 3, 8, 13)
        destinations = (2, 6, 11, 16)
        forward = set()
        backward = set()
        for assignment in itertools.permutations(destinations):
            _, plus, minus = suite._cyclic_destination_transforms(
                sources,
                assignment,
            )
            forward.add(tuple(sorted(plus)))
            backward.add(tuple(sorted(minus)))
        self.assertEqual(len(forward), 24)
        self.assertEqual(forward, backward)

    def test_relational_topologies_survive_rerender_without_scalar_codes(self) -> None:
        rerendered = make_software_pipeline_stream(
            61_201,
            surface_seed=71_301,
            supports_per_motif=2,
            queries=3,
            mechanism_commitment=self.commitment,
            mechanism_partition="development",
        )
        for left, right in zip(
            (*self.stream.supports, *self.stream.queries),
            (*rerendered.supports, *rerendered.queries),
            strict=True,
        ):
            left_signatures = sorted(
                _topology_signature(value) for value in left.learner.components
            )
            right_signatures = sorted(
                _topology_signature(value) for value in right.learner.components
            )
            self.assertEqual(left_signatures, right_signatures)
            self.assertEqual(len(set(left_signatures)), len(left_signatures))

    def test_public_digest_ignores_presentation_order(self) -> None:
        task = self.stream.queries[0].learner
        reordered = replace(
            task,
            components=tuple(reversed(task.components)),
            grounded_candidates=tuple(reversed(task.grounded_candidates)),
            states=tuple(reversed(task.states)),
            observations=tuple(reversed(task.observations)),
        )
        self.assertEqual(task.to_canonical(), reordered.to_canonical())
        self.assertEqual(suite._public_digest(task), suite._public_digest(reordered))


class SoftwarePipelineShortcutRegressionTests(unittest.TestCase):
    def test_incidence_count_plus_type_chain_is_bounded_near_chance(self) -> None:
        commitments = software_pipeline_mechanism_partition("development")
        by_support_count: dict[int, list[float]] = {1: [], 4: []}
        for supports_per_motif in by_support_count:
            for commitment_index, commitment in enumerate(commitments):
                for surface_index in range(4):
                    stream = make_software_pipeline_stream(
                        91_000 + commitment_index * 31 + surface_index,
                        surface_seed=(
                            101_000 + commitment_index * 47 + surface_index
                        ),
                        supports_per_motif=supports_per_motif,
                        queries=1,
                        maximum_steps=4,
                        mechanism_commitment=commitment,
                        mechanism_partition="development",
                    )
                    pair = stream.queries[0]
                    proposal = _count_and_type_chain_baseline(stream, pair)
                    by_support_count[supports_per_motif].append(
                        judge_software_pipeline_attempt(pair, proposal)
                    )

        for supports_per_motif, rewards in by_support_count.items():
            accuracy = sum(rewards) / len(rewards)
            self.assertLessEqual(
                accuracy,
                0.35,
                msg=f"supports_per_motif={supports_per_motif}",
            )

    def test_prior_exact_wl_retrieval_is_bounded_near_chance(self) -> None:
        commitments = software_pipeline_mechanism_partition("development")
        by_support_count: dict[int, list[float]] = {1: [], 4: []}
        for supports_per_motif in by_support_count:
            for commitment_index, commitment in enumerate(commitments):
                for surface_index in range(4):
                    stream = make_software_pipeline_stream(
                        92_000 + commitment_index * 31 + surface_index,
                        surface_seed=(
                            102_000 + commitment_index * 47 + surface_index
                        ),
                        supports_per_motif=supports_per_motif,
                        queries=1,
                        maximum_steps=4,
                        mechanism_commitment=commitment,
                        mechanism_partition="development",
                    )
                    pair = stream.queries[0]
                    proposal = _wl_signature_retrieval_baseline(stream, pair)
                    by_support_count[supports_per_motif].append(
                        judge_software_pipeline_attempt(pair, proposal)
                    )

        for supports_per_motif, rewards in by_support_count.items():
            accuracy = sum(rewards) / len(rewards)
            self.assertLessEqual(
                accuracy,
                0.35,
                msg=f"supports_per_motif={supports_per_motif}",
            )

    def test_candidate_only_graphlet_knn_is_bounded_near_chance(self) -> None:
        commitments = software_pipeline_mechanism_partition("development")
        rewards = []
        for surface_variant in range(2):
            for supports_per_motif in (1, 4):
                for mechanism_index, commitment in enumerate(commitments):
                    seed = (
                        950_000
                        + surface_variant * 10_000
                        + supports_per_motif * 1_000
                        + mechanism_index
                    )
                    stream = make_software_pipeline_stream(
                        seed,
                        surface_seed=seed + 111,
                        supports_per_motif=supports_per_motif,
                        queries=8,
                        maximum_steps=4,
                        mechanism_commitment=commitment,
                        mechanism_partition="development",
                    )
                    for pair in stream.queries:
                        proposal = _candidate_graphlet_knn_baseline(stream, pair)
                        rewards.append(
                            judge_software_pipeline_attempt(pair, proposal)
                        )
        self.assertEqual(len(rewards), 512)
        self.assertLessEqual(sum(rewards) / len(rewards), 0.30)

    def test_public_joint_relation_remains_complete_across_fresh_graphs(self) -> None:
        rewards = []
        for mechanism_index, commitment in enumerate(
            software_pipeline_mechanism_partition("development")
        ):
            stream = make_software_pipeline_stream(
                980_000 + mechanism_index,
                surface_seed=990_000 + mechanism_index,
                supports_per_motif=1,
                queries=4,
                maximum_steps=4,
                mechanism_commitment=commitment,
                mechanism_partition="development",
            )
            for pair in stream.queries:
                proposal = _joint_relation_oracle(stream, pair)
                rewards.append(judge_software_pipeline_attempt(pair, proposal))
        self.assertEqual(len(rewards), 64)
        self.assertEqual(sum(rewards) / len(rewards), 1.0)


class SoftwarePipelineControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stream = make_software_pipeline_stream(
            61_301,
            surface_seed=71_301,
            mechanism_commitment=software_pipeline_mechanism_partition(
                "development"
            )[4],
            mechanism_partition="development",
        )

    def test_public_controls_preserve_task_and_compute_except_evidence(self) -> None:
        no_evidence = make_software_pipeline_control_stream(
            self.stream,
            "no_evidence",
        )
        wrong = make_software_pipeline_control_stream(
            self.stream,
            "wrong_evidence",
        )
        shuffled = make_software_pipeline_control_stream(
            self.stream,
            "shuffled_outcome",
        )

        self.assertTrue(all(not value.learner.observations for value in no_evidence.supports))
        self.assertEqual(self.stream.queries, no_evidence.queries)
        self.assertEqual(self.stream.queries, wrong.queries)
        self.assertEqual(self.stream.queries, shuffled.queries)
        expected_query_states = tuple(
            {state.digest for state in pair.learner.states}
            for pair in self.stream.queries
        )
        for controlled in (no_evidence, wrong, shuffled):
            self.assertEqual(
                tuple(
                    {state.digest for state in pair.learner.states}
                    for pair in controlled.queries
                ),
                expected_query_states,
            )
        for original, no_item, wrong_item, shuffled_item in zip(
            self.stream.supports,
            no_evidence.supports,
            wrong.supports,
            shuffled.supports,
            strict=True,
        ):
            for altered in (no_item, wrong_item, shuffled_item):
                self.assertEqual(original.learner.components, altered.learner.components)
                self.assertEqual(
                    original.learner.grounded_candidates,
                    altered.learner.grounded_candidates,
                )
                self.assertEqual(original.learner.states, altered.learner.states)
                self.assertEqual(original.learner.origin, altered.learner.origin)
                self.assertEqual(
                    original.learner.required_output,
                    altered.learner.required_output,
                )
                self.assertEqual(original.learner.max_steps, altered.learner.max_steps)
                self.assertEqual(
                    original.hidden.integration_inputs,
                    altered.hidden.integration_inputs,
                )
            original_trace = original.learner.observations[0]
            wrong_trace = wrong_item.learner.observations[0]
            shuffled_trace = shuffled_item.learner.observations[0]
            self.assertEqual(
                tuple(value.after for value in original_trace.transitions),
                tuple(value.after for value in wrong_trace.transitions),
            )
            self.assertNotEqual(
                tuple(value.action for value in original_trace.transitions),
                tuple(value.action for value in wrong_trace.transitions),
            )
            self.assertNotEqual(
                tuple(value.after for value in original_trace.transitions),
                tuple(value.after for value in shuffled_trace.transitions),
            )

    def test_motif_pure_controls_are_owned_by_the_evaluator(self) -> None:
        a_only = make_software_pipeline_control_stream(self.stream, "a_only")
        b_only = make_software_pipeline_control_stream(self.stream, "b_only")
        query_motifs = self.stream.queries[0].hidden.required_motifs

        self.assertEqual(
            {pair.hidden.required_motifs for pair in a_only.supports},
            {(query_motifs[0],)},
        )
        self.assertEqual(
            {pair.hidden.required_motifs for pair in b_only.supports},
            {(query_motifs[1],)},
        )
        self.assertEqual(a_only.queries, self.stream.queries)
        self.assertEqual(b_only.queries, self.stream.queries)
        self.assertTrue(set(a_only.supports).isdisjoint(b_only.supports))

    def test_invalid_control_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid"):
            make_software_pipeline_control_stream(self.stream, "random")  # type: ignore[arg-type]


class SoftwarePipelineJudgingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stream = make_software_pipeline_stream(
            61_401,
            surface_seed=71_401,
            queries=2,
            mechanism_commitment=software_pipeline_mechanism_partition(
                "development"
            )[6],
            mechanism_partition="development",
        )
        self.pair = self.stream.queries[0]

    def test_equivalent_serial_and_interleaved_pipelines_are_accepted(self) -> None:
        serial = _commit_exact(self.pair)
        interleaved = _commit_exact(self.pair, interleave=True)

        serial_score = judge_software_pipeline_attempt(self.pair, serial)
        interleaved_score = judge_software_pipeline_attempt(self.pair, interleaved)
        self.assertIsInstance(serial_score, float)
        self.assertEqual(serial_score, 1.0)
        self.assertEqual(interleaved_score, 1.0)
        self.assertNotEqual(serial.actions, interleaved.actions)

    def test_all_six_stage_preserving_interleavings_have_declared_prefixes(
        self,
    ) -> None:
        action_orders = _all_stage_preserving_actions(self.pair)
        self.assertEqual(len(action_orders), 6)
        self.assertEqual(len(set(action_orders)), 6)
        declared = set(self.pair.learner.states)
        for actions in action_orders:
            state = self.pair.learner.origin
            for action in actions:
                state = _public_successor(self.pair.learner, state, action)
                self.assertIn(state, declared)
            pipeline = commit_software_pipeline(
                self.pair.learner,
                actions,
                stopped=False,
            )
            self.assertEqual(
                judge_software_pipeline_attempt(self.pair, pipeline),
                1.0,
            )

    def test_wrong_order_and_counterfactual_component_fail_terminally(self) -> None:
        actions = list(_actions_for_solution(self.pair))
        actions[0], actions[1] = actions[1], actions[0]
        wrong_order = commit_software_pipeline(
            self.pair.learner,
            actions,
            stopped=False,
        )
        self.assertEqual(
            judge_software_pipeline_attempt(self.pair, wrong_order),
            0.0,
        )

        correct = list(_actions_for_solution(self.pair))
        correct_digests = {value.digest for value in correct}
        alternative = next(
            value
            for value in self.pair.learner.grounded_candidates
            if value.digest not in correct_digests
            and any(
                implementation.action_digest == value.digest
                and implementation.stage == 1
                for implementation in self.pair.hidden.implementations
            )
        )
        implementation = next(
            value
            for value in self.pair.hidden.implementations
            if value.action_digest == alternative.digest
        )
        motif_index = self.pair.hidden.required_motifs.index(implementation.motif)
        correct[motif_index * 2 + 1] = alternative
        wrong_component = commit_software_pipeline(
            self.pair.learner,
            correct,
            stopped=False,
        )
        self.assertEqual(
            judge_software_pipeline_attempt(self.pair, wrong_component),
            0.0,
        )

    def test_public_query_alone_is_counterfactually_underdetermined(self) -> None:
        twin_solution = suite._counterfactual_solution(self.pair.hidden)
        twin_pair = replace(self.pair, hidden=twin_solution)
        original_pipeline = _commit_exact(self.pair)
        twin_pipeline = _commit_exact(self.pair, twin_solution)

        self.assertIs(twin_pair.learner, self.pair.learner)
        self.assertEqual(
            suite._public_digest(twin_pair.learner),
            suite._public_digest(self.pair.learner),
        )
        self.assertEqual(
            judge_software_pipeline_attempt(self.pair, original_pipeline),
            1.0,
        )
        self.assertEqual(
            judge_software_pipeline_attempt(twin_pair, original_pipeline),
            0.0,
        )
        self.assertEqual(
            judge_software_pipeline_attempt(self.pair, twin_pipeline),
            0.0,
        )
        self.assertEqual(
            judge_software_pipeline_attempt(twin_pair, twin_pipeline),
            1.0,
        )

    def test_commit_is_immutable_bounded_and_declared_only(self) -> None:
        exact = _commit_exact(self.pair)
        with self.assertRaises(FrozenInstanceError):
            exact.stopped = True  # type: ignore[misc]

        task = self.pair.learner
        repeated = (task.grounded_candidates[0],) * (task.max_steps + 1)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            commit_software_pipeline(task, repeated, stopped=False)
        with self.assertRaisesRegex(ValueError, "explicit STOP"):
            commit_software_pipeline(task, (), stopped=False)
        undeclared = ActionSchema(
            "angler.microrepo_unknown.component_deadbeef",
            (),
        ).ground()
        with self.assertRaisesRegex(ValueError, "undeclared"):
            commit_software_pipeline(task, (undeclared,), stopped=True)

    def test_cross_task_binding_is_rejected(self) -> None:
        exact = _commit_exact(self.pair)
        other = self.stream.queries[1]
        with self.assertRaisesRegex(ValueError, "do not match"):
            judge_software_pipeline_attempt(other, exact)


class SoftwarePipelineNoLeakSurfaceTests(unittest.TestCase):
    def test_public_api_is_generation_commit_controls_and_scalar_judging_only(
        self,
    ) -> None:
        forbidden = (
            "solve",
            "solver",
            "search",
            "shortest",
            "distance",
            "next_action",
            "reference_pipeline",
            "target_pipeline",
            "failure_location",
            "integration_input",
        )
        for name in suite.__all__:
            self.assertFalse(any(fragment in name.lower() for fragment in forbidden))
        public_functions = {
            name
            for name in suite.__all__
            if inspect.isfunction(getattr(suite, name, None))
        }
        self.assertEqual(
            public_functions,
            {
                "commit_software_pipeline",
                "judge_software_pipeline_attempt",
                "make_software_pipeline_control_stream",
                "make_software_pipeline_stream",
                "software_pipeline_mechanism_partition",
            },
        )
        self.assertNotIn("_HiddenSoftwarePipelineSolution", suite.__all__)
        self.assertNotIn("_PrivateComponentImplementation", suite.__all__)


if __name__ == "__main__":
    unittest.main()
