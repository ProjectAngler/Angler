"""Public procedural-world API.

Evaluator-private latent-program internals are intentionally not re-exported
from this learner-facing namespace.
"""
from .procedural_constraints import (
    FAMILY_ID,
    FAMILY_VERSION,
    DEFAULT_RELATION_SURFACE_FORMS,
    GeneratedRelationalTask,
    HiddenOrderSolution,
    LearnerTask,
    OutcomeFeedback,
    PrecedenceConstraint,
    ScalarOutcomeFeedback,
    generate_relational_task,
    make_held_out_variant,
    score_constraint_satisfaction,
    verify_final_answer,
)
from .symbolic_rule_induction import (
    DEFAULT_DEMONSTRATION_SURFACE_FORMS,
    DEFAULT_GOAL_SURFACE_FORMS,
    DEFAULT_QUERY_SURFACE_FORMS,
    FAMILY_ID as SYMBOLIC_RULE_FAMILY_ID,
    FAMILY_VERSION as SYMBOLIC_RULE_FAMILY_VERSION,
    GeneratedSymbolicRuleTask,
    HiddenSymbolicRuleSolution,
    SymbolicRuleDemonstration,
    SymbolicRuleFeedback,
    SymbolicRuleLearnerTask,
    generate_symbolic_rule_task,
    verify_symbolic_rule_answer,
)

__all__ = [
    "FAMILY_ID",
    "FAMILY_VERSION",
    "DEFAULT_RELATION_SURFACE_FORMS",
    "GeneratedRelationalTask",
    "HiddenOrderSolution",
    "LearnerTask",
    "OutcomeFeedback",
    "PrecedenceConstraint",
    "ScalarOutcomeFeedback",
    "generate_relational_task",
    "make_held_out_variant",
    "score_constraint_satisfaction",
    "verify_final_answer",
    "DEFAULT_DEMONSTRATION_SURFACE_FORMS",
    "DEFAULT_GOAL_SURFACE_FORMS",
    "DEFAULT_QUERY_SURFACE_FORMS",
    "SYMBOLIC_RULE_FAMILY_ID",
    "SYMBOLIC_RULE_FAMILY_VERSION",
    "GeneratedSymbolicRuleTask",
    "HiddenSymbolicRuleSolution",
    "SymbolicRuleDemonstration",
    "SymbolicRuleFeedback",
    "SymbolicRuleLearnerTask",
    "generate_symbolic_rule_task",
    "verify_symbolic_rule_answer",
]
