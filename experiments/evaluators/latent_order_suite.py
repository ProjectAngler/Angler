"""Evaluator-only latent programs loaded after the candidate is frozen.

These exact operator trees are deliberately outside ``angler.worlds`` and
are not part of the learner-facing package API.  The prequential runner imports
this module only after meta-training has finished and the slow-state digest has
been captured.
"""

from __future__ import annotations

from angler.worlds.latent_order_programs import OrderingProgram


def _leaf(name: str) -> OrderingProgram:
    return OrderingProgram(name)


def _unary(name: str, child: OrderingProgram) -> OrderingProgram:
    return OrderingProgram(name, (child,))


def _conditional(
    when_false: OrderingProgram,
    when_true: OrderingProgram,
) -> OrderingProgram:
    return OrderingProgram("IF_FLAG", (when_false, when_true))


def evaluator_programs() -> tuple[OrderingProgram, ...]:
    """Return the exact changing-mechanism evaluation sequence candidates."""

    a_asc = _leaf("A_ASC")
    a_desc = _leaf("A_DESC")
    b_asc = _leaf("B_ASC")
    b_desc = _leaf("B_DESC")
    return (
        _conditional(
            _unary("ROTATE", a_asc),
            _unary("GROUP_10", b_desc),
        ),
        _unary("ROTATE", _unary("ZIGZAG", _unary("GROUP_01", a_desc))),
        _unary(
            "GROUP_10",
            _conditional(b_asc, _unary("ZIGZAG", a_asc)),
        ),
        _unary(
            "ZIGZAG",
            _conditional(
                _unary("ROTATE", b_desc),
                _unary("GROUP_01", a_asc),
            ),
        ),
    )


__all__ = ["evaluator_programs"]
