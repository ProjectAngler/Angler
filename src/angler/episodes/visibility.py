"""Visibility-policy checks with constant-shape denials."""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Mapping


class VisibilityClass(str, Enum):
    LEARNER_VISIBLE = "LEARNER_VISIBLE"
    CONTROL_PLANE = "CONTROL_PLANE"
    SEALED_EVALUATION = "SEALED_EVALUATION"
    HUMAN_AUTHORITY = "HUMAN_AUTHORITY"
    RESTRICTED_PERSONAL = "RESTRICTED_PERSONAL"


_RESTRICTIVENESS = {
    VisibilityClass.LEARNER_VISIBLE: 0,
    VisibilityClass.CONTROL_PLANE: 1,
    VisibilityClass.SEALED_EVALUATION: 2,
    VisibilityClass.HUMAN_AUTHORITY: 3,
    VisibilityClass.RESTRICTED_PERSONAL: 4,
}
_LEARNER_PRINCIPAL = "ANG-AUTH-LEARNER-001"


class VisibilityDenied(PermissionError):
    """A denial whose public representation contains no protected detail."""

    code = "EVIDENCE_VISIBILITY_DENIED"

    def __init__(self) -> None:
        super().__init__(self.code)

    def public_record(self) -> dict[str, bool | str]:
        return {"code": self.code, "denied": True}


def _string_set(value: Any, error_code: str) -> set[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(error_code)
    if len(value) != len(set(value)):
        raise ValueError(error_code)
    return set(value)


def validate_visibility_policy(
    policy: Mapping[str, Any], *, cr0: bool = True
) -> dict[str, Any]:
    required = {
        "class",
        "policy_id_and_version",
        "allowed_payload_principals",
        "allowed_projection_ids",
        "opaque_payload_ref",
    }
    if not isinstance(policy, Mapping) or not required.issubset(policy):
        raise ValueError("EVIDENCE_VISIBILITY_POLICY_INVALID")
    try:
        visibility_class = VisibilityClass(policy["class"])
    except (TypeError, ValueError) as exc:
        raise ValueError("EVIDENCE_VISIBILITY_POLICY_INVALID") from exc
    if cr0 and visibility_class is VisibilityClass.RESTRICTED_PERSONAL:
        raise ValueError("EVIDENCE_PERSONAL_DATA_FORBIDDEN")
    principals = _string_set(
        policy["allowed_payload_principals"], "EVIDENCE_VISIBILITY_POLICY_INVALID"
    )
    projections = _string_set(
        policy["allowed_projection_ids"], "EVIDENCE_VISIBILITY_POLICY_INVALID"
    )
    opaque_ref = policy["opaque_payload_ref"]
    if not isinstance(opaque_ref, str) or not opaque_ref.startswith("opaque:"):
        raise ValueError("EVIDENCE_PAYLOAD_LOCATOR_NOT_OPAQUE")
    if visibility_class is not VisibilityClass.LEARNER_VISIBLE and _LEARNER_PRINCIPAL in principals:
        raise ValueError("EVIDENCE_LEARNER_VISIBILITY_FORBIDDEN")
    return {
        "class": visibility_class.value,
        "policy_id_and_version": policy["policy_id_and_version"],
        "allowed_payload_principals": sorted(principals),
        "allowed_projection_ids": sorted(projections),
        "opaque_payload_ref": opaque_ref,
    }


def authorize_projection(
    policy: Mapping[str, Any],
    *,
    principal: str,
    projection_id: str,
    purpose: str,
) -> bool:
    """Authorize a read projection without returning payload-derived details."""

    try:
        checked = validate_visibility_policy(policy)
    except ValueError as exc:
        raise VisibilityDenied() from exc
    allowed_principals = set(checked["allowed_payload_principals"])
    allowed_projections = set(checked["allowed_projection_ids"])
    if (
        not isinstance(purpose, str)
        or not purpose
        or principal not in allowed_principals
        or projection_id not in allowed_projections
    ):
        raise VisibilityDenied()
    if (
        principal == _LEARNER_PRINCIPAL
        and checked["class"] != VisibilityClass.LEARNER_VISIBLE.value
    ):
        raise VisibilityDenied()
    return True


def may_mutate(
    *,
    principal: str,
    producer_principal: str,
    delegated_writers: Iterable[str] = (),
) -> bool:
    """Visibility is intentionally absent from this write-authority check."""

    return principal == producer_principal or principal in set(delegated_writers)


def combine_policies(policies: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    checked = [validate_visibility_policy(item) for item in policies]
    if not checked:
        raise ValueError("EVIDENCE_VISIBILITY_POLICY_INVALID")
    strictest = max(
        (VisibilityClass(item["class"]) for item in checked),
        key=lambda item: _RESTRICTIVENESS[item],
    )
    principals = set(checked[0]["allowed_payload_principals"])
    projections = set(checked[0]["allowed_projection_ids"])
    for item in checked[1:]:
        principals.intersection_update(item["allowed_payload_principals"])
        projections.intersection_update(item["allowed_projection_ids"])
    return {
        "class": strictest.value,
        "policy_id_and_version": "ANG-POL-VISIBILITY-INTERSECTION-001@1",
        "allowed_payload_principals": sorted(principals),
        "allowed_projection_ids": sorted(projections),
        "opaque_payload_ref": "opaque:combined",
    }
