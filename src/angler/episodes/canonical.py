"""ANG-CANON-JSON-001@1 canonical JSON and evidence identity helpers."""

from __future__ import annotations

import hashlib
import hmac
import json
import unicodedata
from typing import Any, Mapping

CANONICALIZATION_PROFILE = "ANG-CANON-JSON-001@1"
_PROTECTED_MODES = {"HMAC_SHA256", "AUTHENTICATED_CIPHERTEXT_SHA256"}


class CanonicalizationError(ValueError):
    """Typed, non-payload-bearing canonicalization failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _reject_float(_: str) -> None:
    raise CanonicalizationError("EVIDENCE_AMBIGUOUS_NUMBER")


def _reject_constant(_: str) -> None:
    raise CanonicalizationError("EVIDENCE_NONFINITE_NUMBER")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw_key, value in pairs:
        key = unicodedata.normalize("NFC", raw_key)
        if key in result:
            code = (
                "EVIDENCE_UNICODE_KEY_COLLISION"
                if raw_key != key
                else "EVIDENCE_DUPLICATE_KEY"
            )
            raise CanonicalizationError(code)
        result[key] = value
    return result


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        raise CanonicalizationError("EVIDENCE_AMBIGUOUS_NUMBER")
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise CanonicalizationError("EVIDENCE_NON_STRING_KEY")
            key = unicodedata.normalize("NFC", raw_key)
            if key in normalized:
                raise CanonicalizationError("EVIDENCE_UNICODE_KEY_COLLISION")
            normalized[key] = _normalize(item)
        return normalized
    raise CanonicalizationError("EVIDENCE_UNSUPPORTED_JSON_TYPE")


def parse_json(raw: str | bytes) -> Any:
    """Parse JSON while rejecting ambiguity before canonicalization."""

    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise CanonicalizationError("EVIDENCE_INVALID_UTF8") from exc
    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_float=_reject_float,
            parse_int=int,
            parse_constant=_reject_constant,
        )
    except CanonicalizationError:
        raise
    except (json.JSONDecodeError, TypeError) as exc:
        raise CanonicalizationError("EVIDENCE_INVALID_JSON") from exc
    return _normalize(parsed)


def canonical_bytes(value: Any) -> bytes:
    """Return the unique UTF-8 representation for supported JSON values."""

    normalized = _normalize(value)
    try:
        text = json.dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return text.encode("utf-8", errors="strict")
    except (UnicodeEncodeError, ValueError) as exc:
        raise CanonicalizationError("EVIDENCE_INVALID_UNICODE") from exc


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def compute_content_id(
    payload: Any,
    *,
    mode: str = "SHA256",
    key: bytes | None = None,
) -> str:
    """Commit to a payload under the declared public or protected mode."""

    data = canonical_bytes(payload)
    if mode == "SHA256" or mode == "AUTHENTICATED_CIPHERTEXT_SHA256":
        return _digest(data)
    if mode == "HMAC_SHA256":
        if not key:
            raise CanonicalizationError("EVIDENCE_COMMITMENT_KEY_REQUIRED")
        return "hmac-sha256:" + hmac.new(key, data, hashlib.sha256).hexdigest()
    raise CanonicalizationError("EVIDENCE_COMMITMENT_MODE_UNSUPPORTED")


def envelope_identity_material(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only fields explicitly excluded from envelope identity."""

    material = _normalize(dict(envelope))
    material.pop("artifact_id", None)
    material.pop("attestations", None)
    visibility = material.get("visibility")
    if isinstance(visibility, dict):
        visibility.pop("opaque_payload_ref", None)
    return material


def compute_artifact_id(envelope: Mapping[str, Any]) -> str:
    return _digest(canonical_bytes(envelope_identity_material(envelope)))


def verify_content_id(
    payload: Any,
    declared: str,
    *,
    mode: str = "SHA256",
    key: bytes | None = None,
) -> bool:
    expected = compute_content_id(payload, mode=mode, key=key)
    return hmac.compare_digest(expected, declared)


def verify_artifact_id(envelope: Mapping[str, Any]) -> bool:
    declared = envelope.get("artifact_id")
    return isinstance(declared, str) and hmac.compare_digest(
        compute_artifact_id(envelope), declared
    )


def supported_major(version: str, expected: int = 1) -> bool:
    try:
        major_text, _minor, _patch = version.split(".", maxsplit=2)
        return int(major_text) == expected
    except (AttributeError, TypeError, ValueError):
        return False


def is_digest_id(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    prefixes = ("sha256:", "hmac-sha256:")
    prefix = next((item for item in prefixes if value.startswith(item)), None)
    if prefix is None:
        return False
    digest = value[len(prefix) :]
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


def is_protected_mode(mode: str) -> bool:
    return mode in _PROTECTED_MODES
