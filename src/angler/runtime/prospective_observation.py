"""Frozen post-execution observation encoding for the successor cycle.

This module is deliberately a representation boundary, not an evaluator.  Its
synthetic encoder commits only to the exact public execution request and
receipt bytes.  Objective outcomes, predictions, task-family metadata, and
authorization state are neither accepted nor inferred here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import ClassVar, Protocol, runtime_checkable

from angler.cognition.prospective import (
    CognitiveExecutionReceipt,
    CognitiveExecutionRequest,
)
from angler.episodes.canonical import canonical_bytes


SYNTHETIC_OBSERVED_STATE_ALGORITHM = "angler.synthetic-observed-state.v1"
QWEN_RECEIPT_OBSERVED_STATE_ALGORITHM = (
    "angler.qwen-receipt-observed-state.v1"
)
MAX_SYNTHETIC_OBSERVED_LATENT_WIDTH = 1_024
_DOMAIN = b"angler.synthetic-observed-state.v1\0"
_QWEN_DOMAIN = b"angler.qwen-receipt-observed-state.v1\0"
_SHA256_PREFIX = "sha256:"


def _latent_width(value: object, label: str = "latent_width") -> int:
    if type(value) is not int or not 1 <= value <= MAX_SYNTHETIC_OBSERVED_LATENT_WIDTH:
        raise ValueError(
            f"{label} must be an exact integer from 1 through "
            f"{MAX_SYNTHETIC_OBSERVED_LATENT_WIDTH}"
        )
    return value


def _sha256_ref(value: object, label: str) -> str:
    if (
        type(value) is not str
        or not value.startswith(_SHA256_PREFIX)
        or len(value) != len(_SHA256_PREFIX) + 64
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{label} must be a lowercase sha256 reference")
    return value


def _dyadic_from_leading_u64(value: int) -> float:
    """Map one unsigned 64-bit prefix to the frozen binary64 dyadic."""

    if type(value) is not int or not 0 <= value < 2**64:
        raise ValueError("leading digest value must be an unsigned 64-bit integer")
    top53 = value >> 11
    return (top53 / 2**52) - 1.0


def _assert_completed_context(
    execution_request: CognitiveExecutionRequest,
    execution_receipt: CognitiveExecutionReceipt,
) -> None:
    if type(execution_request) is not CognitiveExecutionRequest:
        raise TypeError("execution_request must be an exact CognitiveExecutionRequest")
    if type(execution_receipt) is not CognitiveExecutionReceipt:
        raise TypeError("execution_receipt must be an exact CognitiveExecutionReceipt")
    if execution_receipt.status != "COMPLETED":
        raise ValueError("observed-state encoding requires a completed receipt")
    execution_receipt.assert_request(execution_request)


@dataclass(frozen=True, slots=True)
class ObservedStateEncoding:
    """One immutable observed latent and its exact public evidence boundary."""

    encoder_ref: str
    latent_width: int
    latent: tuple[float, ...]
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        _sha256_ref(self.encoder_ref, "encoder_ref")
        width = _latent_width(self.latent_width)
        if type(self.latent) is not tuple:
            raise TypeError("latent must be an immutable tuple")
        if len(self.latent) != width:
            raise ValueError("latent width does not match latent_width")
        if any(type(value) is not float or not math.isfinite(value) for value in self.latent):
            raise ValueError("latent must contain only finite exact floats")
        if type(self.evidence_refs) is not tuple:
            raise TypeError("evidence_refs must be an immutable tuple")
        if len(self.evidence_refs) != 2:
            raise ValueError("evidence_refs must contain exactly request and receipt refs")
        for value in self.evidence_refs:
            _sha256_ref(value, "evidence_refs item")
        if len(set(self.evidence_refs)) != 2:
            raise ValueError("request and receipt evidence refs must be distinct")
        if self.evidence_refs != tuple(sorted(self.evidence_refs)):
            raise ValueError("evidence_refs must be canonically sorted")

    @property
    def observed_next_latent(self) -> tuple[float, ...]:
        """The exact tuple accepted by ``ProspectiveResolution.observed``."""

        return self.latent

    @property
    def observation_evidence_refs(self) -> tuple[str, ...]:
        return self.evidence_refs

    def assert_context(
        self,
        execution_request: CognitiveExecutionRequest,
        execution_receipt: CognitiveExecutionReceipt,
        *,
        encoder_ref: str,
        latent_width: int,
    ) -> None:
        """Fail closed if a caller tries to reuse this value in another context."""

        _assert_completed_context(execution_request, execution_receipt)
        expected_encoder_ref = _sha256_ref(encoder_ref, "encoder_ref")
        expected_width = _latent_width(latent_width)
        expected_evidence = tuple(
            sorted(
                (
                    execution_request.execution_request_ref,
                    execution_receipt.execution_receipt_ref,
                )
            )
        )
        if self.encoder_ref != expected_encoder_ref:
            raise ValueError("observed-state encoder identity drifted")
        if self.latent_width != expected_width:
            raise ValueError("observed-state latent width drifted")
        if self.evidence_refs != expected_evidence:
            raise ValueError("observed-state evidence does not match the exact execution")


@runtime_checkable
class FrozenObservedStateEncoder(Protocol):
    """Immutable post-execution representation boundary used by the cycle."""

    @property
    def latent_width(self) -> int: ...

    @property
    def encoder_ref(self) -> str: ...

    def encode(
        self,
        execution_request: CognitiveExecutionRequest,
        execution_receipt: CognitiveExecutionReceipt,
        *,
        latent_width: int,
    ) -> ObservedStateEncoding: ...


@dataclass(frozen=True, slots=True)
class SyntheticObservedStateEncoderV1:
    """Frozen, task-agnostic SHA-256 encoder for bounded synthetic tests."""

    latent_width: int

    ALGORITHM: ClassVar[str] = SYNTHETIC_OBSERVED_STATE_ALGORITHM

    def __post_init__(self) -> None:
        _latent_width(self.latent_width)

    @property
    def encoder_ref(self) -> str:
        identity = canonical_bytes(
            {"algorithm": self.ALGORITHM, "latent_width": self.latent_width}
        )
        return _SHA256_PREFIX + hashlib.sha256(identity).hexdigest()

    def encode(
        self,
        execution_request: CognitiveExecutionRequest,
        execution_receipt: CognitiveExecutionReceipt,
        *,
        latent_width: int,
    ) -> ObservedStateEncoding:
        required_width = _latent_width(latent_width, "required latent_width")
        if required_width != self.latent_width:
            raise ValueError("required latent_width differs from the frozen encoder")
        _assert_completed_context(execution_request, execution_receipt)

        request_bytes = execution_request.canonical_bytes()
        receipt_bytes = execution_receipt.canonical_bytes()
        framed = b"".join(
            (
                _DOMAIN,
                struct.pack(">Q", required_width),
                struct.pack(">Q", len(request_bytes)),
                request_bytes,
                struct.pack(">Q", len(receipt_bytes)),
                receipt_bytes,
            )
        )
        latent = tuple(
            _dyadic_from_leading_u64(
                int.from_bytes(
                    hashlib.sha256(framed + struct.pack(">Q", index)).digest()[:8],
                    byteorder="big",
                    signed=False,
                )
            )
            for index in range(required_width)
        )
        evidence_refs = tuple(
            sorted(
                (
                    execution_request.execution_request_ref,
                    execution_receipt.execution_receipt_ref,
                )
            )
        )
        encoding = ObservedStateEncoding(
            encoder_ref=self.encoder_ref,
            latent_width=required_width,
            latent=latent,
            evidence_refs=evidence_refs,
        )
        encoding.assert_context(
            execution_request,
            execution_receipt,
            encoder_ref=self.encoder_ref,
            latent_width=required_width,
        )
        return encoding


@dataclass(frozen=True, slots=True)
class QwenReceiptObservedStateEncoderV1:
    """Exact request/receipt encoder bound to one frozen-Qwen cycle manifest.

    This deliberately does not derive a semantic Qwen embedding after the
    effect.  The exact public Qwen response is already present in the receipt;
    hashing those canonical bytes keeps restart reproduction independent of
    floating hardware while the manifest binds the model-facing encoders.
    """

    model_ref: str
    encoder_ref: str
    manifest_ref: str
    latent_width: int

    ALGORITHM: ClassVar[str] = QWEN_RECEIPT_OBSERVED_STATE_ALGORITHM

    def __post_init__(self) -> None:
        _sha256_ref(self.model_ref, "model_ref")
        _sha256_ref(self.encoder_ref, "encoder_ref")
        _sha256_ref(self.manifest_ref, "manifest_ref")
        _latent_width(self.latent_width)

    def encode(
        self,
        execution_request: CognitiveExecutionRequest,
        execution_receipt: CognitiveExecutionReceipt,
        *,
        latent_width: int,
    ) -> ObservedStateEncoding:
        required_width = _latent_width(latent_width, "required latent_width")
        if required_width != self.latent_width:
            raise ValueError("required latent_width differs from the frozen encoder")
        _assert_completed_context(execution_request, execution_receipt)

        request_bytes = execution_request.canonical_bytes()
        receipt_bytes = execution_receipt.canonical_bytes()
        manifest_bytes = self.manifest_ref.encode("ascii", errors="strict")
        framed = b"".join(
            (
                _QWEN_DOMAIN,
                struct.pack(">Q", len(manifest_bytes)),
                manifest_bytes,
                struct.pack(">Q", required_width),
                struct.pack(">Q", len(request_bytes)),
                request_bytes,
                struct.pack(">Q", len(receipt_bytes)),
                receipt_bytes,
            )
        )
        latent = tuple(
            _dyadic_from_leading_u64(
                int.from_bytes(
                    hashlib.sha256(framed + struct.pack(">Q", index)).digest()[:8],
                    byteorder="big",
                    signed=False,
                )
            )
            for index in range(required_width)
        )
        evidence_refs = tuple(
            sorted(
                (
                    execution_request.execution_request_ref,
                    execution_receipt.execution_receipt_ref,
                )
            )
        )
        encoding = ObservedStateEncoding(
            encoder_ref=self.encoder_ref,
            latent_width=required_width,
            latent=latent,
            evidence_refs=evidence_refs,
        )
        encoding.assert_context(
            execution_request,
            execution_receipt,
            encoder_ref=self.encoder_ref,
            latent_width=required_width,
        )
        return encoding


__all__ = [
    "FrozenObservedStateEncoder",
    "MAX_SYNTHETIC_OBSERVED_LATENT_WIDTH",
    "ObservedStateEncoding",
    "QWEN_RECEIPT_OBSERVED_STATE_ALGORITHM",
    "QwenReceiptObservedStateEncoderV1",
    "SYNTHETIC_OBSERVED_STATE_ALGORITHM",
    "SyntheticObservedStateEncoderV1",
]
