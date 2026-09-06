"""Trusted clock samples and bi-temporal cognitive record boundaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import time
from typing import Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


TEMPORAL_NOW_CONTRACT = "ANG-CTR-TEMPORAL-NOW-001@0.2.0"
TEMPORAL_V2_CONTRACT = "ANG-CTR-TEMPORAL-V2-001@0.2.0"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock returned a naive datetime")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def parse_utc(value: str, label: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise ValueError(f"{label} must be RFC3339 UTC ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} must be RFC3339 UTC") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"{label} must be UTC")
    return parsed


@dataclass(frozen=True, slots=True)
class TemporalNow:
    contract: str
    trusted_utc: str
    local_time: str
    local_timezone: str
    local_utc_offset_seconds: int
    monotonic_ns: int
    clock_anchor_ref: str
    uncertainty_ms: float
    jump_detected: bool
    wall_elapsed_ms: float
    monotonic_elapsed_ms: float
    moving_origin_ordinal: int

    def __post_init__(self) -> None:
        if self.contract != TEMPORAL_NOW_CONTRACT:
            raise ValueError("unsupported TemporalNow contract")
        parse_utc(self.trusted_utc, "trusted_utc")
        if type(self.local_time) is not str:
            raise ValueError("local_time must be text")
        try:
            local = datetime.fromisoformat(self.local_time)
        except ValueError as exc:
            raise ValueError("local_time must be RFC3339 with an offset") from exc
        if local.utcoffset() is None:
            raise ValueError("local_time must carry its UTC offset")
        if type(self.local_timezone) is not str or not self.local_timezone:
            raise ValueError("local_timezone must be a non-empty IANA name")
        if type(self.local_utc_offset_seconds) is not int:
            raise TypeError("local_utc_offset_seconds must be an integer")
        if int(local.utcoffset().total_seconds()) != self.local_utc_offset_seconds:
            raise ValueError("local offset does not match local_time")
        if type(self.monotonic_ns) is not int or self.monotonic_ns < 0:
            raise ValueError("monotonic_ns must be non-negative")
        if not self.clock_anchor_ref.startswith("sha256:") or len(self.clock_anchor_ref) != 71:
            raise ValueError("clock_anchor_ref must be a SHA-256 reference")
        for label, value in (
            ("uncertainty_ms", self.uncertainty_ms),
            ("wall_elapsed_ms", self.wall_elapsed_ms),
            ("monotonic_elapsed_ms", self.monotonic_elapsed_ms),
        ):
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise ValueError(f"{label} must be finite")
        if self.uncertainty_ms < 0:
            raise ValueError("uncertainty_ms must be non-negative")
        if type(self.jump_detected) is not bool:
            raise TypeError("jump_detected must be boolean")
        if type(self.moving_origin_ordinal) is not int or self.moving_origin_ordinal < -1:
            raise ValueError("moving_origin_ordinal must be at least -1")

    @property
    def sample_ref(self) -> str:
        return "sha256:" + hashlib.sha256(_canonical_bytes(asdict(self))).hexdigest()


@dataclass(frozen=True, slots=True)
class TemporalV2:
    contract: str
    moving_origin_ordinal: int
    event_time_utc: str | None
    acquired_time_utc: str
    recorded_time_utc: str
    verified_time_utc: str | None
    valid_from_utc: str | None
    valid_until_utc: str | None
    timezone: str
    source: str
    precision_ms: float
    uncertainty_ms: float
    clock_jump_detected: bool

    def __post_init__(self) -> None:
        if self.contract != TEMPORAL_V2_CONTRACT:
            raise ValueError("unsupported temporal-v2 contract")
        if type(self.moving_origin_ordinal) is not int or self.moving_origin_ordinal < 0:
            raise ValueError("moving_origin_ordinal must be non-negative")
        for label, value in (
            ("event_time_utc", self.event_time_utc),
            ("acquired_time_utc", self.acquired_time_utc),
            ("recorded_time_utc", self.recorded_time_utc),
            ("verified_time_utc", self.verified_time_utc),
            ("valid_from_utc", self.valid_from_utc),
            ("valid_until_utc", self.valid_until_utc),
        ):
            if value is not None:
                parse_utc(value, label)
        if self.valid_from_utc and self.valid_until_utc:
            if parse_utc(self.valid_until_utc, "valid_until_utc") < parse_utc(
                self.valid_from_utc, "valid_from_utc"
            ):
                raise ValueError("valid_until_utc cannot precede valid_from_utc")
        if type(self.timezone) is not str or not self.timezone:
            raise ValueError("timezone must be non-empty")
        if type(self.source) is not str or not self.source:
            raise ValueError("source must be non-empty")
        for label, value in (
            ("precision_ms", self.precision_ms),
            ("uncertainty_ms", self.uncertainty_ms),
        ):
            if type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{label} must be finite and non-negative")
        if type(self.clock_jump_detected) is not bool:
            raise TypeError("clock_jump_detected must be boolean")

    @property
    def temporal_ref(self) -> str:
        return "sha256:" + hashlib.sha256(_canonical_bytes(asdict(self))).hexdigest()

    @classmethod
    def from_samples(
        cls,
        *,
        ordinal: int,
        event: TemporalNow | None,
        acquired: TemporalNow,
        recorded: TemporalNow,
        verified: TemporalNow | None,
        valid_from_utc: str | None = None,
        valid_until_utc: str | None = None,
        source: str,
        precision_ms: float = 1.0,
    ) -> "TemporalV2":
        samples = tuple(
            item for item in (event, acquired, recorded, verified) if item is not None
        )
        if not samples:
            raise ValueError("temporal-v2 requires at least one clock sample")
        timezone_names = {item.local_timezone for item in samples}
        if len(timezone_names) != 1:
            raise ValueError("temporal samples use different local timezones")
        return cls(
            contract=TEMPORAL_V2_CONTRACT,
            moving_origin_ordinal=ordinal,
            event_time_utc=None if event is None else event.trusted_utc,
            acquired_time_utc=acquired.trusted_utc,
            recorded_time_utc=recorded.trusted_utc,
            verified_time_utc=None if verified is None else verified.trusted_utc,
            valid_from_utc=valid_from_utc,
            valid_until_utc=valid_until_utc,
            timezone=acquired.local_timezone,
            source=source,
            precision_ms=precision_ms,
            uncertainty_ms=max(item.uncertainty_ms for item in samples),
            clock_jump_detected=any(item.jump_detected for item in samples),
        )


class TrustedClock:
    """Bind UTC wall time to a monotonic anchor and expose jump evidence."""

    def __init__(
        self,
        *,
        timezone_name: str = "America/New_York",
        uncertainty_ms: float = 1.0,
        jump_tolerance_ms: float = 1_000.0,
        wall_clock: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], int] | None = None,
    ) -> None:
        try:
            self._timezone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("timezone_name must identify an installed IANA timezone") from exc
        if type(uncertainty_ms) not in (int, float) or not math.isfinite(float(uncertainty_ms)) or uncertainty_ms < 0:
            raise ValueError("uncertainty_ms must be finite and non-negative")
        if type(jump_tolerance_ms) not in (int, float) or not math.isfinite(float(jump_tolerance_ms)) or jump_tolerance_ms < 0:
            raise ValueError("jump_tolerance_ms must be finite and non-negative")
        self.timezone_name = timezone_name
        self.uncertainty_ms = float(uncertainty_ms)
        self.jump_tolerance_ms = float(jump_tolerance_ms)
        self._wall_clock = wall_clock or (lambda: datetime.now(timezone.utc))
        self._monotonic_clock = monotonic_clock or time.monotonic_ns
        self._anchor_wall = self._wall_clock()
        if self._anchor_wall.tzinfo is None or self._anchor_wall.utcoffset() is None:
            raise ValueError("wall clock must return timezone-aware datetimes")
        self._anchor_wall = self._anchor_wall.astimezone(timezone.utc)
        self._anchor_mono = self._monotonic_clock()
        if type(self._anchor_mono) is not int or self._anchor_mono < 0:
            raise ValueError("monotonic clock must return non-negative integer nanoseconds")
        anchor = {
            "trusted_utc": _utc_text(self._anchor_wall),
            "monotonic_ns": self._anchor_mono,
            "timezone": self.timezone_name,
        }
        self.clock_anchor_ref = "sha256:" + hashlib.sha256(
            _canonical_bytes(anchor)
        ).hexdigest()

    def sample(self, moving_origin_ordinal: int) -> TemporalNow:
        wall = self._wall_clock()
        if wall.tzinfo is None or wall.utcoffset() is None:
            raise ValueError("wall clock must return timezone-aware datetimes")
        wall = wall.astimezone(timezone.utc)
        mono = self._monotonic_clock()
        if type(mono) is not int or mono < self._anchor_mono:
            raise RuntimeError("monotonic clock moved backwards")
        wall_elapsed = (wall - self._anchor_wall).total_seconds() * 1_000.0
        mono_elapsed = (mono - self._anchor_mono) / 1_000_000.0
        jump = abs(wall_elapsed - mono_elapsed) > self.jump_tolerance_ms
        local = wall.astimezone(self._timezone)
        offset = local.utcoffset()
        if offset is None:
            raise RuntimeError("local timezone produced no UTC offset")
        return TemporalNow(
            contract=TEMPORAL_NOW_CONTRACT,
            trusted_utc=_utc_text(wall),
            local_time=local.isoformat(timespec="microseconds"),
            local_timezone=self.timezone_name,
            local_utc_offset_seconds=int(offset.total_seconds()),
            monotonic_ns=mono,
            clock_anchor_ref=self.clock_anchor_ref,
            uncertainty_ms=self.uncertainty_ms,
            jump_detected=jump,
            wall_elapsed_ms=wall_elapsed,
            monotonic_elapsed_ms=mono_elapsed,
            moving_origin_ordinal=moving_origin_ordinal,
        )


def semantic_age(current_moving_origin_ordinal: int, acquired_ordinal: int) -> int:
    """Semantic age is event distance, never scheduler tick count or wall age."""

    if type(current_moving_origin_ordinal) is not int or current_moving_origin_ordinal < -1:
        raise ValueError("current moving-origin ordinal must be at least -1")
    if type(acquired_ordinal) is not int or acquired_ordinal < 0:
        raise ValueError("acquired ordinal must be non-negative")
    if current_moving_origin_ordinal < acquired_ordinal:
        raise ValueError("acquired ordinal cannot be in the future")
    return current_moving_origin_ordinal - acquired_ordinal
