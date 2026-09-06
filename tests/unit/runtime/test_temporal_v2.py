from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from angler.runtime.temporal_v2 import TemporalV2, TrustedClock, semantic_age


class _ClockSource:
    def __init__(self) -> None:
        self.wall = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
        self.mono = 10_000_000_000

    def wall_now(self):
        return self.wall

    def mono_now(self):
        return self.mono

    def advance(self, *, wall_ms: int, mono_ms: int) -> None:
        self.wall += timedelta(milliseconds=wall_ms)
        self.mono += mono_ms * 1_000_000


class TemporalV2Tests(unittest.TestCase):
    def test_trusted_clock_exposes_local_offset_monotonic_anchor_and_jump(self) -> None:
        source = _ClockSource()
        clock = TrustedClock(
            timezone_name="America/New_York",
            uncertainty_ms=2.5,
            jump_tolerance_ms=100,
            wall_clock=source.wall_now,
            monotonic_clock=source.mono_now,
        )
        source.advance(wall_ms=1000, mono_ms=1000)
        ordinary = clock.sample(4)
        self.assertFalse(ordinary.jump_detected)
        self.assertEqual(ordinary.local_utc_offset_seconds, -4 * 3600)
        self.assertEqual(ordinary.moving_origin_ordinal, 4)
        source.advance(wall_ms=5000, mono_ms=1000)
        jumped = clock.sample(4)
        self.assertTrue(jumped.jump_detected)
        self.assertEqual(jumped.clock_anchor_ref, ordinary.clock_anchor_ref)

    def test_temporal_v2_keeps_six_time_roles_separate(self) -> None:
        source = _ClockSource()
        clock = TrustedClock(
            wall_clock=source.wall_now, monotonic_clock=source.mono_now
        )
        event = clock.sample(-1)
        source.advance(wall_ms=10, mono_ms=10)
        acquired = clock.sample(-1)
        source.advance(wall_ms=10, mono_ms=10)
        recorded = clock.sample(-1)
        temporal = TemporalV2.from_samples(
            ordinal=0,
            event=event,
            acquired=acquired,
            recorded=recorded,
            verified=acquired,
            valid_from_utc="2026-09-01T00:00:00Z",
            valid_until_utc="2026-09-03T00:00:00Z",
            source="synthetic.test",
        )
        self.assertNotEqual(temporal.event_time_utc, temporal.acquired_time_utc)
        self.assertNotEqual(temporal.acquired_time_utc, temporal.recorded_time_utc)
        self.assertEqual(temporal.verified_time_utc, temporal.acquired_time_utc)
        self.assertTrue(temporal.temporal_ref.startswith("sha256:"))

    def test_semantic_age_is_ordinal_distance_only(self) -> None:
        self.assertEqual(semantic_age(9, 3), 6)
        with self.assertRaises(ValueError):
            semantic_age(2, 3)


if __name__ == "__main__":
    unittest.main()
