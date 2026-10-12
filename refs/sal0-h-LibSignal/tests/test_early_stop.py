"""Unit tests for paired-cycle early-stop (no SUMO)."""

import math
import unittest

from common.early_stop import (
    CycleMeanStopper,
    PairedCycleStopper,
    cycle_mean_decision,
    paired_cycle_decision,
)


class PairedCycleDecisionTest(unittest.TestCase):
    def test_clear_same_file_drop_is_improvement(self):
        prev = [200.0] * 10
        curr = [180.0] * 10  # 10% drop, zero spread
        d = paired_cycle_decision(prev, curr, min_delta=0.003, z=1.0)
        self.assertTrue(d["improved"])
        self.assertAlmostEqual(d["mean_delta"], 20.0)
        self.assertEqual(d["se"], 0.0)
        self.assertAlmostEqual(d["threshold"], 0.003 * 180.0)

    def test_tiny_drop_below_relative_floor_is_not_improvement(self):
        # 0.2% drop, zero noise: statistically "sure" but smaller than 0.3% floor.
        prev = [200.0] * 10
        curr = [199.6] * 10
        d = paired_cycle_decision(prev, curr, min_delta=0.003, z=1.0)
        self.assertAlmostEqual(d["mean_delta"], 0.4)
        self.assertGreater(d["threshold"], d["mean_delta"])
        self.assertFalse(d["improved"])

    def test_noisy_mean_drop_below_se_is_not_improvement(self):
        # Mean drop ~1s but one file swings a lot → SE > mean_delta.
        prev = [200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
        curr = [170, 205, 205, 205, 205, 205, 205, 205, 205, 205]
        d = paired_cycle_decision(prev, curr, min_delta=0.0, z=1.0)
        self.assertGreater(d["se"], abs(d["mean_delta"]))
        self.assertFalse(d["improved"])

    def test_rotation_without_learning_stops_looking_like_progress(self):
        # Hard/easy files (200 vs 400) repeating: paired deltas are ~0.
        easy_hard = [200.0, 400.0] * 5
        d = paired_cycle_decision(easy_hard, list(easy_hard), min_delta=0.003, z=1.0)
        self.assertAlmostEqual(d["mean_delta"], 0.0)
        self.assertFalse(d["improved"])


class PairedCycleStopperTest(unittest.TestCase):
    def test_stops_after_patience_flat_cycles(self):
        # 10-file rotation, constant ATT after two improving cycles.
        stopper = PairedCycleStopper(
            cycle_len=10, patience=2, min_delta=0.003, z=1.0, min_episodes=30
        )
        # Cycle 0: 220, cycle 1: 180 (improve), then 180 forever.
        series = [220.0] * 10 + [180.0] * 10 + [180.0] * 30
        stopped_at = None
        last_decision = None
        for i, att in enumerate(series):
            stop, decision = stopper.add(att)
            if decision is not None:
                last_decision = decision
            if stop:
                stopped_at = i
                break
        # First comparison at episode 19 (cycle 1 vs 0): improve.
        # Next two comparisons (ep 29, 39) are flat → stop at 39 (40 episodes).
        self.assertEqual(stopped_at, 39)
        self.assertFalse(last_decision["improved"])
        self.assertEqual(last_decision["cycles_without_improve"], 2)

    def test_keeps_going_while_cycles_improve(self):
        stopper = PairedCycleStopper(
            cycle_len=10, patience=2, min_delta=0.003, z=1.0, min_episodes=30
        )
        # Each cycle 5% faster than the previous, same value within a cycle.
        stopped = False
        for c in range(8):
            att = 200.0 * (0.95 ** c)
            for _ in range(10):
                stop, _ = stopper.add(att)
                if stop:
                    stopped = True
        self.assertFalse(stopped)
        self.assertEqual(stopper.cycles_without_improve, 0)

    def test_no_decision_until_two_full_cycles(self):
        stopper = PairedCycleStopper(cycle_len=10, patience=2, min_episodes=0)
        for i in range(19):
            stop, decision = stopper.add(200.0)
            self.assertFalse(stop)
            self.assertIsNone(decision)
        stop, decision = stopper.add(200.0)
        self.assertIsNotNone(decision)
        # Flat, patience=1, not yet 2.
        self.assertFalse(stop)

    def test_non_finite_ignored(self):
        stopper = PairedCycleStopper(cycle_len=2, patience=1, min_episodes=0)
        stop, decision = stopper.add(float("nan"))
        self.assertFalse(stop)
        self.assertIsNone(decision)
        self.assertEqual(stopper.atts, [])


class CycleMeanDecisionTest(unittest.TestCase):
    def test_large_drop_is_improvement(self):
        prev = [40.0] * 10
        curr = [30.0] * 10  # 25% and 10s
        d = cycle_mean_decision(prev, curr, min_delta=0.05, abs_floor=1.0)
        self.assertTrue(d["improved"])
        self.assertFalse(d["flat"])

    def test_four_percent_drop_is_flat(self):
        prev = [40.0] * 10
        curr = [38.5] * 10  # 3.75% and 1.5s — rel < 5%
        d = cycle_mean_decision(prev, curr, min_delta=0.05, abs_floor=1.0)
        self.assertTrue(d["flat"])
        self.assertFalse(d["improved"])

    def test_sub_second_drop_is_flat_even_if_relative_large(self):
        prev = [8.0] * 10
        curr = [7.2] * 10  # 10% but only 0.8s
        d = cycle_mean_decision(prev, curr, min_delta=0.05, abs_floor=1.0)
        self.assertTrue(d["flat"])
        self.assertFalse(d["improved"])

    def test_rise_is_not_improvement(self):
        prev = [30.0] * 10
        curr = [40.0] * 10
        d = cycle_mean_decision(prev, curr, min_delta=0.05, abs_floor=1.0)
        self.assertFalse(d["improved"])


class CycleMeanStopperTest(unittest.TestCase):
    def test_stops_after_three_flat_cycles(self):
        stopper = CycleMeanStopper(
            cycle_len=10, patience=3, min_delta=0.05, abs_floor=1.0, min_episodes=30
        )
        # Cycle 0: 50, cycle 1: 30 (improve), then 30 forever.
        series = [50.0] * 10 + [30.0] * 10 + [30.0] * 40
        stopped_at = None
        last = None
        for i, val in enumerate(series):
            stop, decision = stopper.add(val)
            if decision is not None:
                last = decision
            if stop:
                stopped_at = i
                break
        # Compare at 19 (improve), 29/39/49 (flat x3) → stop at 49.
        self.assertEqual(stopped_at, 49)
        self.assertEqual(last["cycles_without_improve"], 3)
        self.assertFalse(last["improved"])

    def test_keeps_going_while_wait_drops(self):
        stopper = CycleMeanStopper(
            cycle_len=10, patience=3, min_delta=0.05, abs_floor=1.0, min_episodes=30
        )
        stopped = False
        for c in range(8):
            wait = 40.0 * (0.85 ** c)  # ~15% drop per cycle
            for _ in range(10):
                stop, _ = stopper.add(wait)
                if stop:
                    stopped = True
        self.assertFalse(stopped)
        self.assertEqual(stopper.cycles_without_improve, 0)


if __name__ == "__main__":
    unittest.main()
