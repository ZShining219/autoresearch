"""Generic inference-only LLM controller for LibSignal SUMO."""

from __future__ import annotations

import math
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

from agent.base import BaseAgent
from agent.llm_tsc_backend import LLMBackend, create_backend
from agent.llm_tsc_prompt import (
    APPROACH_WORD,
    LLMParseError,
    SIGNAL_LANE_PAIRS,
    SIGNAL_ORDER,
    build_messages,
    empty_signal_stats,
    parse_signal,
)
from common.registry import Registry
from generator import IntersectionPhaseGenerator, LaneVehicleGenerator

_SHARED_BACKEND: Optional[LLMBackend] = None
_SHARED_BACKEND_LOCK = threading.Lock()


def _get_shared_backend(param: Dict[str, Any]) -> LLMBackend:
    global _SHARED_BACKEND
    if _SHARED_BACKEND is None:
        with _SHARED_BACKEND_LOCK:
            if _SHARED_BACKEND is None:
                _SHARED_BACKEND = create_backend(param)
    return _SHARED_BACKEND


def reset_shared_backend() -> None:
    """Clear the process-wide backend singleton for tests or a new run."""
    global _SHARED_BACKEND
    with _SHARED_BACKEND_LOCK:
        _SHARED_BACKEND = None


def _compass_from_atan2(angle: float) -> str:
    """Map Intersection._get_direction atan2(x,y) angle to N/E/S/W."""
    deg = (angle * 180.0 / math.pi) % 360.0
    if deg < 45.0 or deg >= 315.0:
        return "N"
    if 45.0 <= deg < 135.0:
        return "E"
    if 135.0 <= deg < 225.0:
        return "S"
    return "W"


def _rescale_observation_bins(
    early_queued: int,
    segments: Sequence[int],
    noisy_total: int,
) -> Tuple[int, List[int]]:
    """Reconcile Traffic-R1's spatial bins with a noisy lane total.

    ``lane_count`` is the configured detector reading, while Traffic-R1's prompt
    needs a queue count plus spatial segment counts.  The vehicle records provide
    the spatial proportions; the detector total supplies the noisy magnitude.
    Largest-remainder apportionment keeps the result integral, deterministic, and
    exactly equal to the detector total.

    A positive detector reading with no visible vehicle records has no recoverable
    spatial location.  We represent that false-positive count in the farthest
    segment rather than fabricating an urgent queue.
    """
    bins = [max(0, int(early_queued))]
    bins.extend(max(0, int(value)) for value in segments)
    target = max(0, int(noisy_total))
    raw_total = sum(bins)

    if raw_total == target:
        return bins[0], bins[1:]

    if raw_total == 0:
        if len(bins) == 1:
            return target, []
        # No spatial evidence exists for a positive false-positive reading.
        # Segment 3 (the farthest segment) is the least urgent valid bucket.
        return 0, [0] * (len(bins) - 2) + [target]

    scaled = [value * target / raw_total for value in bins]
    apportioned = [math.floor(value) for value in scaled]
    remainder = target - sum(apportioned)
    if remainder:
        # Stable index tie-break makes equal fractional remainders reproducible.
        order = sorted(
            range(len(bins)),
            key=lambda index: (-(scaled[index] - apportioned[index]), index),
        )
        for index in order[:remainder]:
            apportioned[index] += 1
    return apportioned[0], apportioned[1:]


@Registry.register_model("traffic_r1")
@Registry.register_model("deepseek")
@Registry.register_model("deepseek_r1_8b_2048")
@Registry.register_model("qwen25_7b")
@Registry.register_model("qwen3_4b_no_think")
@Registry.register_model("qwen3_4b_think1024")
@Registry.register_model("qwen3_4b_think1024_sampled")
@Registry.register_model("qwen3_4b_think2048")
@Registry.register_model("qwen36_27b_no_think")
class LLMTSCAgent(BaseAgent):
    """LLM traffic controller used by both public backend configurations."""

    def __init__(self, world, rank):
        super().__init__(world)
        self.world = world
        self.rank = rank
        self.model = None

        param = Registry.mapping["model_mapping"]["setting"].param
        self.param = param
        self.v_stop = float(param.get("v_stop", 0.1))
        self.n_segments = int(param.get("n_segments", 3))
        self.parse_retries = int(param.get("parse_retries", 5))
        self.obs_distance = float(
            param.get("obs_distance", getattr(world, "max_distance", 200.0))
        )

        inter_id = self.world.intersection_ids[self.rank]
        self.inter_obj = self.world.id2intersection[inter_id]
        self._init_generators()

        self.action_space = gym.spaces.Discrete(len(self.inter_obj.phases))
        self._lane_meta = self._build_lane_meta()
        self._signal_to_phase = self._resolve_phase_map(param)
        self._llm_enabled = len(self.inter_obj.phases) >= 4 and bool(
            self._signal_to_phase
        )

        # Lazy: only intersections that need the LLM touch the backend.
        self._backend: Optional[LLMBackend] = None
        self.last_action = 0
        self.last_signal: Optional[str] = None

    def _init_generators(self) -> None:
        self.ob_generator = LaneVehicleGenerator(
            self.world, self.inter_obj, ["lane_count"], in_only=True, average=None
        )
        self.phase_generator = IntersectionPhaseGenerator(
            self.world, self.inter_obj, ["phase"], targets=["cur_phase"], negative=False
        )
        self.reward_generator = LaneVehicleGenerator(
            self.world,
            self.inter_obj,
            ["lane_count"],
            in_only=True,
            average="all",
            negative=True,
        )
        self.queue = LaneVehicleGenerator(
            self.world,
            self.inter_obj,
            ["lane_waiting_count"],
            in_only=True,
            negative=False,
        )
        self.delay = LaneVehicleGenerator(
            self.world, self.inter_obj, ["lane_delay"], in_only=True, negative=False
        )

    def __repr__(self) -> str:
        backend = self.param.get("backend", "api")
        return (
            f"LLMTSCAgent(rank={self.rank}, id={self.inter_obj.id}, "
            f"backend={backend}, llm={self._llm_enabled}, "
            f"phase_map={self._signal_to_phase})"
        )

    def reset(self) -> None:
        inter_id = self.world.intersection_ids[self.rank]
        self.inter_obj = self.world.id2intersection[inter_id]
        self._init_generators()
        self._lane_meta = self._build_lane_meta()
        self._signal_to_phase = self._resolve_phase_map(self.param)
        self._llm_enabled = len(self.inter_obj.phases) >= 4 and bool(
            self._signal_to_phase
        )
        self.last_action = 0
        self.last_signal = None

    # ---- observation / reward / phase (LibSignal interface) ----

    def get_ob(self):
        x_obs = [self.ob_generator.generate()]
        return np.array(x_obs, dtype=np.float32)

    def get_reward(self):
        rewards = [self.reward_generator.generate()]
        return np.squeeze(np.array(rewards)) * 12

    def get_phase(self):
        phase = [self.phase_generator.generate()]
        return (np.concatenate(phase)).astype(np.int8)

    def get_queue(self):
        queue = [self.queue.generate()]
        return np.sum(np.squeeze(np.array(queue)))

    def get_delay(self):
        delay = [self.delay.generate()]
        return np.sum(np.squeeze(np.array(delay)))

    # ---- action ----

    def _ensure_backend(self) -> LLMBackend:
        if self._backend is None:
            self._backend = _get_shared_backend(self.param)
        return self._backend

    def _build_messages(self) -> List[Dict[str, str]]:
        return build_messages(
            self._collect_signal_stats(),
            n_segments=self.n_segments,
        )

    @classmethod
    def get_actions_batch(
        cls,
        agents: Sequence["LLMTSCAgent"],
        obs: Sequence[Any],
        phases: Sequence[Any],
        test: bool = True,
    ) -> List[int]:
        """Collect actions, batching LLM requests through the backend."""
        del obs, phases, test
        if not agents:
            return []

        actions = [0] * len(agents)
        pending: List[
            Tuple[int, "LLMTSCAgent", List[Dict[str, str]]]
        ] = []
        last_errors: Dict[int, Exception] = {}

        for index, agent in enumerate(agents):
            # Fringe / single-phase TLS: no LLM and deterministic action 0.
            if not agent._llm_enabled:
                agent.last_action = 0
                agent.last_signal = None
                continue
            agent._ensure_backend()
            pending.append((index, agent, agent._build_messages()))

        backend = pending[0][1]._backend if pending else None
        configured_attempts = max(1, agents[0].parse_retries)
        max_attempts = (
            backend.parse_attempts(configured_attempts)
            if backend is not None
            else configured_attempts
        )

        for attempt in range(1, max_attempts + 1):
            if not pending:
                break

            backend = pending[0][1]._backend
            if backend is None:
                raise RuntimeError("LLM backend was not initialized for a pending action")
            raw_responses = backend.complete_many([item[2] for item in pending])
            if len(raw_responses) != len(pending):
                raise RuntimeError(
                    "LLM backend returned a different number of responses than requests"
                )

            next_pending = []
            for item, raw in zip(pending, raw_responses):
                index, agent, messages = item
                try:
                    signal = parse_signal(raw, allowed=SIGNAL_ORDER)
                    if signal not in agent._signal_to_phase:
                        raise LLMParseError(
                            f"Signal {signal} has no mapped green index on "
                            f"{agent.inter_obj.id}"
                        )
                except LLMParseError as exc:
                    last_errors[index] = exc
                    retry_messages = backend.retry_messages(messages, raw)
                    next_pending.append((index, agent, retry_messages))
                    continue

                action = int(agent._signal_to_phase[signal])
                agent.last_action = action
                agent.last_signal = signal
                actions[index] = action

            pending = next_pending
            if pending and attempt < max_attempts:
                print(
                    f"[LLM] parse retry attempt={attempt + 1}/{max_attempts} "
                    f"pending={len(pending)}",
                    flush=True,
                )

        if pending:
            failures = "; ".join(
                f"{agent.inter_obj.id}: {last_errors[index]}"
                for index, agent, _messages in pending
            )
            raise RuntimeError(
                f"LLM hard failure after {max_attempts} "
                f"attempts ({failures})"
            )
        return actions

    def get_action(self, ob, phase, test=True):
        return self.get_actions_batch([self], [ob], [phase], test=test)[0]

    # ---- lane / phase geometry ----

    def _build_lane_meta(self) -> Dict[str, Tuple[str, str]]:
        """Map incoming lane id -> (approach_from, movement in R/T/L)."""
        meta: Dict[str, Tuple[str, str]] = {}
        inter = self.inter_obj
        for road, direction, is_out in zip(inter.roads, inter.directions, inter.outs):
            if is_out:
                continue
            approach = _compass_from_atan2(direction)
            lanes = sorted(
                inter.road_lane_mapping[road],
                key=lambda x: int(str(x).rsplit("_", 1)[-1]),
            )
            if len(lanes) >= 3:
                labels = ["R", "T", "L"] + ["L"] * (len(lanes) - 3)
            elif len(lanes) == 2:
                labels = ["T", "L"]
            elif len(lanes) == 1:
                labels = ["T"]
            else:
                continue
            for lane, label in zip(lanes, labels):
                meta[lane] = (approach, label)
        return meta

    def _resolve_phase_map(self, param: Dict[str, Any]) -> Dict[str, int]:
        """Map the four signal names to LibSignal green-phase indices."""
        explicit = param.get("phase_name_to_index")
        if explicit:
            return {str(key).upper(): int(value) for key, value in explicit.items()}

        n = len(self.inter_obj.phases)
        if n == 1:
            return {}
        if n == 4:
            auto = self._autodetect_phase_map()
            if auto:
                return auto
            return {name: index for index, name in enumerate(SIGNAL_ORDER)}

        auto = self._autodetect_phase_map()
        if auto:
            return auto

        # Documented Grid4x4 / NEMA ordering fallback used in this repo.
        fallback = {"NTST": 0, "NLSL": 1, "ETWT": 4, "ELWL": 5}
        if n >= 6 and all(index < n for index in fallback.values()):
            print(
                f"[LLM] Warning: using NEMA fallback phase map on "
                f"{self.inter_obj.id}: {fallback}"
            )
            return fallback
        raise RuntimeError(
            f"LLM controller could not map four signal names onto {n} green phases "
            f"at intersection {self.inter_obj.id}"
        )

    def _autodetect_phase_map(self) -> Dict[str, int]:
        """Find green indices whose non-right start lanes match each signal pair."""
        if not self._lane_meta:
            return {}
        phase_sets: List[frozenset] = []
        for phase_id in range(len(self.inter_obj.phases)):
            keys = []
            for lane in self.inter_obj.phase_available_startlanes[phase_id]:
                if lane not in self._lane_meta:
                    continue
                approach, movement = self._lane_meta[lane]
                if movement == "R":
                    continue
                keys.append((approach, movement))
            phase_sets.append(frozenset(keys))

        mapping: Dict[str, int] = {}
        for signal, pair in SIGNAL_LANE_PAIRS.items():
            target = frozenset(pair)
            matches = [index for index, phase_set in enumerate(phase_sets) if phase_set == target]
            if matches:
                mapping[signal] = matches[0]
        return mapping if len(mapping) == 4 else {}

    def _collect_signal_stats(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        stats = empty_signal_stats(n_segments=self.n_segments)
        full_observation = self.inter_obj.full_observation
        if not full_observation:
            return stats

        lane_counts: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for lane, (approach, movement) in self._lane_meta.items():
            if movement == "R":
                continue
            lane_observation = full_observation.get(lane)
            if not lane_observation:
                continue
            length = float(self.world.eng.lane.getLength(lane))
            early = 0
            segments = [0] * self.n_segments
            for vehicle in lane_observation.get("vehicles", []):
                speed = float(vehicle.get("speed", 0.0))
                position = float(vehicle.get("position", 0.0))
                distance_to_stop = max(0.0, length - position)
                if distance_to_stop > self.obs_distance:
                    continue
                if speed < self.v_stop:
                    early += 1
                    continue
                if length <= 0:
                    continue
                relative_position = min(1.0, distance_to_stop / length)
                segment = min(
                    self.n_segments - 1,
                    int(relative_position * self.n_segments),
                )
                segments[segment] += 1
            noisy_total = lane_observation.get("lane_count")
            if noisy_total is None:
                noisy_total = early + sum(segments)
            early, segments = _rescale_observation_bins(
                early,
                segments,
                noisy_total,
            )
            lane_counts[(approach, movement)] = {
                "early_queued": early,
                "segments": segments,
            }

        for signal, pair in SIGNAL_LANE_PAIRS.items():
            for approach, movement in pair:
                side = APPROACH_WORD[approach]
                data = lane_counts.get(
                    (approach, movement),
                    {"early_queued": 0, "segments": [0] * self.n_segments},
                )
                stats[signal][side] = {
                    "early_queued": int(data["early_queued"]),
                    "segments": [int(value) for value in data["segments"]],
                }
        return stats
