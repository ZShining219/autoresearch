"""Inference-only Jev controller, using Traffic-R1's SUMO observations.

Inheritance reuses the generators, geometry, and signal-to-phase mapping. The
action path below uses native Decisions exclusively; none of the inherited chat
backend, prompt, or text parsing methods are called.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from agent.jev_backend import JEVClient, JEVDecision, JEVError
from agent.jev_prompt import build_jev_request
from agent.llm_tsc import LLMTSCAgent
from agent.llm_tsc_prompt import SIGNAL_ORDER
from common.registry import Registry


@Registry.register_model("jev")
class JEVAgent(LLMTSCAgent):
    def __init__(self, world, rank):
        super().__init__(world, rank)
        self.parallelism = int(self.param.get("parallelism", 16))
        if self.parallelism < 1 or self.n_segments < 1:
            raise ValueError("JEV parallelism and n_segments must be positive")
        self._jev_client: JEVClient | None = None
        self.last_decision: JEVDecision | None = None
        self._decision_log: Path | None = None
        if self.param.get("log_decisions", True):
            root = Registry.mapping["logger_mapping"]["path"].path
            log_dir = Registry.mapping["logger_mapping"]["setting"].param["log_dir"]
            self._decision_log = Path(root) / log_dir / "jev_decisions.jsonl"

    def __repr__(self) -> str:
        return (
            f"JEVAgent(rank={self.rank}, id={self.inter_obj.id}, "
            f"enabled={self._llm_enabled}, phase_map={self._signal_to_phase})"
        )

    def reset(self) -> None:
        super().reset()
        self.last_decision = None

    def _resolve_phase_map(self, param: dict[str, Any]) -> dict[str, int]:
        mapping = super()._resolve_phase_map(param)
        if len(self.inter_obj.phases) == 1:
            return {}
        if (
            set(mapping) != set(SIGNAL_ORDER)
            or len(set(mapping.values())) != 4
            or any(index < 0 or index >= len(self.inter_obj.phases) for index in mapping.values())
        ):
            raise ValueError("JEV needs four distinct, valid green-phase indices for its four signals")
        return mapping

    def _record_decision(self, decision: JEVDecision) -> None:
        if self._decision_log is None:
            return
        record = asdict(decision)
        record.update(
            intersection_id=self.inter_obj.id,
            simulation_time=float(self.world.get_current_time()),
            phase_index=self.last_action,
        )
        self._decision_log.parent.mkdir(parents=True, exist_ok=True)
        with self._decision_log.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, allow_nan=False) + "\n")

    @classmethod
    def get_actions_batch(
        cls,
        agents: Sequence["JEVAgent"],
        obs: Sequence[Any],
        phases: Sequence[Any],
        test: bool = True,
    ) -> list[int]:
        del obs, phases, test
        actions = [0] * len(agents)
        pending = []
        for index, agent in enumerate(agents):
            if not agent._llm_enabled:
                agent.last_action = 0
                agent.last_signal = None
                agent.last_decision = None
                continue
            if agent._jev_client is None:
                agent._jev_client = JEVClient(agent.param)
            # libsumo observation reads stay on the main thread. Only HTTP runs
            # in parallel; each request contains exactly one intersection.
            request = build_jev_request(agent._collect_signal_stats(), agent.n_segments)
            pending.append((index, agent, request))
        if not pending:
            return actions

        def decide(item):
            _index, agent, request = item
            try:
                return agent._jev_client.decide(request)
            except JEVError as exc:
                raise JEVError(f"JEV failed at intersection {agent.inter_obj.id}: {exc}") from None

        with ThreadPoolExecutor(max_workers=min(agents[0].parallelism, len(pending))) as pool:
            decisions = list(pool.map(decide, pending))

        for (index, agent, _request), decision in zip(pending, decisions):
            agent.last_signal = decision.signal
            agent.last_action = agent._signal_to_phase[decision.signal]
            agent.last_decision = decision
            actions[index] = agent.last_action
            agent._record_decision(decision)
        return actions
