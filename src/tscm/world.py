"""SUMO world wrapper for signed max-pressure experiments.

Semantics ported from LibSignal `world/world_sumo.py` (DaRL-LibSignal):
  - legal green phases are read from each TLS program in the net file
    (LibSignal `generate_valid_phase`: non-yellow, non-all-red states)
  - phase->laneLinks come from controlled links whose RYG state is 'G' or 's'
  - automatic yellow phases inserted between phase pairs (`create_yellows`)
  - `pseudo_step` called once per simulated second with the held action
    (LibSignal: env.step inside the action_interval loop); current_phase_time
    is counted in seconds.
  - lane_count = vehicles on the lane (getLastStepVehicleNumber)

Documented difference vs LibSignal: lane_count uses the whole-lane vehicle
count instead of the <=200 m-to-next-TLS filter (closer to CityFlow lane
counts used by LMP-GNN; identical across all compared methods).
"""
import os
import numpy as np

try:
    import libsumo as traci
    _LIBSUMO = True
except ImportError:
    import traci
    _LIBSUMO = False

import sumolib


def _sumo_bin(name):
    import shutil
    p = shutil.which(name)
    if p:
        return p
    import sumo as _pkg
    return os.path.join(os.path.dirname(_pkg.__file__), "bin", name)


class Intersection:
    """One signalised intersection: phases, lanelinks, yellow transitions."""

    def __init__(self, tls_id, eng, green_states, yellow_len):
        self.id = tls_id
        self.eng = eng
        self.current_phase = 0
        self.next_phase = 0
        self.virtual_phase = 0     # last commanded green phase
        self.current_phase_time = 0
        self.yellow_phase_time = yellow_len

        links = eng.trafficlight.getControlledLinks(tls_id)
        self.lanelinks_all = links           # per signal index: [[from,to,via],...]
        self.green_states = green_states     # list[str]: RYG states (green only)
        self.n_phases = len(green_states)
        self.phase_available_lanelinks = []
        lanes = set()
        for st in green_states:
            ll = []
            for n, s in enumerate(st):
                if s in ("G", "s") and n < len(links) and links[n]:
                    fr, to = links[n][0][0], links[n][0][1]
                    ll.append([fr, to])
                    lanes.add(fr); lanes.add(to)
            self.phase_available_lanelinks.append(ll)
        self.lanes = sorted(lanes)
        # legal phases = those releasing at least one lanelink (LMP-GNN Eq.2)
        self.legal_phases = [i for i, ll in enumerate(self.phase_available_lanelinks) if ll]

        self.full_states, self.yellow_dict = self._create_yellows(green_states)
        self._install_program()

    def _create_yellows(self, states):
        """LibSignal create_yellows: between phase i->j, signals going G/g->r/s turn 'y'."""
        new_states = list(states)
        yellow_dict = {}
        for i in range(len(states)):
            for j in range(len(states)):
                if i == j:
                    continue
                need, y = False, ""
                for k in range(len(states[i])):
                    if states[i][k] in ("G", "g") and states[j][k] in ("r", "s"):
                        need = True
                        y += "y"
                    else:
                        y += states[i][k]
                if need:
                    yellow_dict[f"{i}_{j}"] = len(new_states)
                    new_states.append(y)
        return new_states, yellow_dict

    def _install_program(self):
        eng = self.eng
        phases = []
        for i, st in enumerate(self.full_states):
            dur = self.yellow_phase_time if i >= self.n_phases else 3600
            phases.append(eng.trafficlight.Phase(float(dur), st))
        logic = eng.trafficlight.Logic(self.id + "_rl", 0, 0, phases)
        eng.trafficlight.setProgramLogic(self.id, logic)
        eng.trafficlight.setProgram(self.id, self.id + "_rl")
        eng.trafficlight.setPhase(self.id, 0)
        self.current_phase = 0
        self.current_phase_time = 0

    def get_current_phase(self):
        return self.eng.trafficlight.getPhase(self.id)

    def pseudo_step(self, action):
        """LibSignal pseudo_step (called once per simulated second)."""
        self.virtual_phase = action
        if self.current_phase_time >= self.yellow_phase_time:
            self._set_phase(action)
        else:
            if action != self.get_current_phase() and self.current_phase_time > self.yellow_phase_time:
                self.current_phase_time = 0
            if self.current_phase_time == 0:
                self._prep(action)
            elif self.current_phase_time < self.yellow_phase_time:
                pass  # dwell inside yellow until it elapses
            else:
                self._set_phase(action)
        self.current_phase_time += 1

    def _prep(self, new_phase):
        if self.get_current_phase() == new_phase:
            self.next_phase = self.get_current_phase()
            self._set_phase(self.next_phase)
        else:
            self.next_phase = new_phase
            yk = f"{self.get_current_phase()}_{new_phase}"
            if yk in self.yellow_dict:
                self.eng.trafficlight.setPhase(self.id, int(self.yellow_dict[yk]))

    def _set_phase(self, p):
        self.eng.trafficlight.setPhase(self.id, int(p))


class SumoWorld:
    """Closed-loop SUMO world: obs -> held actions -> 1 s steps -> metrics."""

    def __init__(self, net_path, rou_path, yellow_len=3, seed=0, extra_sumo_args=None):
        self.net_path, self.rou_path = net_path, rou_path
        self.yellow_len = yellow_len
        self.seed = seed
        self.extra = extra_sumo_args or []
        self._net = sumolib.net.readNet(net_path)
        self._green = self._read_green_phases()
        self.eng = None
        self.intersections = []
        self.lane_index = []                 # ordered observable lanes
        self._lane_pos = {}
        self.arrived = {}                    # veh_id -> travel_time
        self.inside = {}                     # veh_id -> depart time
        self.queue_hist = []
        self.time = 0.0

    # ---------- construction ----------
    def _read_green_phases(self):
        import xml.etree.ElementTree as ET
        out = {}
        for e in ET.parse(self.net_path).getroot().iter("tlLogic"):
            greens = []
            for ph in e.findall("phase"):
                st = ph.get("state")
                if "y" not in st and st.count("r") + st.count("s") != len(st):
                    greens.append(st)
            out[e.get("id")] = greens
        return out

    def start(self):
        args = [_sumo_bin("sumo"), "-n", self.net_path, "-r", self.rou_path,
                "--no-warnings", "--no-step-log", "--seed", str(self.seed)] + self.extra
        traci.start(args)
        self.eng = traci
        self.intersections = [
            Intersection(tls_id, self.eng, greens, self.yellow_len)
            for tls_id, greens in self._green.items()]
        lanes = set()
        for it in self.intersections:
            lanes.update(it.lanes)
        self.lane_index = sorted(lanes)
        self._lane_pos = {l: k for k, l in enumerate(self.lane_index)}
        self.time = 0.0
        self.arrived.clear(); self.inside.clear()
        self.queue_hist = []
        return self

    # ---------- runtime ----------
    def lane_counts(self):
        q = np.zeros(len(self.lane_index), dtype=np.float32)
        L = self.eng.lane
        for l, k in self._lane_pos.items():
            q[k] = L.getLastStepVehicleNumber(l)
        return q

    def lane_queue_lengths(self):
        q = np.zeros(len(self.lane_index), dtype=np.float32)
        L = self.eng.lane
        for l, k in self._lane_pos.items():
            q[k] = L.getLastStepHaltingNumber(l)
        return q

    def step_seconds(self, actions, n=1):
        """Advance n simulated seconds; pseudo_step each second (LibSignal cadence)."""
        for _ in range(n):
            for it in self.intersections:
                it.pseudo_step(actions[it.id])
            self.eng.simulationStep()
            self.time += 1.0
            for v in self.eng.simulation.getDepartedIDList():
                self.inside[v] = self.time
            for v in self.eng.simulation.getArrivedIDList():
                if v in self.inside:
                    self.arrived[v] = self.time - self.inside[v]
            self.queue_hist.append(float(self.lane_queue_lengths().sum()))

    # ---------- metrics ----------
    def metrics(self):
        arr = np.array(list(self.arrived.values())) if self.arrived else np.array([0.0])
        return {
            "avg_travel_time": float(arr.mean()),
            "throughput": int(len(self.arrived)),
            "unfinished": int(len(self.inside) - len(self.arrived)),
            "avg_queue": float(np.mean(self.queue_hist)) if self.queue_hist else 0.0,
            "max_queue": float(np.max(self.queue_hist)) if self.queue_hist else 0.0,
        }

    def close(self):
        try:
            self.eng.close()
        except Exception:
            pass
