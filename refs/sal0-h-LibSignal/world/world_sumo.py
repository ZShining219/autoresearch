"""
Part of this code is borrowed from RESCO: https://github.com/Pi-Star-Lab/RESCO
"""

import os
import sys
from math import atan2, pi
import xml.etree.cElementTree as ET

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit('No SUMO in environment path')
from common.registry import Registry

import json
import logging
import re
import copy
import hashlib
import random

import sumolib
try:
    import libsumo
except ImportError:
    libsumo = None
import traci

# Hub-OD routes are 100% type="pkw". hetero=true loads car/truck vTypes, then
# demand_train_file replaces the hetero route, so trucks never spawn unless we
# rewrite. Same 80/20 stride as extras/gen_hetero_routes.py.
HETERO_TRUCK_RATIO = 0.2


def _xml_local_tag(el):
    return el.tag.split('}')[-1]


def rewrite_route_file_hetero_mix(route_abs, truck_ratio=HETERO_TRUCK_RATIO):
    """Rewrite a hub-OD .rou.xml so additional car/truck vTypes are used.

    Drops inline <vType id="pkw"> and assigns type="car" / type="truck".
    Files that already have trucks (M0 hetero routes) are left unchanged.
    Cached under <dir>/.hetero_mix/.
    """
    if not route_abs or not os.path.isfile(route_abs):
        return route_abs
    try:
        tree = ET.parse(route_abs)
    except ET.ParseError:
        return route_abs
    root = tree.getroot()
    vehicles = [el for el in root.iter() if _xml_local_tag(el) == 'vehicle']
    if not vehicles:
        return route_abs
    types = [veh.get('type', '') for veh in vehicles]
    n_pkw = sum(1 for t in types if t == 'pkw' or t == '')
    n_truck = sum(1 for t in types if t == 'truck')
    if n_pkw == 0 and n_truck > 0:
        return route_abs
    cache_dir = os.path.join(os.path.dirname(route_abs), '.hetero_mix')
    os.makedirs(cache_dir, exist_ok=True)
    dst = os.path.join(cache_dir, os.path.basename(route_abs))
    if os.path.isfile(dst) and os.path.getmtime(dst) >= os.path.getmtime(route_abs):
        return dst
    for el in [child for child in list(root) if _xml_local_tag(child) == 'vType']:
        root.remove(el)
    n = len(vehicles)
    n_trucks = int(round(n * float(truck_ratio)))
    if n_trucks > 0:
        stride = n / n_trucks
        truck_indices = {int(i * stride) for i in range(n_trucks)}
    else:
        truck_indices = set()
    for idx, veh in enumerate(vehicles):
        veh.set('type', 'truck' if idx in truck_indices else 'car')
    tree.write(dst, encoding='utf-8', xml_declaration=True)
    n_cars = n - len(truck_indices)
    print(
        f"[Hetero] mixed fleet {n_cars} car / {len(truck_indices)} truck -> {dst}"
    )
    return dst


class Intersection(object):
    '''
    Intersection Class is mainly used for describing crossing information and defining acting methods.
    '''
    def __init__(self, id, world, phases):
        self.id = id
        self.world = world
        self.eng = self.world.eng
        self.lanes = []
        self.roads = []
        self.outs = []
        self.directions = []
        self.out_roads = []
        self.in_roads = []
        self.road_lane_mapping = {}
        self.interface_flag = world.interface_flag

        # map_name = Registry.mapping['world_mapping']['setting'].param['network']
        # self.lane_order_cf = None
        # self.lane_order_sumo = None
        # if 'signal_config' in Registry.mapping['world_mapping']['setting'].param.keys():
        #     if 'N' in Registry.mapping['world_mapping']['setting'].param['signal_config'][map_name]['cf_order'].keys():
        #         self.lane_order_cf = Registry.mapping['world_mapping']['setting'].param['signal_config'][map_name]['cf_order']
        #         self.lane_order_sumo = Registry.mapping['world_mapping']['setting'].param['signal_config'][map_name]['sumo_order']
        #     else:
        #         if self.id in Registry.mapping['world_mapping']['setting'].param['signal_config'][map_name]['cf_order'].keys():
        #             self.lane_order_cf = Registry.mapping['world_mapping']['setting'].param['signal_config'][map_name]['cf_order'][self.id]
        #             self.lane_order_sumo = Registry.mapping['world_mapping']['setting'].param['signal_config'][map_name]['sumo_order'][self.id]
        #         else:
        #             self.lane_order_cf = Registry.mapping['world_mapping']['setting'].param['signal_config'][map_name]['cf_order'][self.id[3:]] # exclude 'GS_'
        #             self.lane_order_sumo = Registry.mapping['world_mapping']['setting'].param['signal_config'][map_name]['sumo_order'][self.id[3:]]

        # links and phase information of each intersection
        self.current_phase = 0
        self.virtual_phase = 0  # see yellow phase as the same phase after changing
        self.next_phase = 0
        self.current_phase_time = 0

        self.yellow_phase_time = min([i.duration for i in self.eng.trafficlight.getAllProgramLogics(self.id)[0].phases])
        self.map_name = world.map  # TODO: try to add it to Registry later

        self.lanelinks = world.eng.trafficlight.getControlledLinks(self.id)
        for link in self.lanelinks:
            # skip if empty link
            if not link:
                continue
            link = link[0]
            if link[0][:-2] not in self.road_lane_mapping.keys():
                self.road_lane_mapping.update({link[0][:-2]: []})  # assume less than 9 lanes in each road
                self.road_lane_mapping[link[0][:-2]].append(link[0])
                self.roads.append(link[0][:-2])
                self.outs.append(False)
                road = self.eng.lane.getShape(link[0])
                self.directions.append(self._get_direction(road, False))
            elif link[0][:-2] in self.road_lane_mapping.keys() and link[0] not in self.road_lane_mapping[link[0][:-2]]:
                self.road_lane_mapping[link[0][:-2]].append(link[0])
            if link[1][:-2] not in self.road_lane_mapping.keys():
                self.road_lane_mapping.update({link[1][:-2]: []})  # assume less than 9 lanes in each road
                self.road_lane_mapping[link[1][:-2]].append(link[1])
                self.roads.append(link[1][:-2])
                self.outs.append(True)
                road = self.eng.lane.getShape(link[1])
                self.directions.append(self._get_direction(road, True))
            elif link[1][:-2] in self.road_lane_mapping.keys() and link[1] not in self.road_lane_mapping[link[1][:-2]]:
                self.road_lane_mapping[link[1][:-2]].append(link[1])

        self._sort_roads()
        for key in self.road_lane_mapping.keys():
            for lane in self.road_lane_mapping[key]:
                self.lanes.append(lane)

        self.green_phases = phases
        self.phases = [i for i in range(len(phases))]
        self.phase_available_startlanes = []
        self.startlanes = []
        self.phase_available_lanelinks = []
        for r, p in enumerate(self.green_phases):
            tmp_lanelinks = []
            tmp_startane = []
            for n, i in enumerate(p.state):
                if i == 'G' or i == 's':
                    # skip if empty link
                    links = self.world.eng.trafficlight.getControlledLinks(self.id)
                    
                    # links = links[n]
                    if n >= len(links):
                        break
                    else:
                        links = links[n]
                                            
                    if not links:
                        continue
                    links = links[0]
                    tmp_lanelinks.append([links[0], links[1]])
                    if links[0] not in tmp_startane:
                        tmp_startane.append(links[0])
                    if links[0] not in self.startlanes:
                        self.startlanes.append(links[0])
            self.phase_available_startlanes.append(tmp_startane)
            self.phase_available_lanelinks.append(tmp_lanelinks)

        self.full_phases, self.yellow_dict = self.create_yellows(self.green_phases, self.yellow_phase_time, self.interface_flag)
        # programs = self.eng.trafficlight.getAllProgramLogics(self.id)
        tl_id = self.id + "_rl"
        logic = self.eng.trafficlight.Logic(tl_id, 0, 0, self.full_phases)
        self.eng.trafficlight.setProgramLogic(self.id, logic)
        self.eng.trafficlight.setProgram(self.id, tl_id)

        # dictionary of remembered features
        self.waiting_times = dict()
        self.full_observation = None
        self.last_step_vehicles = None

        # TODO: check .signals .full_observation .last_stet_vehicles need to be set or not

    def _sort_roads(self):
        '''
        _sort_roads
        Sort roads information by arranging an order.
        
        :param: None
        :return: None
        '''
        order = sorted(range(len(self.roads)),
                       key=lambda i: (self.directions[i],
                                      self.outs[i] if self.world.RIGHT else not self.outs[i]))
        self.roads = [self.roads[i] for i in order]
        self.directions = [self.directions[i] for i in order]
        self.outs = [self.outs[i] for i in order]
        self.out_roads = [self.roads[i] for i, x in enumerate(self.outs) if x]
        self.in_roads = [self.roads[i] for i, x in enumerate(self.outs) if not x]  # TODO: check if its 4

    def reset(self):
        '''
        reset
        Reset information, including current_phase, full_observation and last_step_vehicles, etc.
        
        :param: None
        :return: None
        '''
        self.current_phase_time = 0
        self.virtual_phase = 0
        self.next_phase = 0
        self.waiting_times = dict()
        self.full_observation = None
        self.last_step_vehicles = None
        self.current_phase = self.get_current_phase()
        # eng is set in world
        programs = self.eng.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.full_phases
        self.eng.trafficlight.setProgramLogic(self.id, logic)

    def get_current_phase(self):
        '''
        get_current_phase
        Get current phase of current intersection.
        
        :param: None
        :return cur_phase: current phase of current intersection
        '''
        cur_phase = self.eng.trafficlight.getPhase(self.id)
        return cur_phase

    # TODO: change cityflow phase generator into phase property
    def prep_phase(self, new_phase):
        '''
        prep_phase
        Prepare change phase of current intersection

        :param new_phase: phase that will be executed in the later
        :return: None
        '''
        if self.get_current_phase() == new_phase:
            self.next_phase = self.get_current_phase()
            if self.interface_flag:
                self.eng.trafficlight.setPhase(self.id, int(self.next_phase))
            else:
                self.eng.trafficlight.setPhase(self.id, self.next_phase)
            self.current_phase = self.get_current_phase()
        else:
            self.next_phase = new_phase
            # find yellow phase between cur and next phases
            y_key = str(self.get_current_phase()) + '_' + str(new_phase)
            if y_key in self.yellow_dict:
                y_id = self.yellow_dict[y_key]
                if self.interface_flag:
                    self.eng.trafficlight.setPhase(self.id, int(y_id))  # phase turns into yellow here
                else:
                    self.eng.trafficlight.setPhase(self.id, y_id)  # phase turns into yellow here
                self.current_phase = self.get_current_phase()

    def _change_phase(self, phase):
        '''
        _change_phase
        Change phase at current intersection.
        
        :param phase: phase to be executed at the next step
        :return: None
        '''
        if self.interface_flag:
            self.eng.trafficlight.setPhase(self.id, int(phase))
        else:
            self.eng.trafficlight.setPhase(self.id, phase)
        self.current_phase = self.get_current_phase()

    def pseudo_step(self, action):
        '''
        pseudo_step
        Take relative actions and calculate time duration of current phase.
        
        :param action: the changes to take
        :return: None
        '''
        ctrl = self.world.crossing_proxy_ctrl
        if ctrl is not None and ctrl.is_phase_locked(self.id):
            action = self.virtual_phase
        # TODO: check if change state, yellow phase must less than minimum of action time
        # test yellow finished first
        self.virtual_phase = action
        if self.current_phase_time >= self.yellow_phase_time:
            self._change_phase(action)
        else:
            if action != self.get_current_phase() and self.current_phase_time > self.yellow_phase_time:
                self.current_phase_time = 0
            if self.current_phase_time == 0:
                self.prep_phase(action)
            elif self.current_phase_time < self.yellow_phase_time:
                self._change_phase(self.current_phase)
            else:
                self._change_phase(action)

        self.current_phase_time += 1

    def observe(self, step_length, distance):
        '''
        observe
        Get observation of the whole roadnet, including lane_waiting_time_count, lane_waiting_count, lane_count and queue_length.
        
        :param step_length: time duration of step
        :param distance: distance limitation that it can only get vehicles which are within the length of the road
        :return: None
        '''
        full_observation = dict()
        all_vehicles = set()
        for lane in self.lanes:
            vehicles = []
            lane_measures = {'lane_waiting_time_count': 0, 'lane_waiting_count': 0, 'lane_count': 0, 'queue_length': 0}
            lane_vehicles = self._get_vehicles(lane, distance)
            for v in lane_vehicles:
                all_vehicles.add(v)
                if v in self.waiting_times:
                    self.waiting_times[v] += step_length
                elif self.eng.vehicle.getWaitingTime(v) > 0:
                    self.waiting_times[v] = self.eng.vehicle.getWaitingTime(v)
                v_measures = dict()
                v_measures['name'] = v
                v_measures['wait'] = self.waiting_times[v] if v in self.waiting_times else 0
                #TODO: CHEC ITS RIGHT CALCULATION?
                lane_measures['queue_length'] = lane_measures['queue_length'] + 1
                v_measures['speed'] = self.eng.vehicle.getSpeed(v)
                v_measures['position'] = self.eng.vehicle.getLanePosition(v)
                vehicles.append(v_measures)
                if v_measures['wait'] > 0:
                    lane_measures['lane_waiting_time_count'] += v_measures['wait']
                    lane_measures['lane_waiting_count'] += 1
                lane_measures['lane_count'] += 1
            lane_measures['vehicles'] = vehicles
            full_observation[lane] = lane_measures
        """
        full_observation['num_vehicles'] = all_vehicles
        if self.last_step_vehicles is None:
            full_observation['arrivals'] = full_observation['num_vehicles']
            full_observation['departures'] = set()
        else:
            full_observation['arrivals'] = self.last_step_vehicles.difference(all_vehicles)
            departs = all_vehicles.difference(self.last_step_vehicles)
            full_observation['departures'] = departs
            # Clear departures from waiting times
            for vehicle in departs:
                if vehicle in self.waiting_times: self.waiting_times.pop(vehicle)
        self.last_step_vehicles = all_vehicles
        """
        self.full_observation = full_observation

    def _get_vehicles(self, lane, max_distance):
        '''
        _get_vehicles
        Get number of vehicles running on the specific lane within max distance.
        
        :param lane: lane id
        :param max_distance: distance limitation that it can only get vehicles which are within the length of the lane
        :return detectable: number of vehicles
        '''
        # TODO: reduce complexity -> find all vehicles within max_distance and on this lane
        detectable = []
        for v in self.eng.lane.getLastStepVehicleIDs(lane):
            path = self.eng.vehicle.getNextTLS(v)
            if len(path) > 0:
                next_light = path[0]
                distance = next_light[2]
                if distance <= max_distance:
                    # Partial observability: only "connected"/detected vehicles are
                    # counted. At p=1.0 this is a no-op (every vehicle visible).
                    if self.world._vehicle_visible(v):
                        detectable.append(v)
        return detectable

    # TODO: revert x and y
    def _get_direction(self, road, out=True):
        if out:
            x = road[1][0] - road[0][0]
            y = road[1][1] - road[0][1]
        else:
            x = road[-2][0] - road[-1][0]
            y = road[-2][1] - road[-1][1]
        tmp = atan2(x, y)
        return tmp if tmp >= 0 else (tmp + 2 * pi)

    def create_yellows(self, phases, yellow_length, interface_flag):
        # interface_flag: 1:libsumo, 0: traci
        new_phases = copy.copy(phases)
        yellow_dict = {}    # current phase + next phase keyed to corresponding yellow phase index
        # Automatically create yellow phases, traci will report missing phases as it assumes execution by index order
        for i in range(0, len(phases)):
            for j in range(0, len(phases)):
                if i != j:
                    need_yellow, yellow_str = False, ''
                    for sig_idx in range(len(phases[i].state)):
                        if (phases[i].state[sig_idx] == 'G' or phases[i].state[sig_idx] == 'g') and (phases[j].state[sig_idx] == 'r' or phases[j].state[sig_idx] == 's'):
                            need_yellow = True
                            yellow_str += 'r'
                        else:
                            yellow_str += phases[i].state[sig_idx]
                    if need_yellow:  # If a yellow is required
                        if interface_flag:
                            new_phases.append(libsumo.trafficlight.Phase(yellow_length, yellow_str))
                        else:
                            new_phases.append(traci.trafficlight.Phase(yellow_length, yellow_str))
                        yellow_dict[str(i) + '_' + str(j)] = len(new_phases) - 1  # The index of the yellow phase in SUMO
        return new_phases, yellow_dict


# Ghost: speedMode 22 + setSpeed ignores car-following; getAllowedSpeed gates red lights.
GHOST_SPEED_MODE = 22


def _physics_flags(mode):
    """Return SUMO CLI flags for the given physics mode."""
    if mode == 'standard':
        return []
    if mode == 'ghost':
        return ['--collision.action', 'none', '--time-to-teleport', '-1']
    raise ValueError(f"Unknown physics_mode: {mode}")


class CrossingProxyController(object):
    """
    Actuated pedestrian-crossing proxy (see docs/CROSSING_PROXY.md).

    NEMA concurrent walk: on a ped call at through-green entry, hold the current
    green phase for T seconds (actuated max extension / walk+FDW) and halt any
    crosswalk-conflicting lanes that still have G/g/s during that phase.
    """

    def __init__(self, world, lane_map_path, world_param):
        self.world = world
        self.lane_map = {}
        self.through_phases = set()
        self.last_phase = {}
        self.active = {}
        self.phase_lock_until = {}
        self.rng = {}
        self.event_count = 0
        self.nominal_speed = {}

        self.call_prob = float(world_param.get('crossing_call_prob', 0.12))
        self.service_min = float(world_param.get('crossing_service_min', 7.0))
        self.service_max = float(world_param.get('crossing_service_max', 10.0))
        seed_offset = int(world_param.get('crossing_seed_offset', 7))
        _cmd_seed = Registry.mapping['command_mapping']['setting'].param.get('seed')
        base_seed = int(_cmd_seed if _cmd_seed is not None else world_param.get('seed', 0))

        with open(lane_map_path, encoding='utf-8') as f:
            payload = json.load(f)
        self.lane_map = payload.get('intersections', payload)
        self.through_phases = set(payload.get('through_phases', ('0', '4')))
        for tls_id in self.lane_map:
            digest = hashlib.md5(f"{base_seed}:{tls_id}".encode()).hexdigest()
            subseed = int(digest[:8], 16) + seed_offset
            self.rng[tls_id] = random.Random(subseed)
            self.last_phase[tls_id] = None

        print(
            f"[CrossingProxy] actuated extension — p={self.call_prob}, "
            f"service=[{self.service_min},{self.service_max}]s, "
            f"TLS={len(self.lane_map)}, seed_base={base_seed}"
        )

    @property
    def eng(self):
        return self.world.eng

    def nominal_lane_speed(self, lane_id, default=11.11):
        """Pre-halt lane max speed for metrics when setMaxSpeed(lane, 0) is active."""
        return self.nominal_speed.get(lane_id, default)

    def cache_nominal_speeds(self):
        for phases in self.lane_map.values():
            if not isinstance(phases, dict):
                continue
            for phase_data in phases.values():
                lanes = self._service_lanes(phase_data)
                for lane in lanes:
                    if lane in self.nominal_speed:
                        continue
                    try:
                        speed = self.eng.lane.getMaxSpeed(lane)
                        if speed > 0:
                            self.nominal_speed[lane] = speed
                    except Exception:
                        pass

    @staticmethod
    def _service_lanes(phase_data):
        if isinstance(phase_data, list):
            return phase_data
        if not isinstance(phase_data, dict):
            return []
        lanes = []
        lanes.extend(phase_data.get("through_incoming", []))
        lanes.extend(phase_data.get("conflict_lanes", []))
        return list(dict.fromkeys(lanes))

    def is_phase_locked(self, tls_id):
        return self.eng.simulation.getTime() < self.phase_lock_until.get(tls_id, -1)

    def step_before_actions(self):
        """Run before pseudo_step so locks apply the same second a ped call starts."""
        now = self.eng.simulation.getTime()
        self._expire_events(now)
        for tls_id in self.lane_map:
            intsec = self.world.id2intersection.get(tls_id)
            if intsec is None:
                continue
            phase = intsec.virtual_phase
            prev = self.last_phase.get(tls_id)
            if prev is not None and phase != prev:
                self._maybe_start_event(tls_id, phase, now)
            self.last_phase[tls_id] = phase

    def _maybe_start_event(self, tls_id, phase, now):
        phase_key = str(phase)
        if phase_key not in self.through_phases:
            return
        if tls_id in self.phase_lock_until and now < self.phase_lock_until[tls_id]:
            return
        if self.rng[tls_id].random() >= self.call_prob:
            return
        duration = self.rng[tls_id].uniform(self.service_min, self.service_max)
        phase_data = self.lane_map.get(tls_id, {}).get(phase_key, [])
        service_lanes = self._service_lanes(phase_data)
        self._start_event(tls_id, phase_key, service_lanes, now + duration)

    def _start_event(self, tls_id, phase_key, service_lanes, end_time):
        self.phase_lock_until[tls_id] = end_time
        saved = {}
        for lane in service_lanes:
            try:
                speed = self.eng.lane.getMaxSpeed(lane)
                if speed > 0:
                    self.nominal_speed[lane] = speed
                elif lane in self.nominal_speed:
                    speed = self.nominal_speed[lane]
                saved[lane] = speed
                self.eng.lane.setMaxSpeed(lane, 0.0)
            except Exception:
                continue
        if saved:
            self.active[tls_id] = {
                'end_time': end_time,
                'saved': saved,
            }
        self.event_count += 1
        if self.event_count == 1 or self.event_count % 100 == 0:
            print(
                f"[CrossingProxy] ped calls={self.event_count} "
                f"(latest tls={tls_id} phase={phase_key} "
                f"lanes={len(service_lanes)} until t={end_time:.0f})"
            )

    def reset(self):
        self._clear_all_halts()
        self.active.clear()
        self.phase_lock_until.clear()
        self.event_count = 0
        for tls_id in self.lane_map:
            intsec = self.world.id2intersection.get(tls_id)
            if intsec is not None:
                self.last_phase[tls_id] = intsec.virtual_phase
            else:
                self.last_phase[tls_id] = None
        self.cache_nominal_speeds()

    def log_summary(self):
        logging.info(
            "[CrossingProxy] total ped calls this episode: %d",
            self.event_count,
        )

    def _expire_events(self, now):
        finished = [
            tls for tls, end in self.phase_lock_until.items()
            if now >= end
        ]
        for tls in finished:
            self.phase_lock_until.pop(tls, None)
            self._restore_event(tls)

    def _restore_event(self, tls_id):
        ev = self.active.pop(tls_id, None)
        if not ev:
            return
        for lane, speed in ev['saved'].items():
            try:
                self.eng.lane.setMaxSpeed(lane, speed)
            except Exception:
                pass

    def _clear_all_halts(self):
        for tls_id in list(self.active.keys()):
            self._restore_event(tls_id)


@Registry.register_world('sumo')
class World(object):
    '''
    World Class is mainly used for creating a SUMO engine and maintain information about SUMO world.
    '''
    def __init__(self, sumo_config, placeholder=0, **kwargs):
        if kwargs['interface'] == 'libsumo':
            if libsumo is None:
                print("[Warning] libsumo Python module not found; falling back to traci.")
                kwargs['interface'] = 'traci'
                self.interface_flag = False
            else:
                self.interface_flag = True
        elif kwargs['interface'] == 'traci':
            self.interface_flag = False
        else:
            raise Exception('NOT IMPORTED YET')

        world_param = Registry.mapping['world_mapping']['setting'].param
        self.physics_mode = world_param.get('physics_mode', 'standard')
        # Only implemented for libusmo for now
        if self.physics_mode == 'ghost' and kwargs['interface'] != 'libsumo':
            raise ValueError("physics_mode='ghost' requires --interface libsumo")

        # Realism toggles (composable; see docs/REALISM_FULL.md for all-axes profile).
        self.hetero = world_param.get('hetero', False)
        self.slow_start = world_param.get('slow_start', False)

        # Partial observability (see docs/PARTIAL_OBSERVABILITY.md). Two composable axes that
        # corrupt what the controller perceives while leaving the SUMO physics untouched:
        #   (1) penetration rate — each vehicle is independently "visible"/connected with
        #       probability p, PERSISTENT for its whole trip; lane stats are built from
        #       visible vehicles only. p=1.0 => full observability (exact baseline, no draws).
        #   (2) Gaussian count noise — each per-lane vehicle count is corrupted by zero-mean
        #       Gaussian measurement noise (noisy loop/camera detectors), re-sampled each
        #       step, rounded and clamped >=0. sigma=0.0 => exact counts (baseline).
        #       obs_noise_mode: 'additive' -> N(0,sigma^2); 'proportional' -> N(0,(sigma*true)^2).
        # Both compose: penetration reduces counts to the visible subset, then noise perturbs
        # them. The seed is resolved identically to SUMO/trainer (CLI --seed first, else the
        # config world.seed) so --seed overrides the corruption exactly as it does everywhere
        # else, and the same seeded corruption is applied for every agent so ground-truth ATT
        # stays comparable across DQN/PressLight/CoLight/MaxPressure.
        _obs_cmd_seed = Registry.mapping['command_mapping']['setting'].param.get('seed')
        self.obs_seed = int(_obs_cmd_seed if _obs_cmd_seed is not None else world_param.get('seed', 0))
        self.obs_penetration = float(world_param.get('obs_penetration', 1.0))
        self.obs_count_noise_std = float(world_param.get('obs_count_noise_std', 0.0))
        self.obs_noise_mode = str(world_param.get('obs_noise_mode', 'additive'))
        if self.obs_penetration < 1.0:
            print(f"[PartialObs] penetration rate p={self.obs_penetration} "
                  f"(persistent per-vehicle, seed={self.obs_seed})")
        if self.obs_count_noise_std > 0.0:
            print(f"[PartialObs] Gaussian count noise sigma={self.obs_count_noise_std} "
                  f"mode={self.obs_noise_mode} (per-step, seed={self.obs_seed})")

        # Crossing proxy — actuated ped crossing delay on conflicting lanes (see
        # docs/CROSSING_PROXY.md). Composable with other realism axes.
        self.crossing_proxy = bool(world_param.get('crossing_proxy', False))
        self.crossing_proxy_ctrl = None

        with open(sumo_config) as f:
            sumo_dict = json.load(f)

        if self.crossing_proxy:
            lane_map_rel = sumo_dict.get('crossingProxyLanes', '')
            if not lane_map_rel:
                raise ValueError(
                    "crossing_proxy=true but crossingProxyLanes not set in .cfg"
                )
            lane_map_path = os.path.join(sumo_dict['dir'], lane_map_rel)
            if not os.path.isfile(lane_map_path):
                raise FileNotFoundError(
                    f"crossing_proxy lane map not found: {lane_map_path}"
                )
            self.crossing_proxy_ctrl = CrossingProxyController(
                self, lane_map_path, world_param
            )
        self._use_gui = sumo_dict['gui'] == "True" or sumo_dict['gui'] == True

        physics_flags = _physics_flags(self.physics_mode)
        if self.physics_mode == 'ghost':
            print("[Physics] mode=ghost (libsumo, obey signals, ignore other cars)")

        route_file = sumo_dict['flowFile']
        additional_files = []
        if self.hetero and self.slow_start:
            if not sumo_dict.get('flowFileHetero'):
                raise ValueError(
                    "hetero+slow_start requires flowFileHetero in .cfg"
                )
            route_file = sumo_dict['flowFileHetero']
            add_rel = sumo_dict.get('realismFullAdditional', '')
            if not add_rel:
                raise ValueError(
                    "hetero+slow_start requires realismFullAdditional in .cfg"
                )
            additional_files.append(os.path.join(sumo_dict['dir'], add_rel))
            print(
                f"[RealismFull] hetero+slow_start — routes={route_file}, "
                f"vTypes={add_rel}"
            )
        elif self.hetero:
            if not sumo_dict.get('flowFileHetero'):
                raise ValueError("hetero=true but flowFileHetero not set in .cfg")
            route_file = sumo_dict['flowFileHetero']
            add_file = sumo_dict.get('heteroAdditional', '')
            if add_file:
                additional_files.append(os.path.join(sumo_dict['dir'], add_file))
            print(f"[Hetero] enabled — routes={route_file}, vTypes={add_file}")

        elif self.slow_start:
            add_file = sumo_dict.get('slowStartAdditional', '')
            if not add_file:
                raise ValueError("slow_start=true but slowStartAdditional not set in .cfg")
            additional_files.append(os.path.join(sumo_dict['dir'], add_file))
            slow_route = sumo_dict.get('flowFileSlowStart', '')
            if not slow_route:
                raise ValueError("slow_start=true but flowFileSlowStart not set in .cfg")
            route_file = slow_route
            print(f"[SlowStart] enabled — routes={route_file}, vTypes={add_file}")

        # Optional demand-set / fixed-file override (after physics route selection so
        # realism_full can keep hetero vTypes while swapping OD hub routes).
        demand_train_file = world_param.get('demand_train_file')
        demand_set = world_param.get('demand_set') or world_param.get('demand_bag') or []
        if demand_train_file:
            route_file = demand_train_file
            print(f"[Demand] train file override={route_file}")

        additional_flags = []
        if additional_files:
            additional_flags = ['--additional-files', ','.join(additional_files)]

        # Shared simulation arguments (network/route + flags); the binary is chosen per use.
        # Tie SUMO's RNG (vehicle insertion, etc.) to the global seed so demand stochasticity
        # is reproducible. Resolve the seed identically to the trainer: the CLI --seed takes
        # precedence, falling back to the config world.seed (default 0). So --seed overrides
        # the config for SUMO exactly as it does for random/numpy/torch.
        _cmd_seed = Registry.mapping['command_mapping']['setting'].param.get('seed')
        effective_seed = _cmd_seed if _cmd_seed is not None else world_param.get('seed', 0)
        seed_flags = ['--seed', str(int(effective_seed))]
        # Use explicit -n/-r when realism toggles need additional-files or alternate routes.
        # Demand-set rotation also requires -r so set_route_file can swap files.
        use_explicit_net_route = (
            not sumo_dict.get('combined_file') or self.hetero or self.slow_start
            or self.crossing_proxy
            or bool(demand_train_file)
            or bool(demand_set)
        )
        self._dir = sumo_dict['dir']
        self._roadnet_file = os.path.join(sumo_dict['dir'], sumo_dict['roadnetFile'])
        self._physics_flags = physics_flags
        self._additional_flags = additional_flags
        self._seed_flags = seed_flags
        self._no_warning = str(sumo_dict['no_warning'])
        self._use_explicit_net_route = use_explicit_net_route
        self._combined_file = (
            os.path.join(sumo_dict['dir'], sumo_dict['combined_file'])
            if sumo_dict.get('combined_file') else None
        )
        self._headless_bin = sumolib.checkBinary('sumo')
        self._gui_bin = sumolib.checkBinary('sumo-gui') if self._use_gui else None
        self._gui_flags = []
        if self._use_gui:
            self._gui_flags = ['--delay', '0', '--threads', '4']
            if sumo_dict.get('gui_step_length'):
                self._gui_flags += ['--step-length', str(sumo_dict['gui_step_length'])]

        self.net = self._roadnet_file
        self.route = os.path.join(sumo_dict['dir'], route_file)
        self.route = self._hetero_route_with_fleet_mix(self.route)
        self.route = self._slow_start_route_without_inline_vtype(self.route)
        self.warning = sumo_dict['no_warning']
        self._rebuild_sumo_cmd()
        print("building world...")
        self.connection_name = sumo_dict['name']
        self.map = sumo_dict['roadnetFile'].split('/')[-1].split('.')[0]
        
        if self.interface_flag:
            libsumo.start(self.warmup_cmd)
            self.eng = libsumo
        else:
            if not sumo_dict['name']:
                traci.start(self.warmup_cmd)
                self.eng = traci
            else:
                traci.start(self.warmup_cmd, label=sumo_dict['name'])
                self.eng = traci.getConnection(sumo_dict['name'])
        # TODO: roadnet not implemented but not necessary
        self.RIGHT = True  # TODO: currently set to be true
        self.interval = sumo_dict['interval']
        self.step_ratio = 1  # TODO: register in Registry later
        self.step_length = 1  # should be 1 in our setting
        self.max_distance = 200 # TODO: set in registry
        # get all intersections (dict here)
        self.intersection_ids = self.eng.trafficlight.getIDList()
        # prepare phase information for each intersections
        self._sim_ready = False
        self.green_phases = self.generate_valid_phase()

        # creating all intersections
        self.id2intersection = dict()
        self.intersections = []
        for ts in self.eng.trafficlight.getIDList():
            self.id2intersection[ts] = Intersection(ts, self, self.green_phases[ts])  # this IntSec has different phases
            self.intersections.append(self.id2intersection[ts])
        self.id2idx = {i: idx for idx,i in enumerate(self.id2intersection)}
        # TODO: to see if its necessary to test .intersections or .observe here
        # TODO: to see if pass observation and its shape by generator
        self.all_roads = [x for x in self.eng.edge.getIDList()]
        self.all_lanes = [ x for x in self.eng.lane.getIDList()]
        # for itsec in self.intersections:
        #     for road in itsec.road_lane_mapping.keys():
        #         if itsec.road_lane_mapping[road] and road not in self.all_roads:
        #             # append road name into all_roads if road exists
        #             self.all_roads.append(road)
                    # for lane in itsec.road_lane_mapping[road]:
                    #     if lane not in self.all_lanes:
                    #         self.all_lanes.append(lane)

        # restart eng
        self.run = 0
        self.inside_vehicles = dict()
        self.vehicles = dict()
        # Uncapped trip wait (speed < 0.1 m/s). Not SUMO getAccumulatedWaitingTime (100s cap).
        self.track_trip_wait = False
        self.vehicle_wait_s = dict()
        self.departed_wait_s = dict()
        for intsec in self.intersections:
            intsec.observe(self.step_length, self.max_distance)
        if self.interface_flag:
            if not self.connection_name: 
                libsumo.switch(self.connection_name)  # TODO: make sure what's this step doing
            libsumo.close()
        else:
            if not self.connection_name: 
                traci.switch(self.connection_name)  # TODO: make sure what's this step doing
            traci.close()
        # self.connection_name = self.map + '-' + self.connection_name
        if not os.path.exists(os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                           self.connection_name)):
            os.mkdir(os.path.join(Registry.mapping['logger_mapping']['path'].path, self.connection_name))

        print('Connection ID', self.connection_name)

        self.info_functions = {
            "vehicles": self.get_vehicles, # TODO check this func
            "lane_count": self.get_lane_vehicle_count,
            "lane_waiting_count": self.get_lane_waiting_vehicle_count,
            "lane_vehicles": self.get_lane_vehicles,
            "time": self.get_current_time,
            "vehicle_distance": None,
            "pressure": self.get_pressure,
            "lane_pressure": self.get_lane_pressure,
            "lane_waiting_time_count": self.get_lane_waiting_time_count,
            "lane_delay": self.get_lane_delay,
            "real_delay": self.get_real_delay,
            "vehicle_trajectory": self.get_vehicle_trajectory,
            "history_vehicles": None,
            "phase": self.get_cur_phase,
            "throughput": self.get_cur_throughput,
            "average_travel_time": None
        }
        self.fns = []
        self.info = {}
        # test generate observation information
        self.vehicle_trajectory = {}
        self.vehicle_maxspeed = {}
        self.real_delay = {}
        # Per-step vehicle_trajectory maintenance is expensive (scans all lanes).
        # Enable only when needed: subscribed "vehicle_trajectory", or real-delay
        # metrics (trainer sets this when --delay_type real). Default agents
        # (DQN/maxpressure/etc.) do not need it.
        self.update_vehicle_trajectory = False

        # get in_lanes and out_lanes
        self.in_lanes, self.out_lanes = self.get_in_out_lanes()
        self._sim_ready = True

    def generate_valid_phase(self):
        '''
        generate_valid_phase
        Generate valid phases that will be executed by intersections later.
        
        :param: None
        :return valid_phases: valid phases that will be executed by intersections later.
        '''
        valid_phases = dict()
        for lightID in self.intersection_ids:
            logic = self.eng.trafficlight.getAllProgramLogics(lightID)[0]
            seen_states = []
            for phase in logic.phases:
                state = phase.state
                if state not in seen_states:
                    seen_states.append(state)
            valid_phases[lightID] = seen_states
        for ts in valid_phases:
            green_phases = []
            for phase in valid_phases[ts]:     # Convert to SUMO phase type
                if 'y' not in phase:
                    if phase.count('r') + phase.count('s') != len(phase):
                        green_phases.append(self.eng.trafficlight.Phase(self.step_length, phase))
            valid_phases[ts] = green_phases
        return valid_phases

    def step_sim(self):
        '''
        step_sim
        Simulate 1s. The monaco scenario expects .25s steps instead of 1s, account for that here.
        
        :param: None
        :return: None
        '''
        if self.physics_mode == 'ghost' and self._sim_ready:
            self._enforce_ghost_physics_all()
            self._collapse_ghost_lanes()
        for _ in range(self.step_ratio):
            self.eng.simulationStep()
        if self.physics_mode == 'ghost' and self._sim_ready:
            self._enforce_ghost_physics_all()
            self._collapse_ghost_lanes()

    def _enforce_ghost_vehicle(self, veh_id):
        """Ignore other vehicles but obey signals via getAllowedSpeed + setSpeed."""
        self.eng.vehicle.setSpeedMode(veh_id, GHOST_SPEED_MODE)
        self.eng.vehicle.setMinGap(veh_id, 0)
        self.eng.vehicle.setTau(veh_id, 0)
        allowed = self.eng.vehicle.getAllowedSpeed(veh_id)
        if allowed < 0.1:
            self.eng.vehicle.setSpeed(veh_id, 0)
        else:
            self.eng.vehicle.setSpeed(veh_id, self.eng.vehicle.getMaxSpeed(veh_id))

    def _enforce_ghost_physics_all(self):
        for veh_id in self.eng.vehicle.getIDList():
            self._enforce_ghost_vehicle(veh_id)

    def _collapse_ghost_lanes(self):
        """Teleport same-lane followers onto the lead vehicle so gaps stay at 0."""
        by_lane = {}
        for veh_id in self.eng.vehicle.getIDList():
            lane = self.eng.vehicle.getLaneID(veh_id)
            if lane.startswith(':'):
                continue
            by_lane.setdefault(lane, []).append(veh_id)
        for lane, veh_ids in by_lane.items():
            if len(veh_ids) < 2:
                continue
            lead = max(veh_ids, key=lambda v: self.eng.vehicle.getLanePosition(v))
            lead_pos = self.eng.vehicle.getLanePosition(lead)
            for veh_id in veh_ids:
                if veh_id == lead:
                    continue
                self.eng.vehicle.moveTo(veh_id, lane, lead_pos)

    def step(self, action=None):
        '''
        step
        Take relative actions and update information.
        
        :param actions: actions list to be executed at all intersections at the next step
        :return: None
        '''
        # TODO: support interval != 1
        if self.crossing_proxy_ctrl is not None:
            self.crossing_proxy_ctrl.step_before_actions()
        if action is not None:
            for i, intersection in enumerate(self.intersections):
                intersection.pseudo_step(action[i])
            self.step_sim()
        for intsec in self.intersections:
            intsec.observe(self.step_length, self.max_distance)
        # TODO: register vehicles here
        entering_v = self.eng.simulation.getDepartedIDList()
        for v in entering_v:
            self.inside_vehicles.update({v: self.get_current_time()})
            if self.track_trip_wait:
                self.vehicle_wait_s.setdefault(v, 0.0)
            if self.physics_mode == 'ghost':
                self._enforce_ghost_vehicle(v)
        if self.physics_mode == 'ghost' and entering_v:
            self._collapse_ghost_lanes()
        if self.track_trip_wait:
            self._accumulate_trip_wait()
        exiting_v = self.eng.simulation.getArrivedIDList()
        for v in exiting_v:
            if v not in self.inside_vehicles:
                continue
            self.vehicles.update({v: self.get_current_time() - self.inside_vehicles[v]})
            if self.track_trip_wait:
                self.departed_wait_s[v] = float(self.vehicle_wait_s.pop(v, 0.0))
            del self.inside_vehicles[v]
        self._update_infos()
        if self._trajectory_tracking_enabled():
            self.vehicle_trajectory, self.vehicle_maxspeed = self.get_vehicle_trajectory()
        self.run += 1

    def _trajectory_tracking_enabled(self):
        '''
        _trajectory_tracking_enabled
        Whether per-step vehicle trajectory maintenance should run.

        :param: None
        :return: bool
        '''
        return self.update_vehicle_trajectory or ("vehicle_trajectory" in self.fns)

    def _rebuild_sumo_cmd(self):
        '''Rebuild sumo_cmd / warmup_cmd from current net + route paths.'''
        if self._use_explicit_net_route:
            sim_args = [
                '-n', self._roadnet_file,
                '-r', self.route,
                '--no-warnings', self._no_warning,
            ] + self._seed_flags + self._physics_flags + self._additional_flags
        else:
            sim_args = [
                '-c', self._combined_file,
                '--no-warnings', self._no_warning,
            ] + self._seed_flags + self._physics_flags + self._additional_flags
        # Warm-up always headless; GUI (if any) opens in reset().
        self.warmup_cmd = [self._headless_bin] + sim_args
        if self._use_gui:
            self.sumo_cmd = [self._gui_bin] + self._gui_flags + sim_args
        else:
            self.sumo_cmd = [self._headless_bin] + sim_args

    def _hetero_route_with_fleet_mix(self, route_abs):
        """Hub-OD demand overrides flowFileHetero and leaves type=pkw. When
        hetero is on, rewrite those files to 80/20 car/truck so the additional
        vTypes (vTypes_mixed or vTypes_realism_full) actually attach.
        """
        if not getattr(self, 'hetero', False):
            return route_abs
        return rewrite_route_file_hetero_mix(route_abs)

    def _slow_start_route_without_inline_vtype(self, route_abs):
        """Hub-OD .rou.xml embeds <vType id="pkw">. slowStartAdditional also
        defines pkw, so SUMO exits immediately with a duplicate-id error.
        Write a sibling without the inline vType so the additional file wins
        (same recipe as extras/gen_slow_start_routes.py). Only for
        slow_start-alone; hetero+slow_start uses car/truck vTypes.
        """
        if not (getattr(self, 'slow_start', False) and not getattr(self, 'hetero', False)):
            return route_abs
        if not route_abs or not os.path.isfile(route_abs):
            return route_abs
        try:
            tree = ET.parse(route_abs)
        except ET.ParseError:
            return route_abs
        root = tree.getroot()
        vtypes = [el for el in list(root) if el.tag.split('}')[-1] == 'vType']
        if not vtypes:
            return route_abs
        cache_dir = os.path.join(os.path.dirname(route_abs), '.slow_start_notype')
        os.makedirs(cache_dir, exist_ok=True)
        dst = os.path.join(cache_dir, os.path.basename(route_abs))
        if os.path.isfile(dst) and os.path.getmtime(dst) >= os.path.getmtime(route_abs):
            return dst
        for el in vtypes:
            root.remove(el)
        for veh in root.iter():
            if veh.tag.split('}')[-1] == 'vehicle' and 'type' not in veh.attrib:
                veh.set('type', 'pkw')
        tree.write(dst, encoding='utf-8', xml_declaration=True)
        print(f"[SlowStart] stripped inline vType -> {dst}")
        return dst

    def set_route_file(self, route_rel):
        '''
        Swap the demand/route file used on the next reset().

        :param route_rel: path relative to world dir (usually data/), or absolute
        '''
        route_abs = route_rel if os.path.isabs(route_rel) else os.path.join(self._dir, route_rel)
        if not os.path.exists(route_abs):
            raise FileNotFoundError(f"demand route file not found: {route_abs}")
        route_abs = self._hetero_route_with_fleet_mix(route_abs)
        route_abs = self._slow_start_route_without_inline_vtype(route_abs)
        if not self._use_explicit_net_route:
            self._use_explicit_net_route = True
        self.route = route_abs
        self._rebuild_sumo_cmd()
        print(f"[Demand] route -> {route_abs}")

    def reset(self):
        '''
        reset
        reset information, including vehicles, vehicle_trajectory, etc.
       
        :param: None
        :return: None
        '''
        if self.run != 0:
            if self.crossing_proxy_ctrl is not None:
                self.crossing_proxy_ctrl.reset()
            # TODO: test why need switch in original code
            if self.interface_flag:
                libsumo.close()
            else:
                traci.close()
        self.run = 0
        self.vehicles = dict()
        self.inside_vehicles = dict()
        self.vehicle_wait_s = dict()
        self.departed_wait_s = dict()
        # TODO: check when to close traci
        if self.interface_flag:
            libsumo.start(self.sumo_cmd)
            # TODO: set trip info output
            self.eng = libsumo
        else:
            traci.start(self.sumo_cmd, label=self.connection_name)
            self.eng = traci.getConnection(self.connection_name)
        self.id2intersection = dict()
        self.intersections = []
        for ts in self.eng.trafficlight.getIDList():
            self.id2intersection[ts] = Intersection(ts, self, self.green_phases[ts])  # this IntSec has different phases
            self.intersections.append(self.id2intersection[ts])
        self.id2idx = {i: idx for idx,i in enumerate(self.id2intersection)}

        for intsec in self.intersections:
            intsec.observe(self.step_length, self.max_distance)
        self._update_infos()
        # TODO: check if its the problem
        entering_v = self.eng.simulation.getDepartedIDList()
        for v in entering_v:
            self.inside_vehicles.update({v: self.get_current_time()})
            if self.physics_mode == 'ghost':
                self._enforce_ghost_vehicle(v)
        if self.physics_mode == 'ghost' and entering_v:
            self._collapse_ghost_lanes()
        if self.crossing_proxy_ctrl is not None:
            self.crossing_proxy_ctrl.reset()
        self.vehicle_trajectory = {}
        self.vehicle_maxspeed = {}
        self.real_delay= {}

    def get_current_time(self):
        '''
        get_current_time
        Get simulation time (in seconds).
        
        :param: None
        :return result: current time
        '''
        result = self.eng.simulation.getTime()
        return result

    def get_vehicles(self):
        '''
        get_vehicles
        Get all vehicle ids.
        
        :param: None
        :return: None
        '''
        result = 0
        count = 0
        for v in self.vehicles.keys():
            count += 1
            result += self.vehicles[v]
        if count == 0:
            return 0
        else:
            return result/count

    def subscribe(self, fns):
        '''
        subscribe
        Subscribe information you want to get when training the model.
        
        :param fns: information name you want to get
        :return: None
        '''
        if isinstance(fns, str):
            fns = [fns]
        for fn in fns:
            if fn in self.info_functions:
                if fn not in self.fns:
                    self.fns.append(fn)
            else:
                raise Exception(f'Info function {fn} not implemented')

    def get_info(self, info):
        '''
        get_info
        Get specific information.
        
        :param info: the name of the specific information
        :return _info: specific information
        '''
        _info = self.info[info]
        return _info

    def _update_infos(self):
        '''
        _update_infos
        Update global information after reset or each step.
        
        :param: None
        :return: None
        '''
        self._apply_obs_count_noise()
        self.info = {}
        for fn in self.fns:
            self.info[fn] = self.info_functions[fn]()

    def _vehicle_visible(self, veh_id):
        '''
        _vehicle_visible
        Partial-observability gate: return True if this vehicle is "connected"/detected
        under the current penetration rate. Visibility is PERSISTENT per vehicle (decided
        once, deterministically, from a seeded hash of the vehicle id) so a car is either
        observed for its whole trip or never — matching real connected-vehicle fleets.
        At p>=1.0 this short-circuits to True (no hashing, exact baseline behaviour).

        :param veh_id: SUMO vehicle id
        :return: bool, whether the vehicle is observable
        '''
        if self.obs_penetration >= 1.0:
            return True
        if self.obs_penetration <= 0.0:
            return False
        h = hashlib.md5(f"{self.obs_seed}:{veh_id}".encode()).digest()
        frac = int.from_bytes(h[:4], 'big') / 0xFFFFFFFF
        return frac < self.obs_penetration
    def _apply_obs_count_noise(self):
        '''
        _apply_obs_count_noise
        Partial observability (noisy detector): perturb the per-lane vehicle counts stored
        in each intersection's full_observation with Gaussian measurement noise, in place.
        Runs once per step before any info function reads the observation, so counts,
        queue length, waiting count and both pressure variants all inherit the SAME noisy
        realization consistently. At sigma<=0 this is a no-op (exact baseline).

        Ground-truth ATT/throughput use a separate path (self.vehicles) and are untouched.

        :param: None
        :return: None
        '''
        sigma = self.obs_count_noise_std
        if sigma <= 0.0:
            return
        # Per-step RNG keyed on (seed, sim time): reproducible and re-sampled each step.
        rng = random.Random((self.obs_seed * 1000003 + int(self.get_current_time())) & 0xFFFFFFFF)
        proportional = self.obs_noise_mode == 'proportional'
        for intsec in self.intersections:
            for lane in intsec.lanes:
                fo = intsec.full_observation[lane]
                # lane_count and queue_length are the same physical vehicle count here,
                # so they share one detector reading; waiting count is a separate detector.
                shared = self._noisy_count(fo['lane_count'], sigma, proportional, rng)
                fo['lane_count'] = shared
                fo['queue_length'] = shared
                fo['lane_waiting_count'] = self._noisy_count(
                    fo['lane_waiting_count'], sigma, proportional, rng)

    @staticmethod
    def _noisy_count(true_val, sigma, proportional, rng):
        '''Return an unbiased Gaussian-noised, rounded, non-negative count.'''
        s = sigma * true_val if proportional else sigma
        if s <= 0.0:
            return true_val
        noisy = true_val + rng.gauss(0.0, s)
        noisy = int(round(noisy))
        return noisy if noisy > 0 else 0

    def get_lane_vehicle_count(self):
        '''
        get_lane_vehicle_count
        Get number of vehicles in each lane.
        
        :param: None
        :return result: number of vehicles in each lane
        '''
        result = dict()
        for intsec in self.intersections:
            for lane in intsec.lanes:
                result.update({lane: intsec.full_observation[lane]['lane_count']})
        return result

    def get_pressure(self):
        '''
        get_pressure
        Get pressure of each intersection. 
        Pressure of an intersection equals to number of vehicles that in in_lanes minus number of vehicles that in out_lanes.
        
        :param: None
        :return pressures: pressure of each intersection
        '''
        pressures = dict()
        lane_vehicles = self.get_lane_vehicle_count()
        for i in self.intersections:
            pressure = 0
            for road in i.in_roads:
                for k in i.road_lane_mapping[road]:
                    pressure += lane_vehicles[k]
            for road in i.out_roads:
                for k in i.road_lane_mapping[road]:
                    pressure -= lane_vehicles[k]
            pressures[i.id] = pressure
        return pressures
        
    def get_in_out_lanes(self):
        in_lanes = []
        out_lanes = []
        for i in self.intersections:
            for road in i.in_roads:
                for lane in i.road_lane_mapping[road]:
                    in_lanes.append(lane)
            for road in i.out_roads:
                for lane in i.road_lane_mapping[road]:
                    out_lanes.append(lane)
        # add in_lanes of virtual intersections which can be regarded as out_lanes of non-virtual intersections.
        for lane in self.all_lanes:
            if lane not in out_lanes:
                out_lanes.append(lane)
        return in_lanes, out_lanes

    def get_lane_pressure(self):
        '''
        get_lane_pressure
        Get pressure of each lane in an intersection. 
        Pressure of each lane equals to number of vehicles that in the in_lane minus number of vehicles that in out_lane.
        
        :param: None
        :return pressures: pressure of each lane
        '''
        lvc = self.get_lane_vehicle_count()
        pressures = {}
        pressures = {x:0 for x in self.in_lanes}
        for inter_obj in self.intersections:
            for lanelink in inter_obj.lanelinks:
                start, end = lanelink[0][0], lanelink[0][1]
                pressures[start] += lvc[start]
                pressures[start] -= lvc[end]
        return pressures

    def get_lane_waiting_time_count(self):
        '''
        get_lane_waiting_time_count
        Get waiting time of vehicles in each lane.
        
        :param: None
        :return result: waiting time of vehicles in each lane
        '''
        result = dict()
        for intsec in self.intersections:
            for lane in intsec.lanes:
                result.update({lane: intsec.full_observation[lane]['lane_waiting_time_count']})
        return result

    def get_lane_waiting_vehicle_count(self):
        '''
        get_lane_waiting_vehicle_count
        Get number of waiting vehicles in each lane.
        
        :param: None
        :return result: number of waiting vehicles in each lane
        '''
        result = dict()
        for intsec in self.intersections:
            for lane in intsec.lanes:
                result.update({lane: intsec.full_observation[lane]['lane_waiting_count']})
        return result

    def get_cur_phase(self):
        '''
        get_cur_phase
        Get current phase of each intersection.

        :param: None
        :return result: current phase of each intersection
        '''
        result = []
        for intsec in self.intersections:
            result.append(intsec.get_current_phase())
        return result

    def get_average_travel_time(self):
        '''
        get_average_travel_time
        Get average travel time of all vehicles.
        
        :param: None
        :return tvg_time: average travel time of all vehicles
        '''
        tvg_time = self.get_vehicles()
        return tvg_time

    def _accumulate_trip_wait(self):
        '''Add step_length of wait for each in-network vehicle with speed < 0.1 m/s.'''
        dt = float(self.step_length)
        for v in self.eng.vehicle.getIDList():
            try:
                speed = self.eng.vehicle.getSpeed(v)
            except Exception:
                continue
            if speed < 0.1:
                self.vehicle_wait_s[v] = self.vehicle_wait_s.get(v, 0.0) + dt

    def get_median_waiting_time(self):
        '''Median uncapped wait (s) over finished trips this episode, or None.'''
        waits = [w for w in self.departed_wait_s.values() if w is not None]
        if not waits:
            return None
        waits.sort()
        n = len(waits)
        mid = n // 2
        if n % 2:
            return float(waits[mid])
        return 0.5 * (float(waits[mid - 1]) + float(waits[mid]))

    def get_lane_vehicles(self):
        '''
        get_lane_vehicles
        Get vehicles' id of each lane.

        :param: None
        :return vehicle_lane: vehicles' id of each lane
        '''
        result = dict()
        for inter in self.intersections:
            for key in inter.full_observation.keys():
                result.update({key: inter.full_observation[key]})
        return result

    def get_lane_queue_length(self):
        '''
        get_lane_queue_length
        Get queue length of all lanes in the traffic network.
        
        :param: None
        :return result: queue length of all lanes
        '''
        #TODO: CHECK DEFINATION
        result = dict()
        for inter in self.intersections:
            for key in inter.full_observation.keys():
                result.update({key: inter.full_observation[key]['queue_length']})
        return result

    def get_lane_delay(self):
        '''
        get_lane_delay
        Get approximate delay of each lane. 
        Approximate delay of each lane equals to (1 - lane_avg_speed)/lane_speed_limit.
        
        :param: None
        :return lane_delay: approximate delay of each lane
        '''
        # the delay of each lane: 1 - lane_avg_speed/speed_limit
        # set speed limit to 11.11 by default
        lane_vehicles = self.get_lane_vehicles()
        lane_delay = dict()
        for key in lane_vehicles.keys():
            vehicles = lane_vehicles[key]['vehicles']
            lane_vehicle_count = len(vehicles)
            lane_avg_speed = 0.0
            speed_limit = self.eng.lane.getMaxSpeed(key)
            if speed_limit <= 1e-9 and self.crossing_proxy_ctrl is not None:
                speed_limit = self.crossing_proxy_ctrl.nominal_lane_speed(key)
            if speed_limit <= 1e-9:
                lane_delay[key] = 1.0
                continue
            for vehicle in vehicles:
                speed = vehicle['speed']
                lane_avg_speed += speed
            if lane_vehicle_count == 0:
                lane_avg_speed = speed_limit
            else:
                lane_avg_speed /= lane_vehicle_count
            lane_delay[key] = 1 - lane_avg_speed / speed_limit
        return lane_delay

    # def get_plan_depart_time(self):
    #     """
    #     Get planned depart time for all vehicles appeared in sumo.rou.xml file.
    #     In SUMO and Cityflow, travel time = arriving time-planned depart time.
    #     Note: Not real depart time, but planned depart time.
    #     return: planned depart time of all vehicles.
    #     """
    #     vehicles_all = dict()
    #     tree = ET.parse(self.route)
    #     root = tree.getroot()
    #     vehicles_all.update({obj.attrib['id']: int(float(obj.attrib['depart'])) \
    #         for obj in root.iter('vehicle')})
    #     return vehicles_all

    def get_cur_throughput(self):
        '''
        get_cur_throughput
        Get vehicles' count in the whole roadnet at current step.

        :param: None
        :return throughput: throughput in the whole roadnet at current step
        '''
        throughput = len(self.vehicles)
        # TODO: check if only trach left cars
        return throughput

    def get_vehicle_lane(self):
        '''
        get_vehicle_lane
        Get current lane id and max speed of each vehicle that is running.

        :param: None
        :return vehicle_lane: current lane id of each vehicle
        :return vehicle_maxspeed: max speed of each vehicle
        '''
        # get the current lane of each vehicle. {vehicle_id: lane_id}
        vehicle_lane = {}
        for lane in self.all_lanes:
            vehicles = 	self.eng.lane.getLastStepVehicleIDs(lane)
            for vehicle in vehicles:
                vehicle_lane[vehicle] = lane
                self.vehicle_maxspeed[(vehicle,lane)] = self.eng.vehicle.getAllowedSpeed(vehicle)
        return vehicle_lane, self.vehicle_maxspeed

    def get_vehicle_trajectory(self):
        '''
        get_vehicle_trajectory
        Get trajectory of vehicles that have entered in roadnet, including vehicle_id, enter time, leave time or current time.
        
        :param: None
        :return vehicle_trajectory: trajectory of vehicles that have entered in roadnet
        :return vehicle_maxspeed: max speed of each vehicle that have entered in roadnet
        '''
        # lane_id and time spent on the corresponding lane that each vehicle went through
        vehicle_lane, self.vehicle_maxspeed = self.get_vehicle_lane() # get vehicles on tne roads except turning
        vehicles = list(self.eng.vehicle.getIDList())
        # vehicles = [x for x in vehicle_lane]
        for vehicle in vehicles:
            if vehicle not in self.vehicle_trajectory:
                self.vehicle_trajectory[vehicle] = [[vehicle_lane[vehicle], int(self.eng.simulation.getTime()), 0]]
            else:
                if vehicle not in vehicle_lane.keys(): # vehicle is turning
                    continue
                if vehicle_lane[vehicle] == self.vehicle_trajectory[vehicle][-1][0]: # vehicle is running on the same lane 
                    self.vehicle_trajectory[vehicle][-1][2] += 1
                else: # vehicle has changed the lane
                    self.vehicle_trajectory[vehicle].append(
                        [vehicle_lane[vehicle], int(self.eng.simulation.getTime()), 0])
        return self.vehicle_trajectory, self.vehicle_maxspeed

    def get_real_delay(self):
        '''
        get_real_delay
        Calculate average real delay. 
        Real delay of a vehicle is defined as the time a vehicle has traveled within the environment minus the expected travel time.
        
        :param: None
        :return avg_delay: average real delay of all vehicles
        '''
        self.vehicle_trajectory, self.vehicle_maxspeed = self.get_vehicle_trajectory()
        for v in self.vehicle_trajectory:
            # get road level routes of vehicle
            routes = self.vehicle_trajectory[v] # lane_level
            for idx, lane in enumerate(routes):
                speed = min(self.eng.lane.getMaxSpeed(lane[0]), self.vehicle_maxspeed[(v,lane[0])])
                if speed <= 1e-9 and self.crossing_proxy_ctrl is not None:
                    speed = self.crossing_proxy_ctrl.nominal_lane_speed(lane[0])
                if speed <= 1e-9:
                    speed = 11.11
                lane_length = self.eng.lane.getLength(lane[0])
                if idx == len(routes)-1: # the last lane
                    # judge whether the vehicle run over the whole lane.
                    lane_length = self.eng.vehicle.getLanePosition(v) if v in self.eng.vehicle.getIDList() else lane_length
                planned_tt = float(lane_length)/speed
                real_delay = lane[-1] - planned_tt if lane[-1]>planned_tt else 0.
                if v not in self.real_delay.keys():
                    self.real_delay[v] = real_delay
                else:
                    self.real_delay[v] += real_delay

        avg_delay = 0.
        count = 0
        for dic in self.real_delay.items():
            avg_delay += dic[1]
            count += 1
        avg_delay = avg_delay / count
        return avg_delay



