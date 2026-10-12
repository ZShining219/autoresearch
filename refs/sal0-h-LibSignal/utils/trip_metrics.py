"""
Per-vehicle trip metrics for SUMO TSC runs.

Used by TSCTrainer (via run.py) and extras/run_new_metrics.py.

Primary ranking fields for journals:
  - travel_time_s mean / median / p95 (completed trips)
  - throughput (completed count; also in LibSignal Metrics)
  - waiting_fraction / system_idle_share  (= idle / travel; ride effectiveness)
  - schedule_efficiency (free_flow / travel)
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime

import numpy as np

METRICS_CSV_NAME = "new_metrics.csv"
METRICS_META_NAME = "new_metrics_meta.json"
STOP_SPEED_THRESHOLD = 0.1

CSV_FIELDS = [
    "vehicle_id",
    "vehicle_type",
    "enter_time_s",
    "departure_time_s",
    "travel_time_s",
    "route_distance_m",
    "distance_traveled_m",
    "accumulated_waiting_time_s",
    "custom_wait_s",
    "moving_time_s",
    "time_loss_s",
    "free_flow_time_s",
    "delay_s",
    "waiting_fraction",
    "moving_fraction",
    "schedule_efficiency",
    "completed_trip",
    "trip_status",
]


def _simulation_id_list(world, method_name):
    method = getattr(world.eng.simulation, method_name, None)
    if method is None:
        return []
    try:
        return list(method())
    except Exception:
        return []


def _vehicle_route(world, veh_id):
    try:
        return list(world.eng.vehicle.getRoute(veh_id))
    except Exception:
        return None


def _route_distance_m(world, veh_id):
    route = _vehicle_route(world, veh_id)
    if not route:
        return None
    try:
        return float(sum(world.eng.edge.getLength(edge_id) for edge_id in route))
    except Exception:
        return None


def _free_flow_route_time(world, veh_id):
    route = _vehicle_route(world, veh_id)
    if not route:
        return None
    try:
        total = 0.0
        for edge_id in route:
            length = world.eng.edge.getLength(edge_id)
            lane_id = f"{edge_id}_0"
            speed = world.eng.lane.getMaxSpeed(lane_id)
            if speed > 0:
                total += length / speed
        return total
    except Exception:
        return None


def _distance_traveled_m(world, veh_id):
    try:
        return float(world.eng.vehicle.getDistance(veh_id))
    except Exception:
        return None


def _vehicle_type(world, veh_id):
    try:
        return str(world.eng.vehicle.getTypeID(veh_id))
    except Exception:
        return ""


def _vehicle_enter_time(world, veh_id, sim_time):
    return float(world.inside_vehicles.get(veh_id, sim_time))


def _total_metric(per_step, cumulative, veh_id):
    return float(per_step.get(veh_id, 0.0)) + float(cumulative.get(veh_id, 0.0))


def _make_record(
    veh_id,
    enter_time,
    departure_time,
    travel_time,
    sumo_wait,
    custom_wait,
    time_loss,
    free_flow_time,
    trip_status,
    vehicle_type="",
    route_distance_m=None,
    distance_traveled_m=None,
):
    delay = None
    schedule_efficiency = None
    moving_time = max(0.0, travel_time - custom_wait) if travel_time > 0 else 0.0
    waiting_fraction = custom_wait / travel_time if travel_time > 0 else 0.0
    moving_fraction = 1.0 - waiting_fraction if travel_time > 0 else 0.0
    if free_flow_time is not None and travel_time > 0:
        delay = max(0.0, travel_time - free_flow_time)
        schedule_efficiency = min(1.0, max(0.0, free_flow_time / travel_time))
    return {
        "vehicle_id": veh_id,
        "vehicle_type": vehicle_type,
        "enter_time_s": round(enter_time, 3),
        "departure_time_s": round(departure_time, 3) if departure_time is not None else None,
        "travel_time_s": round(travel_time, 3),
        "route_distance_m": round(route_distance_m, 3) if route_distance_m is not None else "",
        "distance_traveled_m": round(distance_traveled_m, 3) if distance_traveled_m is not None else "",
        "accumulated_waiting_time_s": round(sumo_wait, 3),
        "custom_wait_s": round(custom_wait, 3),
        "moving_time_s": round(moving_time, 3),
        "time_loss_s": round(time_loss, 3),
        "free_flow_time_s": round(free_flow_time, 3) if free_flow_time is not None else "",
        "delay_s": round(delay, 3) if delay is not None else "",
        "waiting_fraction": round(waiting_fraction, 4),
        "moving_fraction": round(moving_fraction, 4),
        "schedule_efficiency": round(schedule_efficiency, 4) if schedule_efficiency is not None else "",
        "completed_trip": trip_status == "completed",
        "trip_status": trip_status,
    }


def system_idle_share(records):
    """Mean ride-effectiveness inverse: sum(idle) / sum(travel). Lower is better."""
    total_time = sum(float(r["travel_time_s"]) for r in records if r.get("travel_time_s"))
    total_idle = sum(float(r["custom_wait_s"]) for r in records if r.get("custom_wait_s") is not None)
    if total_time <= 0:
        return None
    return float(total_idle / total_time)


def metric_stats(records, field):
    values = [r[field] for r in records if r.get(field) not in (None, "")]
    if not values:
        return None
    values = [float(v) for v in values]
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


class TripMetricsTracker:
    """Accumulate per-vehicle trip metrics across one SUMO eval episode."""

    def __init__(self, world):
        self.world = world
        self.dt = float(getattr(world, "step_length", 1.0))
        self.reset()

    def reset(self):
        self.custom_wait = {}
        self.cumulative_sumo_wait = {}
        self.cumulative_time_loss = {}
        self.free_flow_times = {}
        self.route_distances = {}
        self.vehicle_types = {}
        self.seen_logged = set()
        self.vehicle_records = []
        self._pre_sumo_wait = {}
        self._pre_time_loss = {}
        self._pre_speed = {}
        self._pre_distance = {}

    def before_step(self):
        world = self.world
        for veh_id in world.eng.vehicle.getIDList():
            fft = _free_flow_route_time(world, veh_id)
            if fft is not None:
                self.free_flow_times[veh_id] = fft
            rd = _route_distance_m(world, veh_id)
            if rd is not None:
                self.route_distances[veh_id] = rd
            vtype = _vehicle_type(world, veh_id)
            if vtype:
                self.vehicle_types[veh_id] = vtype

        self._pre_sumo_wait = {}
        self._pre_time_loss = {}
        self._pre_speed = {}
        self._pre_distance = {}
        for veh_id in world.eng.vehicle.getIDList():
            self._pre_sumo_wait[veh_id] = float(world.eng.vehicle.getAccumulatedWaitingTime(veh_id))
            try:
                self._pre_time_loss[veh_id] = float(world.eng.vehicle.getTimeLoss(veh_id))
            except Exception:
                self._pre_time_loss[veh_id] = 0.0
            self._pre_speed[veh_id] = float(world.eng.vehicle.getSpeed(veh_id))
            dist = _distance_traveled_m(world, veh_id)
            if dist is not None:
                self._pre_distance[veh_id] = dist

        for veh_id, speed in self._pre_speed.items():
            if speed < STOP_SPEED_THRESHOLD:
                self.custom_wait[veh_id] = self.custom_wait.get(veh_id, 0.0) + self.dt

    def after_step(self):
        world = self.world
        sim_time = world.eng.simulation.getTime()
        pre_sumo_wait = self._pre_sumo_wait
        pre_time_loss = self._pre_time_loss
        pre_distance = self._pre_distance

        newly_completed = set(world.vehicles.keys()) - self.seen_logged
        for veh_id in sorted(newly_completed):
            travel_time = float(world.vehicles[veh_id])
            sumo_wait = _total_metric(pre_sumo_wait, self.cumulative_sumo_wait, veh_id)
            time_loss = _total_metric(pre_time_loss, self.cumulative_time_loss, veh_id)
            wait_custom = float(self.custom_wait.pop(veh_id, 0.0))
            self.cumulative_sumo_wait.pop(veh_id, None)
            self.cumulative_time_loss.pop(veh_id, None)
            departure_time = sim_time
            enter_time = departure_time - travel_time
            self.vehicle_records.append(
                _make_record(
                    veh_id,
                    enter_time,
                    departure_time,
                    travel_time,
                    sumo_wait,
                    wait_custom,
                    time_loss,
                    self.free_flow_times.pop(veh_id, None),
                    "completed",
                    vehicle_type=self.vehicle_types.pop(veh_id, ""),
                    route_distance_m=self.route_distances.pop(veh_id, None),
                    distance_traveled_m=pre_distance.get(veh_id),
                )
            )
        self.seen_logged |= newly_completed

        for veh_id in _simulation_id_list(world, "getTeleportingIDList"):
            self.cumulative_sumo_wait[veh_id] = self.cumulative_sumo_wait.get(veh_id, 0.0) + float(
                pre_sumo_wait.get(veh_id, 0.0)
            )
            self.cumulative_time_loss[veh_id] = self.cumulative_time_loss.get(veh_id, 0.0) + float(
                pre_time_loss.get(veh_id, 0.0)
            )

        removed = set(_simulation_id_list(world, "getRemovedIDList")) - self.seen_logged
        for veh_id in sorted(removed):
            sumo_wait = _total_metric(pre_sumo_wait, self.cumulative_sumo_wait, veh_id)
            time_loss = _total_metric(pre_time_loss, self.cumulative_time_loss, veh_id)
            wait_custom = float(self.custom_wait.pop(veh_id, 0.0))
            self.cumulative_sumo_wait.pop(veh_id, None)
            self.cumulative_time_loss.pop(veh_id, None)
            enter_time = _vehicle_enter_time(world, veh_id, sim_time)
            travel_time = sim_time - enter_time
            self.vehicle_records.append(
                _make_record(
                    veh_id,
                    enter_time,
                    sim_time,
                    travel_time,
                    sumo_wait,
                    wait_custom,
                    time_loss,
                    self.free_flow_times.pop(veh_id, None),
                    "removed",
                    vehicle_type=self.vehicle_types.pop(veh_id, ""),
                    route_distance_m=self.route_distances.pop(veh_id, None),
                    distance_traveled_m=pre_distance.get(veh_id),
                )
            )
            self.seen_logged.add(veh_id)

    def finalize(self):
        world = self.world
        sim_time = world.eng.simulation.getTime()
        for veh_id in world.eng.vehicle.getIDList():
            if veh_id in self.seen_logged:
                continue
            sumo_wait = float(world.eng.vehicle.getAccumulatedWaitingTime(veh_id))
            sumo_wait += float(self.cumulative_sumo_wait.get(veh_id, 0.0))
            try:
                time_loss = float(world.eng.vehicle.getTimeLoss(veh_id))
            except Exception:
                time_loss = 0.0
            time_loss += float(self.cumulative_time_loss.get(veh_id, 0.0))
            wait_custom = float(self.custom_wait.get(veh_id, 0.0))
            enter_time = _vehicle_enter_time(world, veh_id, sim_time)
            travel_time = sim_time - enter_time
            self.vehicle_records.append(
                _make_record(
                    veh_id,
                    enter_time,
                    None,
                    travel_time,
                    sumo_wait,
                    wait_custom,
                    time_loss,
                    self.free_flow_times.get(veh_id),
                    "on_map_at_end",
                    vehicle_type=self.vehicle_types.get(veh_id, _vehicle_type(world, veh_id)),
                    route_distance_m=self.route_distances.get(veh_id),
                    distance_traveled_m=_distance_traveled_m(world, veh_id),
                )
            )
        return list(self.vehicle_records)


def write_trip_metrics(records, output_dir, meta_extra=None, stem=None):
    """
    Write CSV + JSON under output_dir.

    stem: optional filename stem (default 'new_metrics').
          Use e.g. 'new_metrics_hold_00' for held-out demand files.
    """
    os.makedirs(output_dir, exist_ok=True)
    stem = stem or "new_metrics"
    csv_path = os.path.join(output_dir, f"{stem}.csv")
    meta_path = os.path.join(output_dir, f"{stem}_meta.json")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in CSV_FIELDS})

    completed = [r for r in records if r.get("trip_status") == "completed"]
    incomplete = [r for r in records if r.get("trip_status") != "completed"]
    status_counts = {}
    for rec in records:
        status = rec.get("trip_status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    n_departed = len(records)
    completion_rate = len(completed) / n_departed if n_departed else None

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_vehicles_logged": len(records),
        "n_completed_trips": len(completed),
        "n_incomplete_trips": len(incomplete),
        "completion_rate": completion_rate,
        "trip_status_counts": status_counts,
        "stop_speed_threshold_mps": STOP_SPEED_THRESHOLD,
        "primary_efficiency_metric": "schedule_efficiency",
        "ride_effectiveness_note": (
            "waiting_fraction = custom_wait_s / travel_time_s per vehicle; "
            "system_idle_share = sum(custom_wait_s)/sum(travel_time_s). "
            "Lower idle share = higher ride effectiveness."
        ),
        "metric_definitions": {
            "travel_time_s": "Wall-clock time in network (LibSignal ATT = mean over completed).",
            "route_distance_m": "Sum of edge lengths on assigned route.",
            "distance_traveled_m": "SUMO odometer at trip end (or horizon for censored).",
            "custom_wait_s": f"Uncapped idle time (speed < {STOP_SPEED_THRESHOLD} m/s each step).",
            "waiting_fraction": "custom_wait_s / travel_time_s (per-vehicle idle share).",
            "moving_fraction": "1 - waiting_fraction.",
            "schedule_efficiency": "free_flow_time_s / travel_time_s in [0,1] (higher is better).",
            "system_idle_share": "sum(custom_wait_s) / sum(travel_time_s) over a record set.",
            "throughput": "Number of completed trips in the eval window.",
        },
        "sumo_accumulated_wait_note": (
            "accumulated_waiting_time_s uses SUMO getAccumulatedWaitingTime(), capped by "
            "default --waiting-time-memory=100s. Prefer custom_wait_s for tails/fairness."
        ),
        "travel_time_stats_completed": metric_stats(completed, "travel_time_s"),
        "travel_time_stats_incomplete": metric_stats(incomplete, "travel_time_s"),
        "route_distance_stats_completed": metric_stats(completed, "route_distance_m"),
        "distance_traveled_stats_completed": metric_stats(completed, "distance_traveled_m"),
        "schedule_efficiency_stats_completed": metric_stats(completed, "schedule_efficiency"),
        "moving_fraction_stats_completed": metric_stats(completed, "moving_fraction"),
        "waiting_fraction_stats_completed": metric_stats(completed, "waiting_fraction"),
        "custom_wait_stats_completed": metric_stats(completed, "custom_wait_s"),
        "custom_wait_stats_incomplete": metric_stats(incomplete, "custom_wait_s"),
        "custom_wait_stats_all": metric_stats(records, "custom_wait_s"),
        "delay_stats_completed": metric_stats(completed, "delay_s"),
        "time_loss_stats_all": metric_stats(records, "time_loss_s"),
        "system_idle_share_completed": system_idle_share(completed),
        "system_idle_share_all": system_idle_share(records),
        "mean_schedule_efficiency_completed": (
            float(np.mean([float(r["schedule_efficiency"]) for r in completed if r.get("schedule_efficiency") not in ("", None)]))
            if any(r.get("schedule_efficiency") not in ("", None) for r in completed)
            else None
        ),
        "mean_travel_time_s": float(np.mean([r["travel_time_s"] for r in completed])) if completed else None,
        "median_travel_time_s": float(np.median([r["travel_time_s"] for r in completed])) if completed else None,
        "p95_travel_time_s": float(np.percentile([r["travel_time_s"] for r in completed], 95)) if completed else None,
        "throughput_completed": len(completed),
        "vehicle_type_counts": {
            t: sum(1 for r in records if r.get("vehicle_type") == t)
            for t in sorted({r.get("vehicle_type", "") for r in records})
        },
        "csv_path": csv_path,
        "output_dir": output_dir,
    }
    if meta_extra:
        meta.update(meta_extra)

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return csv_path, meta_path, meta
