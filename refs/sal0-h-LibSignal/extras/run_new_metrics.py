#!/usr/bin/env python3
"""
Standalone exporter: run a TSC agent and write new_metrics to extras/output/.

Prefer the main path for research runs — `python run.py ...` now writes the same
metrics under the run's logger/ directory on final test (see configs/tsc/base.yml
`trainer.save_trip_metrics`). This script remains for ad-hoc exports to extras/output/.

Usage (from repo root):
  python extras/run_new_metrics.py --agent maxpressure --network sumo4x4 --seed 42
  python extras/run_new_metrics.py --agent dqn --network sumo4x4 --seed 42 --train
"""
import os
import sys

# Match run.py reproducibility (must run before other imports).
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import logging

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

_saved_argv = sys.argv[:]
sys.argv = [_saved_argv[0]]
from run import Runner  # noqa: E402
sys.argv = _saved_argv

import task  # noqa: F401,E402
import trainer  # noqa: F401,E402
import agent  # noqa: F401,E402
import dataset  # noqa: F401,E402

from common.registry import Registry
from utils.logger import setup_logging
from utils.trip_metrics import TripMetricsTracker, write_trip_metrics

EXTRAS_DIR = os.path.join(PROJECT_ROOT, "extras")
DEFAULT_OUTPUT_ROOT = os.path.join(EXTRAS_DIR, "output")


def resolve_output_dir(agent, network, seed, test_steps, run_name=None, output_root=None):
    """extras/output/<agent>/<network>/<run_name>/"""
    root = output_root or DEFAULT_OUTPUT_ROOT
    folder = run_name or f"seed{seed}_steps{test_steps}"
    return os.path.join(root, agent, network, folder)


def _default_interface(requested):
    if requested != "libsumo":
        return requested
    try:
        import libsumo  # noqa: F401
    except ImportError:
        print("[Info] libsumo not available — using traci instead.")
        return "traci"
    return "libsumo"


def run_with_new_metrics(runner, output_dir):
    """Greedy test rollout + per-vehicle trip metrics (shared with TSCTrainer)."""
    trainer_obj = runner.trainer
    env = trainer_obj.env
    world = trainer_obj.world

    test_steps = Registry.mapping["trainer_mapping"]["setting"].param["test_steps"]
    action_interval = Registry.mapping["trainer_mapping"]["setting"].param["action_interval"]
    cmd = Registry.mapping["command_mapping"]["setting"].param

    trainer_obj.metric.clear()
    obs = env.reset()
    for ag in trainer_obj.agents:
        ag.reset()

    tracker = TripMetricsTracker(world)
    dones = [False]
    i = 0
    while i < test_steps:
        if i % action_interval == 0:
            phases = np.stack([ag.get_phase() for ag in trainer_obj.agents])
            if hasattr(trainer_obj, "_collect_actions"):
                actions = trainer_obj._collect_actions(obs, phases, test=True)
            else:
                actions = np.stack(
                    [
                        ag.get_action(obs[idx], phases[idx], test=True)
                        for idx, ag in enumerate(trainer_obj.agents)
                    ]
                )
            rewards_list = []
            for t in range(action_interval):
                tracker.before_step()
                obs, rewards, dones, _ = env.step(
                    actions.flatten(),
                    collect_obs=(t == action_interval - 1),
                )
                tracker.after_step()
                i += 1
                rewards_list.append(np.stack(rewards))
                if i >= test_steps:
                    break
            if rewards_list:
                trainer_obj.metric.update(np.mean(rewards_list, axis=0))
        if all(dones):
            break

    vehicle_records = tracker.finalize()
    meta_extra = {
        "agent": cmd.get("agent"),
        "network": cmd.get("network"),
        "world": cmd.get("world"),
        "prefix": cmd.get("prefix"),
        "seed": cmd.get("seed"),
        "avg_travel_time_metric": float(trainer_obj.metric.real_average_travel_time()),
        "throughput_metric": int(trainer_obj.metric.throughput()),
        "notes": (
            "Standalone extras/output export. Main runs also write these files under "
            "data/output_data/.../logger/ via run.py (trainer.save_trip_metrics)."
        ),
    }
    csv_path, meta_path, meta = write_trip_metrics(
        vehicle_records, output_dir, meta_extra=meta_extra, stem="new_metrics"
    )
    trainer_obj.logger.info(
        "Saved %d vehicle records to %s (ATT=%.2fs, completion=%.1f%%)",
        len(vehicle_records),
        csv_path,
        meta.get("mean_travel_time_s") or 0.0,
        (meta.get("completion_rate") or 0.0) * 100.0,
    )
    print(f"\nnew_metrics CSV: {csv_path}")
    print(f"Metadata:                 {meta_path}")
    if meta.get("travel_time_stats_completed"):
        ts = meta["travel_time_stats_completed"]
        print(
            f"travel_time_s (completed): mean={ts['mean']:.1f}s median={ts['median']:.1f}s "
            f"p95={ts['p95']:.1f}s"
        )
    if meta.get("mean_schedule_efficiency_completed") is not None:
        print(f"mean schedule_efficiency (completed): {meta['mean_schedule_efficiency_completed']:.4f}")
    if meta.get("system_idle_share_completed") is not None:
        print(f"system_idle_share (completed): {meta['system_idle_share_completed']:.4f}")
    print(f"throughput (completed): {meta.get('throughput_completed')}")
    print(f"completion_rate: {meta.get('completion_rate')}")
    return csv_path, meta_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a TSC agent on SUMO and export per-vehicle trip metrics (new_metrics CSV)"
    )
    parser.add_argument(
        "--agent", "-a", type=str, default="maxpressure",
        help="Agent name (e.g. maxpressure, fixedtime, sotl)",
    )
    parser.add_argument("--network", "-n", type=str, default="sumo1x1")
    parser.add_argument(
        "--prefix", type=str, default="analysis",
        help="LibSignal run prefix (for data/output_data only; analysis CSV uses extras/output/)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Subfolder under extras/output/<agent>/<network>/ (default: seed<N>_steps<T>)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Override output root (default: extras/output)",
    )
    parser.add_argument(
        "--interface", type=str, default="libsumo", choices=["libsumo", "traci"]
    )
    parser.add_argument(
        "--test-steps", type=int, default=None, help="Override trainer.test_steps"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train RL agent first (e.g. dqn), then export trip metrics on greedy test rollout",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override trainer.episodes when --train is set",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    interface = _default_interface(args.interface)

    run_args = argparse.Namespace(
        thread_num=4,
        ngpu="-1",
        prefix=args.prefix,
        seed=args.seed,
        debug=False,
        interface=interface,
        delay_type="apx",
        task="tsc",
        agent=args.agent,
        world="sumo",
        network=args.network,
        dataset="onfly",
        no_trip_metrics=True,  # avoid duplicate write under data/output_data from trainer.test
    )

    runner = Runner(run_args)
    test_steps = args.test_steps
    if test_steps is not None:
        Registry.mapping["trainer_mapping"]["setting"].param["test_steps"] = test_steps
        Registry.mapping["trainer_mapping"]["setting"].param["steps"] = test_steps
    else:
        test_steps = Registry.mapping["trainer_mapping"]["setting"].param["test_steps"]
    if args.episodes is not None:
        Registry.mapping["trainer_mapping"]["setting"].param["episodes"] = args.episodes

    output_dir = resolve_output_dir(
        agent=args.agent,
        network=args.network,
        seed=args.seed,
        test_steps=test_steps,
        run_name=args.run_name,
        output_root=args.output_root,
    )

    logger = setup_logging(logging.INFO)
    runner.trainer = Registry.mapping["trainer_mapping"][
        Registry.mapping["command_mapping"]["setting"].param["task"]
    ](logger)

    if args.train:
        train_model = Registry.mapping["model_mapping"]["setting"].param.get("train_model", False)
        if not train_model:
            print(f"[Warning] --train set but agent '{args.agent}' has train_model=False; exporting only.")
        else:
            print(f"Training {args.agent} for {Registry.mapping['trainer_mapping']['setting'].param['episodes']} episodes...")
            runner.trainer.train()
            for ag in runner.trainer.agents:
                ag.load_model(runner.trainer.episodes)
            print("Training done; running trip-metrics export rollout...")

    return run_with_new_metrics(runner, output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
