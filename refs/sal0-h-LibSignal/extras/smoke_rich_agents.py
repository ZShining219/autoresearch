#!/usr/bin/env python3
"""300-second CPU smoke check, three gradient updates, and checkpoint round trip.

Run from the repository root, for example:
  python extras/smoke_rich_agents.py --agent dqn_rich --network sumo4x4 --ngpu -1
"""

import os
import sys

if os.environ.get('PYTHONHASHSEED') != '0':
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import json
import logging
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# run.py parses argv when imported; follow extras/run_new_metrics.py.
_saved_argv = sys.argv[:]
sys.argv = [_saved_argv[0]]
from run import Runner, parser as run_parser
sys.argv = _saved_argv

from common.registry import Registry
from generator.rich_lane_vehicle import RichLaneVehicleGenerator
from utils.logger import setup_logging


def smoke(args):
    suffix = '' if args.network == 'sumo4x4' else '_1x21'
    command = run_parser.parse_args([
        '--agent', f'{args.agent}_odh_l2{suffix}', '--network', args.network,
        '--world', 'sumo', '--seed', '42', '--ngpu', '-1', '--prefix', 'rich_smoke',
    ])
    runner = Runner(command)
    # Test-only batch size; production configs keep the native hyperparameters.
    runner.config['model']['batch_size'] = 8
    runner.config['trainer'].update(episodes=1, steps=300, test_steps=300)
    torch.set_num_threads(1)
    logger = setup_logging(logging.INFO)
    trainer = Registry.mapping['trainer_mapping']['tsc'](logger)
    world = trainer.world
    agents = trainer.agents
    started = time.monotonic()
    losses = []
    peak_count = 0
    peak_visible_vehicles = 0
    try:
        trainer._select_train_demand(0)
        trainer.env.reset()
        for ag in agents:
            ag.reset()
        obs = [ag.get_ob() for ag in agents]
        phases = np.stack([ag.get_phase() for ag in agents])
        action_limits = np.array([len(inter.phases) for inter in world.intersections])
        graph_order_matches = None
        if args.agent == 'colight_rich':
            ag = agents[0]
            gen_ids = [gen.I.id for _, gen in ag.ob_generator]
            world_ids = [inter.id for inter in world.intersections]
            graph_order_matches = gen_ids == world_ids
            edges = ag.edge_idx.detach().cpu()
            n_world = len(world_ids)
            edges_in_world = (
                edges.numel() == 0
                or (int(edges.min()) >= 0 and int(edges.max()) < n_world)
            )
            if not graph_order_matches or not edges_in_world:
                raise AssertionError(
                    'CoLight generators/edges are not in world.intersections order '
                    f'(gens_match={graph_order_matches}, edges_in_world={edges_in_world})'
                )

        for decision in range(30):
            for ag, observation in zip(agents, obs):
                assert np.isfinite(observation).all()
                expected = ag.ob_length - (ag.action_space.n if ag.phase else 0)
                assert observation.shape == (ag.sub_agents, expected)
            actions = trainer._collect_actions(obs, phases, test=True)
            assert np.all((0 <= actions.flatten()) & (actions.flatten() < action_limits))
            rewards_list = []
            for step in range(10):
                next_obs, rewards, dones, _ = trainer.env.step(actions.flatten(), collect_obs=(step == 9))
                rewards_list.append(np.stack(rewards))
            rewards = np.mean(rewards_list, axis=0)
            assert np.isfinite(rewards).all()
            next_phases = np.stack([ag.get_phase() for ag in agents])
            for idx, ag in enumerate(agents):
                ag.remember(obs[idx], phases[idx], actions[idx], None, rewards[idx],
                            next_obs[idx], next_phases[idx], dones[idx], f'smoke_{decision}_{idx}')
            peak_count = max(peak_count, sum(world.get_info('lane_count').values()))
            peak_visible_vehicles = max(peak_visible_vehicles, sum(
                len(inter.full_observation[lane]['vehicles'])
                for inter in world.intersections for road in inter.in_roads
                for lane in inter.road_lane_mapping[road]
            ))
            obs, phases = next_obs, next_phases

        assert peak_visible_vehicles > 0, 'Smoke test must observe traffic'
        for ag in agents:
            before = [p.detach().clone() for p in ag.model.parameters()]
            for _ in range(3):
                loss = float(ag.train())
                assert np.isfinite(loss)
                losses.append(loss)
            assert any(not torch.equal(a, b) for a, b in zip(before, ag.model.parameters()))

        for idx, ag in enumerate(agents):
            ag.update_target_network()
            ag.save_model(0)
            expected = {k: v.detach().clone() for k, v in ag.model.state_dict().items()}
            expected_action = ag.get_action(obs[idx], phases[idx], test=True)
            with torch.no_grad():
                for param in ag.model.parameters():
                    param.zero_()
            ag.load_model(0)
            for key, value in ag.model.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
            np.testing.assert_array_equal(ag.get_action(obs[idx], phases[idx], test=True), expected_action)

        trainer.env.reset()
        for ag in agents:
            ag.reset()
            generators = [g for _, g in ag.ob_generator] if ag.sub_agents > 1 else [ag.ob_generator]
            for generator in generators:
                assert isinstance(generator, RichLaneVehicleGenerator)
                assert generator.I is world.id2intersection[generator.I.id]
            assert np.isfinite(ag.get_ob()).all()
        result = {
            'agent': args.agent, 'network': args.network, 'profile': 'L2', 'seed': 42,
            'simulated_seconds': 300, 'updates_per_agent': 3,
            'controlled_signals': len(action_limits),
            'model_input_sizes': sorted({ag.ob_length for ag in agents}),
            'mean_loss': float(np.mean(losses)), 'peak_observed_count': peak_count,
            'peak_visible_vehicles': peak_visible_vehicles,
            'graph_order_matches_world': graph_order_matches,
            'checkpoint_reload': 'passed', 'reset': 'passed',
            'wall_seconds': round(time.monotonic() - started, 2),
        }
        output = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'smoke_result.json')
        with open(output, 'w') as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps(result))
    finally:
        world.eng.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', required=True, choices=['dqn_rich', 'presslight_rich', 'colight_rich'])
    parser.add_argument('--network', required=True, choices=['sumo4x4', 'sumo1x21'])
    parser.add_argument('--ngpu', default='-1', choices=['-1'])
    smoke(parser.parse_args())
