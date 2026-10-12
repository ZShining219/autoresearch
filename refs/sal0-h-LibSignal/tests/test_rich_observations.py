"""Observation semantics and config inheritance; no simulator rollout needed."""

import copy
import random
import unittest
from types import SimpleNamespace

import numpy as np

from generator.rich_lane_vehicle import RichLaneVehicleGenerator
from utils.logger import load_config
from world.world_sumo import World


class ObservationWorld:
    RIGHT = True
    obs_seed = 42
    obs_count_noise_std = 0.0
    obs_noise_mode = 'additive'
    time = 10
    _noisy_count = staticmethod(World._noisy_count)

    def __init__(self):
        self.counts = {'in_0': 2, 'in_1': 1, 'out_0': 3}

    def subscribe(self, name):
        pass

    def get_info(self, name):
        return self.counts

    def get_current_time(self):
        return self.time


class RichObservationTest(unittest.TestCase):
    def setUp(self):
        self.world = ObservationWorld()
        self.inter = SimpleNamespace(
            in_roads=['in'], out_roads=['out'],
            road_lane_mapping={'in': ['in_1', 'in_0'], 'out': ['out_0']},
            full_observation={
                'in_0': {'vehicles': [{'speed': 0.0, 'wait': 0}, {'speed': 8.0, 'wait': 20}]},
                'in_1': {'vehicles': [{'speed': 0.1, 'wait': 10}]},
                'out_0': {'vehicles': []},
            },
        )

    def test_features_use_instantaneous_speed_and_keep_lane_order(self):
        obs = RichLaneVehicleGenerator(self.world, self.inter).generate()
        # A moving vehicle with historical wait is not currently stopped.
        np.testing.assert_allclose(obs, [2, 1, 1, 0, 4, 0.1, 3])
        self.assertEqual(obs.dtype, np.float32)

    def test_empty_visible_lane_has_zero_speed_and_stopped_count(self):
        self.inter.full_observation['in_0']['vehicles'] = []
        self.world.counts['in_0'] = 0
        obs = RichLaneVehicleGenerator(self.world, self.inter).generate()
        np.testing.assert_allclose(obs, [0, 1, 0, 0, 0, 0.1, 3])

    def test_graph_padding_keeps_feature_block_offsets(self):
        obs = RichLaneVehicleGenerator(self.world, self.inter, lane_sizes=(3, 2)).generate()
        np.testing.assert_allclose(obs, [2, 1, 0, 1, 0, 0, 4, 0.1, 0, 3, 0])

    def test_lane_order_reverses_for_left_hand_traffic(self):
        self.world.RIGHT = False
        obs = RichLaneVehicleGenerator(self.world, self.inter).generate()
        np.testing.assert_allclose(obs, [1, 2, 0, 1, 0.1, 4, 3])

    def test_noise_is_repeatable_and_does_not_change_counts_speeds_or_rng(self):
        self.world.obs_count_noise_std = 5.0
        generator = RichLaneVehicleGenerator(self.world, self.inter)
        original = copy.deepcopy(self.inter.full_observation)
        counts = self.world.counts.copy()
        rng_state = random.getstate()
        first = generator.generate()
        np.testing.assert_array_equal(first, generator.generate())
        self.assertEqual(random.getstate(), rng_state)
        self.assertEqual(self.inter.full_observation, original)
        self.assertEqual(self.world.counts, counts)
        np.testing.assert_allclose(first[[0, 1, 4, 5, 6]], [2, 1, 4, 0.1, 3])
        readings = set()
        for time in range(20):
            self.world.time = time
            obs = generator.generate()
            self.assertTrue(np.all(obs[2:4] >= 0))
            readings.add(tuple(obs[2:4]))
        self.assertGreater(len(readings), 1)

    def test_proportional_noise_does_not_invent_stops_in_empty_lane(self):
        self.world.obs_count_noise_std = 2.0
        self.world.obs_noise_mode = 'proportional'
        self.inter.full_observation['in_0']['vehicles'] = []
        obs = RichLaneVehicleGenerator(self.world, self.inter).generate()
        self.assertEqual(obs[2], 0)
        self.assertEqual(obs[3], 0)


class RichConfigTest(unittest.TestCase):
    def test_variants_inherit_native_experiment_settings(self):
        for name in ('dqn', 'presslight', 'colight'):
            for suffix in ('', '_odh_l1', '_odh_l2', '_odh_l1_1x21', '_odh_l2_1x21'):
                with self.subTest(agent=name, suffix=suffix):
                    native, _ = load_config(f'configs/tsc/{name}{suffix}.yml')
                    rich, _ = load_config(f'configs/tsc/{name}_rich{suffix}.yml')
                    self.assertEqual(rich['model']['name'], f'{name}_rich')
                    rich['model']['name'] = name
                    self.assertEqual(native, rich)


if __name__ == '__main__':
    unittest.main()
