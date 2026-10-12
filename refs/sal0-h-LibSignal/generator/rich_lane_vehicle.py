import hashlib
import random

import numpy as np

from .base import BaseGenerator


class RichLaneVehicleGenerator(BaseGenerator):
    '''SUMO lane counts, instantaneous stopped counts, speeds, and downstream counts.'''

    def __init__(self, world, I, lane_sizes=None):
        self.world = world
        self.I = I
        self.in_lanes = self._lanes(I.in_roads)
        self.out_lanes = self._lanes(I.out_roads)
        self.in_size, self.out_size = lane_sizes or (len(self.in_lanes), len(self.out_lanes))
        self.ob_length = 3 * self.in_size + self.out_size
        self.world.subscribe('lane_count')

    def _lanes(self, roads):
        lanes = []
        for road in roads:
            lanes.extend(sorted(
                self.I.road_lane_mapping[road],
                key=lambda lane: int(lane.rsplit('_', 1)[-1]),
                reverse=not self.world.RIGHT,
            ))
        return lanes

    def generate(self):
        counts = self.world.get_info('lane_count')
        obs = np.zeros(self.ob_length, dtype=np.float32)
        sigma = self.world.obs_count_noise_std
        proportional = self.world.obs_noise_mode == 'proportional'
        now = int(self.world.get_current_time()) if sigma > 0 else 0

        for i, lane in enumerate(self.in_lanes):
            vehicles = self.I.full_observation[lane]['vehicles']
            stopped = sum(vehicle['speed'] < 0.1 for vehicle in vehicles)
            if sigma > 0:
                # A separate lane/time stream leaves existing count and reward noise intact.
                key = f'rich_stopped:{self.world.obs_seed}:{now}:{lane}'
                seed = int.from_bytes(hashlib.md5(key.encode()).digest()[:8], 'big')
                stopped = self.world._noisy_count(stopped, sigma, proportional, random.Random(seed))
            obs[i] = counts[lane]
            obs[self.in_size + i] = stopped
            if vehicles:
                obs[2 * self.in_size + i] = sum(v['speed'] for v in vehicles) / len(vehicles)

        for i, lane in enumerate(self.out_lanes):
            obs[3 * self.in_size + i] = counts[lane]
        return obs
