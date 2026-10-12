from common.registry import Registry
from generator.rich_lane_vehicle import RichLaneVehicleGenerator

from .colight import CoLightAgent


@Registry.register_model('colight_rich')
class CoLightRichAgent(CoLightAgent):
    '''CoLight with the richer observation at each node of the existing graph.'''

    def __init__(self, world, rank):
        # Pad each feature block separately so its offsets agree across graph nodes.
        self.lane_sizes = tuple(
            max(sum(len(inter.road_lane_mapping[road]) for road in getattr(inter, roads))
                for inter in world.intersections)
            for roads in ('in_roads', 'out_roads')
        )
        super().__init__(world, rank)

    def _make_observation_generator(self, inter):
        return RichLaneVehicleGenerator(self.world, inter, lane_sizes=self.lane_sizes)
