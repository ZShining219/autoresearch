from common.registry import Registry
from generator.rich_lane_vehicle import RichLaneVehicleGenerator

from .presslight import PressLightAgent


@Registry.register_model('presslight_rich')
class PressLightRichAgent(PressLightAgent):
    '''PressLight with stopped counts and speeds in addition to in/out counts.'''

    def _make_observation_generator(self, inter_obj):
        return RichLaneVehicleGenerator(self.world, inter_obj)
