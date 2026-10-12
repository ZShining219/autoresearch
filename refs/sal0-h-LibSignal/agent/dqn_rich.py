from common.registry import Registry
from generator.rich_lane_vehicle import RichLaneVehicleGenerator

from .dqn import DQNAgent


@Registry.register_model('dqn_rich')
class DQNRichAgent(DQNAgent):
    '''DQN with richer local traffic observations and the original reward/actions.'''

    def _make_observation_generator(self, inter_obj):
        return RichLaneVehicleGenerator(self.world, inter_obj)
