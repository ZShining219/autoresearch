import torch
from common.registry import Registry


@Registry.register_model('base')
class BaseAgent(object):
    '''
    BaseAgent Class is mainly used for creating a base agent and base methods.
    '''
    def __init__(self, world):
        # revise if it is multi-agents in one model
        self.world = world
        self.sub_agents = 1
        self.device = torch.device('cpu')

    def to_device(self, device):
        '''
        to_device
        Move agent's models to the specified device (CPU/GPU).

        :param device: torch.device to move models to
        :return: None
        '''
        self.device = device

    def get_ob(self):
        raise NotImplementedError()

    def get_reward(self):
        raise NotImplementedError()

    def get_action(self, ob, phase):
        raise NotImplementedError()

    def get_action_prob(self, ob, phase):
        return None
