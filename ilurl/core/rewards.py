import inspect
from sys import modules

import numpy as np
from ilurl.core.meta import MetaReward


def get_rewards():
    this = modules[__name__]
    names, objects = [], []
    for name, obj in inspect.getmembers(this):

        # Is a definition a class?
        if inspect.isclass(obj):
            # Is defined in this module
            if inspect.getmodule(obj) == this:
                names.append(name)
                objects.append(obj)

    return tuple(names), tuple(objects)
        
    

def build_rewards(mdp_params):
    """Builder that defines all rewards
    """
    return MaxSpeedCountReward(mdp_params)

class MaxSpeedCountReward(object, metaclass=MetaReward):

    def __init__(self,  mdp_params):
        """Creates a reference to the input state"""
        if not hasattr(mdp_params, 'target_velocity'):
            raise ValueError('MDPParams must define target_velocity')
        else:
            self.target_velocity = mdp_params.target_velocity

    def calculate(self, state):
        speeds_counts = state.split()

        ret = {}
        for k, v in speeds_counts.items():
            speeds, counts = v

            if sum(counts) <= 0:
                reward = 0
            else:
                max_cost = \
                    np.array([self.target_velocity] * len(speeds))

                reward = \
                    -np.maximum(max_cost - speeds, 0).dot(counts)

            ret[k] = reward
        return ret

# TODO: implement Rewards definition 1
class MinDelayReward(object, metaclass=MetaReward):
    def __init__(self, state):
        """Creates a reference to the input state"""
        if state.__class__ != 'Delay':
            raise ValueError(
                'MinDelay reward expects `Delay` state')
        else:
            self._state = state

    # TODO: make state sumable
    def calculate(self):
        return -sum(self._state.to_list())


# TODO: implement Rewards definition 2
class MinDeltaDelayReward(object, metaclass=MetaReward):
    def __init__(self, state):
        """Creates a reference to the input state"""
        if state.__class__ != 'DeltaDelay':
            raise ValueError(
                'MinDeltaDelay reward expects `DeltaDelay` state')

    def calculate(self):
        return -sum(self._state.to_list())
