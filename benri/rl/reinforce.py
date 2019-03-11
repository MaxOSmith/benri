""" REINFOCE learning algorithm. """
import torch

from benri.gym.action import Action, SequenceAction


def _fix_nan(x):
    x[torch.isnan(x)] = 0
    return x


def reinforce_loss(actions: dict):

    for action in actions.values():
        
        a = action.value
        p_s = None

        raise NotImplementedError()