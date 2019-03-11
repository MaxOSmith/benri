""" Action data structures. """
from typing import NamedTuple

import numpy as np
import torch

import benri.dict as dict_ops


class Action(NamedTuple(
    "Action",
    [("index", np.array), ("distribution", np.array)])):
    pass


class SequenceAction(NamedTuple(
    "SequenceAction",
    [("index", np.array),
     ("distribution", np.array),
     ("length", np.array),
     ("state", np.array)])):
    pass


def select_action_from_batch(action, batch_i):
    """ Select a single action from batched-actions.

    :param action: Action, SequenceAction, or dict of actions.
    :return: Corresponding type of action without batch dimension.
    """
    # If we were given a single action, query batch.
    if isinstance(action, Action):
        return Action(
            index=action.index[batch_i],
            distribution=action.distribution[batch_i])

    elif isinstance(action, SequenceAction):
        raise SequenceAction(
            index=action.index[batch_i],
            distribution=action.distribution[batch_i],
            length=action.length[batch_i],
            state=action.length[batch_i])

    # Otherwise, make sure we have a dictionary.
    if not isinstance(action, dict):
        raise NotImplementedError("Must be type: Action, SeqAct, or Dict.")

    selected_actions = {}
    for k, a in action.items():
        if isinstance(a, Action):
            selected_actions[k] = Action(
                index=a.index[batch_i],
                distribution=a.distribution[batch_i])
                
        elif isinstance(a, SequenceAction):
            selected_actions[k] = SequenceAction(
                index=a.index[batch_i],
                distribution=a.distribution[batch_i],
                length=a.length[batch_i],
                state=a.state[batch_i])
                
        elif isinstance(a, torch.Tensor):
            selected_actions[k] = a[batch_i]
                    
        elif isinstance(a, np.ndarray):
            selected_actions[k] = a[batch_i]

        else:
            raise NotImplementedError(
                "Unknown action type: {}".format(type(a)))

    return selected_actions
    

def batch_actions(actions: list, keep_dims: bool=False):
    """ Batch a list of actions.

    :param actions:
    :param keep_dims: Keep the same shape, and batch on 0th dim.
    :return:
    """
    assert len(actions), "Must provide actions."
    assert isinstance(actions[0], dict), "Actions must be dicts."

    keys = actions[0].keys()
    batched = {}

    for key in keys:
        if isinstance(actions[0][key], Action):
            batched[key] = _batch_action(key, actions, keep_dims)

        elif isinstance(actions[0][key], SequenceAction):
            batched[key] = _batch_seqact(key, actions, keep_dims)

        elif isinstance(actions[0][key], torch.Tensor):
            value = [act[key] for act in actions]
            value = torch.cat(value, dim=0)
            batched[key] = value

        elif isinstance(actions[0][key], np.ndarray):
            value = [act[key] for act in actions]
            value = np.concatenate(value, axis=0)
            batched[key] = value

        else:
            raise NotImplementedError(
                "Unknown action type: {}".format(type(actions[0][key])))
 
    return batched


def _batch_action(key: str, actions: list, keep_dims: bool):
    """ Batch together actions.

    :param key: Action key.
    :param actions: List of action dicts.
    :return: Action with batch dimension.
    """
    indexs = []
    dists = []

    is_torch = False
    
    for action in actions:
        if isinstance(action[key].index, torch.Tensor):
            is_torch = True

        indexs += [action[key].index]
        dists += [action[key].distribution]

    if is_torch:
        if keep_dims:
            merge = lambda x: torch.cat(x, dim=0)
        else:
            merge = lambda x: torch.stack(x, dim=0)
    else:
        if keep_dims:
            merge = lambda x: np.concatenate(x, axis=0)
        else:
            merge = lambda x: np.stack(x, axis=0)

    action = Action(
        index=merge(indexs),
        distribution=merge(dists))
    return action    


def _batch_seqact(key: str, actions: list, keep_dims: bool):
    """ Batch together sequence actions.

    :param key: Action key.
    :param actions: List of action dicts.
    :return: SequenceAction with batch dimension.
    """
    indexs = []
    dists = []
    lengths = []
    states = []
    
    is_torch = False

    for action in actions:
        if isinstance(action[key].index, torch.Tensor):
            is_torch = True

        indexs += [action[key].index]
        dists += [action[key].distribution]
        lengths += [action[key].length]
        states += [action[key].state]

    if is_torch:
        if keep_dims:
            merge = lambda x: torch.cat(x, dim=0)
        else:
            merge = lambda x: torch.stack(x, dim=0)
    else:
        if keep_dims:
            merge = lambda x: np.concatenate(x, axis=0)
        else:
            merge = lambda x: np.stack(x, axis=0)

    action = SequenceAction(
        index=merge(indexs),
        distribution=merge(dists),
        length=merge(lengths),
        state=merge(states))
    return action    


def to_observation(actions: dict):
    """ Convert an action dictionary to an observation. 

    :param action:
    :return:
    """
    def _to_obs(x):
        if isinstance(x, Action):
            return x.index
        elif isinstance(x, SequenceAction):
            return x.index
        else:
            return x
    return dict_ops.apply_fn(actions, _to_obs)
