""" Utility functions for observation spaces. """
from collections import defaultdict
import copy
import operator

import numpy as np

import benri.dict as dict_ops


def select_from_batch(observation, batch_i):
    """

    :param observations: List of observations. 
    :return:
    """
    selected = []

    for obs in observation:
        selected_from_obs = {}

        for key, value in obs.items():
            selected_from_obs[key] = value[batch_i]

        selected += [selected_from_obs]

    return selected 


def merge_batch_dynamic_observations(observations, o_types):
    """

    :param observations: List of observations. Each observation is a list 
        of sensor readings.
    :return:
    """
    batch_size = len(observations)

    # Start working backward from each batch.
    pointers = np.array([len(o) - 1 for o in observations])

    merged = []  
    mask = []

    while not np.all(pointers == -1):
        # Look at the pointers, get the most frequent type.
        types = [o_types[i][p] if p != -1 else None for i, p in enumerate(pointers)]

        counts = defaultdict(int)
        for t in types:
            if t is not None:
                counts[t] += 1
        o_type = max(counts.items(), key=operator.itemgetter(1))[0]

        o_is = [i for i, t in enumerate(types) if t == o_type]
        assert len(o_is) > 0

        # If 1: Expand dims
        if counts[o_type] == 1:
            assert len(o_is) == 1
            i = o_is[0]
            o = observations[i][pointers[i]]
            
            # Shift pointer.
            pointers[i] -= 1
            # Record.
            alive = np.zeros([batch_size])
            alive[i] = 1
            merged = [o] + merged
            mask += [alive]
            continue
        
        # Else: > 1.        
        
        # Grab one of the observation types, and tile it so that it 
        # has B as the leading dimension. Also start an aliveness mask
        # to determine which inputs to process.
        obs = observations[o_is[0]][pointers[o_is[0]]]

        alive = np.zeros([batch_size])

        for obs_batch_i in o_is:
            pointer = pointers[obs_batch_i]
            o = observations[obs_batch_i][pointer]

            # Load the correct data at this index.
            for key in o.keys():
                obs[key][obs_batch_i] = o[key][obs_batch_i]

            # Mark this data to be read.
            alive[obs_batch_i] = 1
            pointers[obs_batch_i] -= 1

        # Record.
        merged = [o] + merged
        mask += [alive]
    
    # mask = np.concatenate(mask, axis=0)
    return merged, mask


def merge_dynamic_observations(observations, o_types):
    """

    :param observations: List of observations. Each observation is a list 
        of sensor readings.
    :return:
    """
    batch_size = len(observations)

    def _expand_and_tile(x):
        x = np.expand_dims(x, axis=0)
        shape = np.ones_like(x.shape)
        shape[0] = batch_size        
        x = np.tile(x, shape)
        return x

    # Start working backward from each batch.
    pointers = np.array([len(o) - 1 for o in observations])

    merged = []  
    mask = []

    while not np.all(pointers == -1):
        # Look at the pointers, get the most frequent type.
        types = [o_types[i][p] if p != -1 else None for i, p in enumerate(pointers)]

        counts = defaultdict(int)
        for t in types:
            if t is not None:
                counts[t] += 1
        o_type = max(counts.items(), key=operator.itemgetter(1))[0]

        o_is = [i for i, t in enumerate(types) if t == o_type]
        assert len(o_is) > 0

        # If 1: Expand dims
        if counts[o_type] == 1:
            assert len(o_is) == 1
            i = o_is[0]
            o = copy.deepcopy(observations[i][pointers[i]])
            o = dict_ops.apply_fn(o, _expand_and_tile)
            
            # Shift pointer.
            pointers[i] -= 1
            # Record.
            alive = np.zeros([batch_size])
            alive[i] = 1
            # o = dict_ops.apply_fn(o, lambda x: np.concatenate(x, axis=0))
            merged = [o] + merged
            mask += [alive]
            continue
        
        # Else: > 1.        
        
        # Grab one of the observation types, and tile it so that it 
        # has B as the leading dimension. Also start an aliveness mask
        # to determine which inputs to process.
        obs = copy.deepcopy(observations[o_is[0]][pointers[o_is[0]]])
        obs = dict_ops.apply_fn(obs, _expand_and_tile)

        alive = np.zeros([batch_size])

        for obs_batch_i in o_is:
            pointer = pointers[obs_batch_i]
            o = observations[obs_batch_i][pointer]

            # Load the correct data at this index.
            for key in o.keys():
                obs[key][obs_batch_i] = o[key]

            # Mark this data to be read.
            alive[obs_batch_i] = 1
            pointers[obs_batch_i] -= 1

        # Record.
        # obs = dict_ops.apply_fn(obs, lambda x: np.concatenate(x, axis=0))
        merged = [obs] + merged
        mask += [alive]
    
    # mask = np.concatenate(mask, axis=0)
    return merged, mask
