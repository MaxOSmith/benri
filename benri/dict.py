""" Dictionary operations. """
import numpy as np


def stack(dicts: list, dim: int=0):
    """ Stack dict values.

    :param dicts: List of dictionaries.
    :param dim: New dimension to stack on.
    :return: Dictionary of stacked values.
    """
    stacked = {}
    keys = dicts[0].keys()

    for key in keys:
        vals = [d[key] for d in dicts]
        vals = np.stack(vals, axis=dim)
        stacked[key] = vals

    return vals


def apply_fn(data: dict, fn):
    """ Apply a function to all values in a dictionary.

    :param data:
    :param fn:
    :return: 
    """
    for key, value in data.items():
        data[key] = fn(value)
    return data


def select(data: dict, index: int=0):
    """ Select datum from batched dict values.

    :param data: Dictionary to batched values.
    :param index: Batch index.
    :return: Dictionary without batch dim.
    """
    selected = {}

    for key, value in data.items():
        selected[key] = value[index]

    return selected

