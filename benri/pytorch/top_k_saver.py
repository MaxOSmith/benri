""" Class to save top k instances of a model. """
from collections import OrderedDict

import os
import os.path as osp

import torch


class TopKSaver():
    """ Save only the top k models according to some metric. """

    def __init__(self, k, directory, name):
        """ Constructor.

        :param k: Number of models to save.
        :param directory: Directory to save checkpoints into.
        :param name: Name to save the checkpoints under.
        """
        self.k = k
        self.dir = directory
        self.name = str(name) + "_{}.torch"

        self._value_to_saves = OrderedDict()

    def save(self, state_dict, value):
        """ Save the model.

        :param state_dict: Torch state dictionary.
        :param value: Metric determining goodness of this model.
        """
        if len(self._value_to_saves) < self.k:
            self._save(state_dict, value)
            return

        smallest_saved_value = self._value_to_saves.keys()
        smallest_saved_value = list(smallest_saved_value)[0]

        if smallest_saved_value > value:
            return

        self._delete(smallest_saved_value)
        self._save(state_dict, value)

    def _save(self, data, label):
        """ Save data.

        :param data: Torch state dictionary.
        :param label: Save label.
        """
        path = osp.join(self.dir, self.name.format(label))
        torch.save(data, path)
        self._value_to_saves[label] = path

    def _delete(self, label):
        """ Delete data.

        :param label: Save label.
        """
        path = osp.join(self.dir, self.name.format(label))
        os.remove(path)
        del self._value_to_saves[label]
