""" Configurable recurrent cell. """
import copy
from pydoc import locate

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from benri.configurable import Configurable


class RNN(nn.Module, Configurable):

    def __init__(self, params={}):
        nn.Module.__init__(self)
        Configurable.__init__(self, params=params)

        # Check for unimplemented conditions.
        if self.params["bidirectional"]:
            raise ValueError("Bidirectional not implemented.")
        if self.params["n_layers"] != 1:
            raise ValueError("More than 1 layer not implemented.")

        # Locate and build the cell.
        cell_ctor = locate("torch.nn.{}".format(self.params["cell_type"]))
        if cell_ctor is None:
            raise ValueError("Unknown RNN cell: {}".format(self.params["cell_type"]))
        self.rnn = cell_ctor(
            input_size=self.params["input_size"],
            hidden_size=self.params["hidden_size"],
            num_layers=self.params["n_layers"],
            batch_first=True)

    def forward(self, sequence, state):
        """ Wraps the RNN's forward call. """
        return self.rnn(sequence, state)

    def init_state(self, batch_size):
        """ Get a an initial zero state.

        :param batch_size: Number of examples in the batch.
        :return: Initial RNN state of zeros.
        """
        state_shape = [self.params["n_layers"]*(self.params["bidirectional"]+1)]
        state_shape += [batch_size]
        state_shape += [self.params["hidden_size"]]

        h = Variable(torch.zeros(state_shape), requires_grad=False).float()

        if self.params["cell_type"] == "LSTM":
            c = Variable(torch.zeros(state_shape), requires_grad=False)
            c = c.float()

            return (h, c)

        return h

    @property
    def hidden_size(self):
        if self.params["cell_type"] == "LSTM":
            return 2 * self.params["hidden_size"]
        else:
            return self.params["hidden_size"]

    @staticmethod
    def default_params():
        return {
            "cell_type": "LSTM",
            "input_size": 100,
            "hidden_size": 100,
            "n_layers": 1,
            "bidirectional": False}
