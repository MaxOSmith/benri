""" Configurable recurrent cell. """
import copy
from pydoc import locate

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from benri.configurable import Configurable


class RNN(nn.Module, Configurable):

    def __init__(self, rnn=None, params={}):
        nn.Module.__init__(self)
        Configurable.__init__(self, params=params)

        # Check for unimplemented conditions.
        if self.params["bidirectional"]:
            raise ValueError("Bidirectional not implemented.")
        if self.params["n_layers"] != 1:
            raise ValueError("More than 1 layer not implemented.")

        if rnn:
            rnn = self.rnn
        else:            
            # Locate and build the cell.
            cell_ctor = locate("torch.nn.{}".format(self.params["cell_type"]))
            if cell_ctor is None:
                raise ValueError("Unknown RNN cell: {}".format(self.params["cell_type"]))
            self.rnn = cell_ctor(
                input_size=self.params["input_size"],
                hidden_size=self.params["hidden_size"],
                num_layers=self.params["n_layers"],
                batch_first=True)

    def forward(self, x, state):
        """ Wraps the RNN's forward call. 
        
        :param x: PackedSequence, or [B, S, E].
        :param state: [B, H]
        :return: Tuple
            - Outputs: 
            - State: 
        """
        assert isinstance(x, PackedSequence) or x.shape[0] == state.shape[0]

        # Add the sequence dimension to the hidden state. [B, E] -> [S, B, E].
        state = state.unsqueeze(0)

        if self.params["cell_type"] == "LSTM":
            state = torch.split(state, self.params["hidden_size"], dim=2)

        y, state = self.rnn(x, state)

        if self.params["cell_type"] == "LSTM":
            state = torch.cat(state, dim=2)

        # Remove the N-layers/bidirectional dimension from the hidden state.
        state = state.squeeze(0)

        return y, state

    def init_state(self, batch_size):
        """ Get a an initial zero state.

        :param batch_size: Number of examples in the batch.
        :return: Initial RNN state of zeros.
        """
        if self.params["cell_type"] == "LSTM":
            state_shape = [batch_size, self.params["hidden_size"] * 2]
        else:
            state_shape = [batch_size, self.params["hidden_size"]]
        
        state = Variable(torch.zeros(state_shape), requires_grad=False).float()
        return state

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
