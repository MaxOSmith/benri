""" Encodes sequences with non-uniform length.

References:
 - https://discuss.pytorch.org/t/correct-way-to-declare-hidden-and-cell-states-of-lstm/15745
 - https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from benri.pytorch import RNN
from benri.configurable import Configurable


class SequenceEncoder(nn.Module, Configurable):

    def __init__(self, params={}):
        nn.Module.__init__(self)
        Configurable.__init__(self, params=params)

        self.rnn = RNN(self.params["rnn.params"])

    def forward(self, sequence, sequence_length, hidden_state=None, is_train=True):
        """

        :param sequence:
        :param sequence_length:
        :param hidden_state:
        :param is_train:
        :return:
        """
        if hidden_state is None:
            hidden_state = self.rnn.init_state(sequence.shape[0])

        # Sort the sequences by length for packing.
        sequence_length, new_indices = sequence_length.sort(0, descending=True)
        sequence = sequence[new_indices]

        # Pack the sequences.
        packed_sequence = pack_padded_sequence(
            sequence,
            sequence_length,
            batch_first=True)

        # Process the sequences with the RNN.
        packed_output, hidden_state = self.rnn(
            packed_sequence,
            hidden_state)

        # Unpack the sequences.
        output, output_length = pad_packed_sequence(packed_output, batch_first=True)

        # Unsort the sequences.
        _, original_indices = new_indices.sort(0)
        output = output[original_indices]

        if self.rnn.params["cell_type"] == "LSTM":
            hidden_state = (
                hidden_state[0][:, original_indices],
                hidden_state[1][:, original_indices])
        else:
            hidden_state = hidden_state[:, original_indices]

        return output, hidden_state


    @staticmethod
    def default_params():
        return {
            "rnn.params": {}}
