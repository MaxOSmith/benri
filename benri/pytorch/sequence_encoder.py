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

    def __init__(self, rnn=None, params={}):
        """

        :param rnn: RNN cell to encode with.
        :param params: Configuration dictionary.
        """
        nn.Module.__init__(self)
        Configurable.__init__(self, params=params)

        if rnn:
            self.rnn = rnn
        else:
            self.rnn = RNN(self.params["rnn.params"])

    def forward(self, sequence, sequence_length, hidden_state=None, is_train=True):
        """ Process sequences.

        :param sequence: Sequence to encode [B, S, E].
        :param sequence_length: Length of the sequences [B].
          - Note: All sequences must have length >0.
        :param hidden_state: Optional initial hidden state of RNN.
        :param is_train: Boolean.
        :return:
        """
        if hidden_state is None:
            hidden_state = self.rnn.init_state(sequence.shape[0])

        # Sort the sequences by length for packing.
        sequence_length, new_indices = torch.sort(sequence_length, dim=0, descending=True)
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
        hidden_state = hidden_state[original_indices]

        return output, hidden_state

    @staticmethod
    def default_params():
        return {
            "rnn.params": {}}


class UnpackedSequenceEncoder(nn.Module, Configurable):

    def __init__(self, rnn=None, params={}):
        """

        :param rnn: RNN cell to encode with.
        :param params: Configuration dictionary.
        """
        nn.Module.__init__(self)
        Configurable.__init__(self, params=params)

        if rnn:
            self.rnn = rnn
        else:
            self.rnn = RNN(self.params["rnn.params"])

    def forward(self, sequence, sequence_length, hidden_state=None, is_train=True):
        """ Process sequences.

        :param sequence: Sequence to encode [B, S, E].
        :param sequence_length: Length of the sequences [B].
          - Note: All sequences must have length >0.
        :param hidden_state: Optional initial hidden state of RNN.
        :param is_train: Boolean.
        :return:
        """
        if hidden_state is None:
            hidden_state = self.rnn.init_state(sequence.shape[0])

        raise NotImplementedError("")

        # Process the sequences with the RNN.
        packed_output, hidden_state = self.rnn(
            packed_sequence,
            hidden_state)

        return output, hidden_state

    @staticmethod
    def default_params():
        return {
            "rnn.params": {}}
