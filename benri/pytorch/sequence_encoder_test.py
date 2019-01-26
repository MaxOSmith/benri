""" Test suite for `sequence_encoder`. """
import pytest
import torch

from benri.pytorch.rnn import RNN
from benri.pytorch.sequence_encoder import SequenceEncoder


def _add_one_state_rnn(x, h):
    """ RNN that increases the state by one each timestep. """
    h += 1
    return x, h


def test_basic():
    """ Basic API test. """
    seq_enc = SequenceEncoder(rnn=_add_one_state_rnn, params={})

    sequence = torch.tensor([
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0]])
    sequence = sequence.unsqueeze(2)
    sequence_length = torch.tensor([2, 2, 4, 1])
    hidden_state = torch.zeros([4, 1])

    y, h = seq_enc(sequence, sequence_length, hidden_state)

    print(h)
