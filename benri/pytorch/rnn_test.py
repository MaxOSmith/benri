""" Test suite for `rnn`. """
import pytest
import torch

from benri.pytorch.rnn import RNN


def test_gru():
    """ Test basic usage of GRU. """
    rnn = RNN(params={"cell_type": "GRU"})
    assert rnn.hidden_size == 100

    sequence = torch.ones([32, 10, 100])
    state = torch.ones([32, 100])

    y, state = rnn(sequence, state)

def test_lstm():
    """ Test basic usage of LSTM. """
    rnn = RNN(params={"cell_type": "LSTM"})
    # LSTM has two hidden components, so twice as much memory.
    assert rnn.hidden_size == 200

    sequence = torch.ones([32, 10, 100])
    state = torch.ones([32, 200])

    y, state= rnn(sequence, state)


