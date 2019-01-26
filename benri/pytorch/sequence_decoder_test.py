""" Test suite for `rnn`. """
import numpy as np

import pytest
import torch

from benri.pytorch.rnn import RNN
from benri.pytorch.sequence_decoder import SequenceDecoder


class RNNMinusOne():

    def __init__(self):
        self.params = {}
        self.params["hidden_size"] = 7

    def __call__(self, x, h_):
        """ Mock RNN that subtracts one from the hidden state. """
        y = np.zeros_like(x.detach().numpy())
        y[np.arange(y.shape[0]), :, h_.squeeze(1)-1] = 1
        return torch.tensor(y), h_-1


def test_basic():
    """ Basic API test. """
    print()

    rnn = RNNMinusOne()
    embed = torch.nn.Embedding(
        num_embeddings=5 + 3,
        embedding_dim=7)

    seq_dec = SequenceDecoder(
        vocab_embedder=embed,
        rnn=rnn,
        params={
            "sos": 5,
            "eos": 0,
            "max_length": 5,
            "projection": False})

    state = torch.tensor([[3], [1], [0], [5]])

    o_i, o_v, h, l = seq_dec(state, is_train=False)

    print(o_i)
    print(h)
    print(l)

    np.testing.assert_array_equal(l, [3, 1, 5, 5])
