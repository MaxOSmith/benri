""" Decodes sequences from an initial state.

Resources:
 - https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
 - https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
"""
import numpy as np
import torch
import torch.nn as nn

from benri.pytorch import RNN
from benri.configurable import Configurable


class SequenceDecoder(nn.Module, Configurable):

    def __init__(self, vocab_embedder, rnn=None, params={}):
        nn.Module.__init__(self)
        Configurable.__init__(self, params=params)

        if rnn:
            self.rnn = rnn
        else:
            self.rnn = RNN(self.params["rnn.params"])

        self.embedder = vocab_embedder
        if self.params["projection"]:
            self.projection = nn.Linear(
                self.rnn.output_size,
                self.embedder.num_embeddings)
        self.softmax = nn.Softmax()

    def forward(self, hidden_state, target=None, is_train=True):
        """

        TODO(maxsmith): You can be in training mode and not use teacher forcing.

        :param hidden_state:
        :param target: [B, S].
          - Does not have SOS.
        :param is_train:
        :return:

        """
        batch_size = hidden_state.shape[0]

        # Determine the maximum number of iterations to perform.
        seq_len = self.params["max_length"]
        if target is not None:
            seq_len = min(seq_len, target.shape[1])

        # Start by feeding the decoder the <SOS> token.
        out = torch.ones(batch_size, dtype=torch.long) * self.params["sos"]
        outputs = []
        output_dists = []
        hidden_states = []

        for i in range(seq_len):
            # Teacher forcing, after first symbol.
            if target is not None and i:
                out = target[:, i]

            out = self.embedder(out)  # [B, E]
            out = out.unsqueeze(1)    # [B, 1, E]

            out_dist, hidden_state = self.rnn(out, hidden_state)

            out_dist = out_dist.squeeze(1)        # [B, 1, E] --> [B, E]
            if self.params["projection"]:
                out_dist = self.projection(out_dist)  # [B, E] --> [B, V]

            out_dist = torch.distributions.Categorical(logits=out_dist)

            if is_train:
                out = out_dist.sample()
            else:
                out = out_dist.probs.argmax(1)

            out = out.long()
            outputs += [out.unsqueeze(1)]
            output_dists += [out_dist.probs]
            hidden_states += [hidden_state.unsqueeze(1)]

        outputs = torch.cat(outputs, dim=1)
        output_dists = torch.cat(output_dists, dim=1)

        # Calculate the sequence length.
        length = outputs == self.params["eos"]
        _, length = length.max(1)
        length += 1  # argmax accounts for 0 index.

        # Get the right hidden states.
        hidden_states = torch.cat(hidden_states, dim=1)
        batch_i = np.arange(batch_size)
        hidden_state = hidden_states[batch_i, length-1]

        return outputs, output_dists, hidden_state, length

    @staticmethod
    def default_params():
        return {
            "sos": 9,
            "eos": 10,
            "max_length": 6,
            "rnn.params": {},
            "projection": True}
