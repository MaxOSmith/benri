""" Decodes sequences from an initial state.

Resources:
 - https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
 - https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
"""
import torch
import torch.nn as nn

from benri.pytorch import RNN
from benri.configurable import Configurable


class SequenceDecoder(nn.Module, Configurable):

    def __init__(self, vocab_embedder, params={}):
        nn.Module.__init__(self)
        Configurable.__init__(self, params=params)

        self.rnn = RNN(self.params["rnn.params"])
        self.embedder = vocab_embedder
        self.projection = nn.Linear(
            self.rnn.params["hidden_size"],
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
        # Add the sequence dimension to the hidden state. [B, E] -> [S, B, E].
        hidden_state = hidden_state.unsqueeze(0)

        # LSTM needs to have two states.
        if self.rnn.params["cell_type"] == "LSTM":
            hidden_state = torch.split(
                hidden_state,
                [self.rnn.params["hidden_size"], self.rnn.params["hidden_size"]],
                dim=2)

        # Determine the maximum number of iterations to perform.
        seq_len = self.params["max_length"]
        if target is not None:
            seq_len = min(seq_len, target.shape[1])

        # Start by feeding the decoder the <SOS> token.
        out = torch.ones(batch_size, dtype=torch.long) * self.params["sos"]
        outputs = []
        output_dists = []

        for i in range(seq_len):
            # Teacher forcing, after first symbol.
            if target is not None and i:
                out = target[:, i]

            out = self.embedder(out)  # [B, E]
            out = out.unsqueeze(1)    # [B, 1, E]

            out_dist, hidden_state = self.rnn(out, hidden_state)

            out_dist = out_dist.squeeze(1)        # [B, 1, E] --> [B, E]
            out_dist = self.projection(out_dist)  # [B, E] --> [B, V]

            out_dist = torch.distributions.Categorical(logits=out_dist)

            if is_train:
                out = out_dist.sample()
            else:
                out = out_dist.probs.argmax(1)

            out = out.long()
            outputs += [out.unsqueeze(1)]
            output_dists += [out_dist.probs]

        outputs = torch.cat(outputs, dim=1)
        output_dists = torch.cat(output_dists, dim=1)

        # Calculate the sequence length.
        length = outputs == self.params["eos"]
        _, length = length.max(1)
        length += 1  # argmax accounts for 0 index.

        return outputs, output_dists, hidden_state, length

    @staticmethod
    def default_params():
        return {
            "rnn.params": {},
            "sos": 23,
            "eos": 24,
            "max_length": 6}
