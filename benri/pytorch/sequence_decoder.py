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
        if is_train and target is None:
            raise ValueError("Teacher forcing requires a target sequence.")


        batch_size = hidden_state.shape[0]

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

        for i in range(seq_len):
            # Teacher forcing, after first symbol.
            if is_train and i:
                out = target[:, i]

            out = self.embedder(out)
            out = out.unsqueeze(1)

            out, hidden_state = self.rnn(out, hidden_state)

            out = out.squeeze(1)
            out = self.projection(out)
            out = self.softmax(out)
            # TODO(maxsmith): Sampling.
            out = out.max(1)[1]
            out = out.long()

            outputs += [out.unsqueeze(1)]

        outputs = torch.cat(outputs, dim=1)
        return outputs

    @staticmethod
    def default_params():
        return {
            "rnn.params": {},
            "sos": 23,
            "eos": 24,
            "max_length": 10}
