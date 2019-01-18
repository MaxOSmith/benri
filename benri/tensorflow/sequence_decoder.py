""" Module to embed a sequence. """
import sonnet as snt
import tensorflow as tf
from tensorflow.python.util import nest

from benri.configurable import Configurable


class SequenceDecoder(Configurable, snt.AbstractModule):
    """ Decodes a sequence from an embedding. """

    def __init__(self, params, embedder=None, name="sequence_decoder"):
        snt.AbstractModule.__init__(self, name=name)
        Configurable.__init__(self, params)

        self.token_embedder = embedder
        if self.token_embedder is not None:
            self._params["embed.vocab_size"] = self.token_embedder.vocab_size
            self._params["embed.embed_size"] = self.token_embedder.embed_dim

    @snt.reuse_variables
    def _decode(self, init_state, helper):
        """ Decode a sequence.

        :param helper: Decoding helper.
        :return: (Sequence, State, Length).
        """
        decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(
            self.sequence_decoder,
            self._params["embed.vocab_size"])
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_cell,
            helper=helper,
            initial_state=init_state)
        sequence = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            maximum_iterations=self._params["sequence_decoder.max_length"])
        return sequence

    def _build(self, init_state, is_train):
        """

        :param init_state:
        :param is_train:
        """
        batch_size = tf.shape(init_state)[0]
        sos = self._params["vocab.sos"]
        eos = self._params["vocab.eos"]

        self.sequence_decoder = tf.nn.rnn_cell.LSTMCell(
            self._params["sequence_decoder.n_units"],
            name="sequence_decoder")

        init_state = nest.pack_sequence_as(
            structure=self.sequence_decoder.state_size,
            flat_sequence=tf.split(init_state, 2, axis=1))

        sample_helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
            self.token_embedder.embeddings,
            tf.fill([batch_size], sos),
            eos,
            softmax_temperature=self._params["sequence_decoder.temperature"])
        greedy_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            self.token_embedder.embeddings,
            tf.fill([batch_size], sos),
            eos)

        sequence = tf.cond(
            pred=is_train,
            true_fn=lambda:self._decode(init_state, sample_helper),
            false_fn=lambda:self._decode(init_state, greedy_helper))

        return sequence

    @staticmethod
    def default_params():
        return {
            "vocab.sos": None,
            "vocab.eos": None,
            "embed.vocab_size": 100,
            "embed.embed_size": 100,
            "sequence_decoder.n_units": 100,
            "sequence_decoder.max_length": 6,
            "sequence_decoder.temperature": None}
