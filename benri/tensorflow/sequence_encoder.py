""" Module to embed a sequence. """
import sonnet as snt
import tensorflow as tf
from tensorflow.python.util import nest

from benri.configurable import Configurable


class SequenceEncoder(Configurable, snt.AbstractModule):
    """ Embeds a sequence using an RNN. """

    def __init__(self, params, embedder=None, name="sequence_embedder"):
        snt.AbstractModule.__init__(self, name=name)
        Configurable.__init__(self, params)

        self.token_embedder = embedder
        if self.token_embedder is not None:
            self.params["embed.vocab_size"] = self.token_embedder.vocab_size
            self.params["embed.embed_size"] = self.token_embedder.embed_dim

    def _build(self, sequence, sequence_length, is_train, previous_state=None):
        """ Add to graph.

        :param sequence:
        :param sequence_length:
        :param support:
        :param channel_state:
        :param previous_state:
        :param is_train:
        :return:
        """
        batch_size = tf.shape(sequence)[0]

        # Convert sequence from token IDs to embeddings.
        if self.token_embedder is None:
            self.token_embedder = snt.Embed(
                vocab_size=self.params["embed.vocab_size"],
                embed_dim=self.params["embed.embed_size"])
        sequence = self.token_embedder(sequence)

        # Get an representation of the read sequence.
        sequence_encoder = tf.nn.rnn_cell.LSTMCell(
            self.params["sequence_encoder.n_units"])
        self.state_size = sum(sequence_encoder.state_size)

        if previous_state is not None:
            init_state = previous_state
            init_state = tf.split(
                init_state,
                len(nest.flatten(sequence_encoder.state_size)),
                axis=1)
            init_state = nest.pack_sequence_as(
                structure=sequence_encoder.state_size,
                flat_sequence=init_state)
        else:
            init_state = sequence_encoder.zero_state(
                batch_size=batch_size,
                dtype=tf.float32)

        _, current_state = tf.nn.dynamic_rnn(
            cell=sequence_encoder,
            inputs=sequence,
            sequence_length=sequence_length,
            initial_state=init_state)

        current_state = nest.flatten(current_state)
        current_state = tf.concat(current_state, axis=1)
        current_state = tf.reshape(current_state, [batch_size, self.state_size])
        return current_state

    @staticmethod
    def default_params():
        return {
            "embed.vocab_size": 11,
            "embed.embed_size": 100,
            "sequence_encoder.n_units": 100}
