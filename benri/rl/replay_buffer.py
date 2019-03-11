""" Experience replay buffer. """
from collections import deque
import random

import numpy as np

from benri.configurable import Configurable
import benri.dict as dict_ops
import benri.gym.action as action_ops
import benri.gym.observation as observation_ops


class ReplayBuffer(Configurable):
    """ Maintains a buffer of experiences.

    Experiences are the tuple: (s, a, r, s', a').
    """

    def __init__(self, params: dict={}):
        """ Constructor.

        :param params: Configuration dictionary.
        """
        Configurable.__init__(self, params=params)
                
        self._buffer = deque(maxlen=self.params["buffer_size"])

        # Set random seed.
        random.seed(self.params["random_seed"])
        np.random.seed(self.params["random_seed"])

    def sample(self, batch_size: int):
        """ Sample a mini-batch of experiences.
        
        :param batch_size: Number of experiences.
        :return: Experiences, 5 tuple of:
            - s: State.
            - a: Action.
            - r: Reward.
            - sp: Next state.
            - ap: Next action.
        """    
        sample_size = batch_size if batch_size < self.n else n
        batch = random.sample(self._buffer, sample_size)
        
        batch_o_types = []
        for batch_i in range(sample_size):
            o_types = []

            for observation in batch[batch_i][0]:
                name = '_'.join(observation.keys())
                o_types += [name]

            batch_o_types += [o_types]
                
        s = observation_ops.merge_dynamic_observations(
            [x[0] for x in batch], 
            batch_o_types)
        a = action_ops.batch_actions([x[1] for x in batch])
        r = np.stack([x[2] for x in batch], axis=0)
        # sp = dict_ops.stack([x[3] for x in batch])
        # ap = action_ops.batch_actions([x[4] for x in batch])

        return s, a, r, None, None

    def add_experience(self, s, a, r, sp, ap):
        """ Add a single experiences.

        """
        exp = (s, a, r, sp, ap)
        # Because we're using a maxlen deque, if we exceed capacity 
        # an item will get removed from the other end.
        self._buffer.append(exp)

    def add_buffer(self, other_buffer: "ReplayBuffer"):
        """ Add a sample of experiences.

        :param other_buffer:
        """
        self._buffer.extend(other_buffer._buffer)
    
    @property
    def n(self):
        """ Get the number of elements in the buffer.

        :return: Number of elements in buffer. 
        """
        return len(self._buffer)

    def clear(self):
        """ Empty replay buffer. """
        self._buffer.clear()

    @staticmethod
    def default_params():
        return {
            "buffer_size": 10000,
            "random_seed": 1}