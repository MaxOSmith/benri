""" Batched experience replay buffer that tracks epsiodes. 

Does not conform to ReplayBuffer class, because the buffer is maintained 
differently. The only API change that this has ann effect on is that 
`add_buffer` is not defined.

You probably should NOT be using this, it is kept for backwards compatibility.
"""
import torch

import numpy as np

from benri.configurable import Configurable
import benri.gym.action as action_ops
import benri.gym.observation as observation_ops


class BatchedEpisodicReplayBuffer(Configurable):
    """ Maintains buffers for a batch of a single episode. """

    def __init__(self, params: dict={}):
        """ Constructor. 

        :param params: Configuration dictionary.
        """
        Configurable.__init__(self, params=params)
        self._buffer = []

    def add_experience(self, o, o_type, a, r, alive):
        """ Add an experience tuple.

        :param o:
        :param o_type:
        :param a:
        :param r:
        :param alive:
        """
        self._buffer += [(o, o_type, a, r, alive)]

    def reward_last_experience(self, r):
        """ Reward the last experience.

        """
        exp = self._buffer[-1]
        exp = list(exp)
        exp[3] = r
        self._buffer[-1] = tuple(exp)

    def get_experiences(self, batch_size):
        # Batch rewards.
        rs = [x[3] for x in self._buffer]
        # Discounting
        gamma = self.params["gamma"]
        ttl_reward = np.zeros_like(rs[-1])
        for i in range(len(rs)-1, -1, -1):
            ttl_reward = rs[i] + gamma*ttl_reward
            rs[i] = ttl_reward
        
        experiences = []

        for exp_i, batched_exp in enumerate(self._buffer):

            for batch_i in range(batch_size):
                # If not alive, skip.
                if not batched_exp[-1][batch_i]:
                    continue

                action = action_ops.select_action_from_batch(
                    batched_exp[2], 
                    batch_i)
                observation = observation_ops.select_from_batch(
                    batched_exp[0],
                    batch_i)
                # Look-up the discounted reward.
                reward = rs[exp_i][batch_i]
                
                experiences += [(observation, action, reward)]

        return experiences

    def get_batched_experiences(self):
        """ Get all experiences.

        :return: 
        """
        # Batch observations.
        os = [x[0] for x in self._buffer]
        o_types = [x[1] for x in self._buffer]
        batch_os, batch_os_mask = observation_ops.merge_batch_dynamic_observations(
            os, o_types)

        # Batch actions.
        actions = [x[2] for x in self._buffer]
        batch_as = action_ops.batch_actions(actions, keep_dims=True)

        # Batch rewards.
        rs = [x[3] for x in self._buffer]
        # Discounting
        gamma = self.params["gamma"]
        ttl_reward = np.zeros_like(rs[-1])
        for i in range(len(rs)-1, -1, -1):
            ttl_reward = rs[i] + gamma*ttl_reward
            rs[i] = ttl_reward
        
        batch_rs = np.concatenate(rs, axis=0)

        return batch_os, batch_os_mask, batch_as, batch_rs

    def clear(self):
        """ Empty replay buffer. """
        self._buffer = []

    @staticmethod
    def default_params():
        params = {}
        params["gamma"] = 0.99
        return params
