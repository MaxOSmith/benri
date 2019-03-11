""" Experience replay buffer that tracks an episode. """
from .replay_buffer import ReplayBuffer


class EpisodicReplayBuffer(ReplayBuffer):
    """ Maintains a buffer of experiences for only a single episode.

    Experiences are the tuple: (s, a, r, s', a'), with shapes [B, ...].
    """ 
    
    def __init__(self, params: dict={}):
        """ Constructor.

        :param params: Configuration dictionary. 
        """
        ReplayBuffer.__init__(self, params)
        self._terminated = False

    def add_experience(self, s, a, r, sp, ap, terminal):
        """ Add a single experiences.

        """
        assert not self._terminated, "Can only maintain a single episode."
        self._terminated |= terminal

        exp = (s, a, r, sp, ap)
        # Because we're using a maxlen deque, if we exceed capacity 
        # an item will get removed from the other end.
        self._buffer.append(exp)

        # If the state was terminal, add discounted sum of rewards.
        gamma = self.params["gamma"]
        reward = 0.0

        for i in range(self.n-1, -1, -1):
            reward = self._buffer[i][2] + gamma*reward
            self._buffer[i][2] = reward

    def clear():
        """ Empty replay buffer. """
        self._buffer.clear()
        self._terminated = False

    @staticmethod
    def default_params():
        params = ReplayBuffer.default_params()
        params["gamma"] = 0.99
        return params
