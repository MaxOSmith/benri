""" Experience replay buffer for a batched policy. """
import benri.dict as dict_ops
import benri.gym.action as action_ops
from .replay_buffer import ReplayBuffer


class BatchedReplayBuffer(ReplayBuffer):
    """ Maintains a buffer of experiences.

    Experiences are the tuple: (s, a, r, s', a'), with shapes [B, ...].
    """ 

    def add_experience(self, s, a, r, sp, ap, alive):
        """ Add a single batched experiences.

        """
        batch_size = len(r)

        for i in range(batch_size):
            if not alive[i]:
                continue

            s_ = dict_ops.select(s, i)
            a_ = action_ops.select_action_from_batch(a, i)
            r_ = r[i]
            sp_ = dict_ops.select(sp, i)
            ap_ = action_ops.select_action_from_batch(ap, i)

            exp = (s_, a_, r_, sp_, ap_)
            self._buffer.append(exp)

    

