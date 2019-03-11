""" Class to serialize data for multiprocessing.

Inspired by: https://github.com/openai/baselines/blob/master/baselines/common/vec_env/__init__.py
"""
import cloudpickle
import pickle


class CloudpickleWrapper(object):
    """ Uses `cloudpickle` to serialize contents. """

    def __init__(self, data):
        self.data = data

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, data):
        self.data = pickle.loads(data)
        
    