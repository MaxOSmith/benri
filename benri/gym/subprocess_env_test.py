""" Test suite for `subprocess_env`. """
import numpy as np
import pytest 

from benri.gym.env import Env
from benri.gym.subprocess_env import SubprocessEnv


class PlusOneEnv(Env):
    """ . """
    pass    


def test_episode():
    """ Tests running through a SubprocessEnv episode. 
    
    Resources:
      - https://github.com/openai/baselines/blob/master/baselines/ppo2/runner.py
      - 
    """

    agents = {}

    env_ctors = []
    params = {}

    env = SubprocessEnv(env_ctors=env_ctors, params=params)



