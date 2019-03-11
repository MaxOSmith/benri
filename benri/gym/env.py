""" Reinforcement learning environment. 

This is a redesign of OpenAI's gym, so that it is compatible with
multi-agent systems. We also emphasize dynamic observation spaces in 
our design.

Inspired by: https://github.com/openai/gym/blob/master/gym/core.py
"""
from benri.configurable import abstractstaticmethod
from benri.configurable import Configurable
from benri.gym.action import Action, SequenceAction


class Env(Configurable):

    metadata = {"render.modes": []}
    reward_range = (-float("inf"), float("inf"))

    action_space = None
    observation_space = None
    n_agents = None

    def __init__(self, debug=False, params: dict={}):
        """ Constructor.

        :param params: Parameter dictionary.
        :return: Self.
        """
        Configurable.__init__(self, params=params)
        self.debug = debug

    def step(self, actions: dict):
        """ Steps the environment forward one timestep following its dynamics.

        :param actions: Dictionary from agent ID to their action.
        :return: Dictionary from agent ID to their observation.
        """
        if self.debug:
            assert len(actions) >= self.n_agents

            for agent_id, agent_action in actions.items():
                assert isinstance(agent_id, int), \
                    "Agent ID must be int [0, n_agents)."
                assert isinstance(agetn_action, dict), \
                    "Actions must be a dict from type to Action/SequenceAction"

                for action_name, action in agent_action.items():
                    assert isinstance(action_name, str)
                    assert isinstance(action, Action) or \
                        isinstance(action, SequenceAction)

        o, r, done, info = self._step(actions)

        if self.debug:
            raise NotImplementedError

        return o, r, done, info

    def reset(self):
        """ Resets the state of the environment.

        :return: Dictionary from agent ID to their initial observation.
        """
        o = self._reset()

        if self.debug:
            raise NotImplementedError

        return o

    def render(self, mode: str):
        """ Render the environment, if supported.

        :param mode: The rendering mode.
        :return: None, a renderable object.
        """
        return self._render(mode)

    @property
    def unwrapped(self):
        """ Remove all environment wrappers.

        :return: The base environment.
        """
        return self

    def __str__(self):
        """ Get a string representation of this environment. 
        
        :return: String.
        """
        return "<{} instance>".format(type(self).__name__)

    @abstractstaticmethod
    def default_params():
        """ Default parameter dictionary. 
        
        :return: Dictionary.
        """
        return {}
