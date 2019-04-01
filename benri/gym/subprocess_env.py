""" Handle multiple envs in parallel subprocesses. 

Inspired from:
  - VecEnv: https://github.com/openai/baselines/blob/master/baselines/common/vec_env/__init__.py
  - SubprocVecEnv: https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py 
"""
from collections import OrderedDict
from multiprocessing import Process, Pipe, Connection

import numpy as np

from benri.cloudpickle_wrapper import CloudpickleWrapper
from benri.gym.env import Env


class SubprocessEnv(Env):
    """ Manages multiple envs in subprocesses through pipes.

    TODO(max): `render()`.
    """

    def __init__(self, env_ctors: list, params: dict={}):
        """ Constructor

        :param env_ctors: List of env constructors.
        :param params: Parameter dictionary.
        :return: Self.
        """
        # Process communication status.
        self.waiting = False
        self.closed = False

        # Pipe returns a duplex connections through a pipe (parent, child).
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_envs)])

        # Build env processes.
        self.processes = []
        payload = zip(self.work_remotes, self.remotes, env_ctors)
        for i, (work_remote, remote, env_ctor) in enumerate(payload):
            process = Process(
                target=worker_process,
                args=(i, work_remote, remote, CloudpickleWrapper(env_ctor)))
            self.processes += [process]

        # Start running all env processes as daemons.
        for p in self.processes:
            p.daemon = True  
            p.start()

        # TODO(max): Understand.
        for worker in self.work_remotes:
            worker.close()

        # Ask an env for space information.
        o_space, a_space = self.remotes[0].recv()
        
        self.n_envs = len(env_ctors)
        self.observation_space = o_space
        self.action_space = a_space

    def step(self, actions: dict):
        """ Synchronously step all environments.
        
        :param actions: Dictionary from agent ID to their action.
        """
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions: dict):
        """ Tell all envs to start processing the given actions.

          - Call `step_wait()` to get the results of the step.
          - You should not be calling this if a `step_async()` run is 
            already pending.

        :param actions: Dictionary from agent ID to their action.
        """
        self._assert_not_closed()

        raise NotImplementedError("Cannot traverse over agent actions?")
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        """ Wait for the transition from `step_async()`.

        :return: (o, r, done, info):
          - o:
          - r:
          - done:
          - info:
        """
        self._assert_not_closed()

        results = [remote.recv() for remote in self.remotes]
        self.waiting = False

        o, r, done, info = zip(*results)
        return _flatten_obs(o), np.stack(r), np.stack(done), info

    def reset(self):
        """ Reset all enviornments.

        :return: Dictionary from agent ID to their initial observation.
        """
        self._assert_not_closed()

        for remote in self.remotes:
            remote.send(("reset", None))

        return _flatten_obs([remote.recv() for remote in self.remotes])

    def _assert_not_closed(self):
        """ Assert that the envs & connections are not closed. """
        assert not self.closed, "Cannot use a SubprocessEnv after calling `close()`"


def worker_process(
    pid: int, 
    remote: Connection, 
    parent_remote: Connection, 
    pickled_env_ctor: CloudpickleWrapper):
    """ Worker processes that handles a single env.

    :param pid: Process ID.
    :param remote: 
    :param parent_remote:
    :param pickled_env_ctor: 
    """
    # TODO(max): Understand.
    parent_remote.close()

    # Get data from pickle, which is our ctor and call it.
    env = pickled_env_ctor.data()

    try: 
        # Process incoming env requests.
        while True:    
            # Wait on request.
            cmd, data = remote.recv()

            if cmd == "step":
                o, r, done, info = env.step(data)
                remote.send((o, r, done, info))

            elif cmd == "reset":
                o = env.reset()
                remote.send(o)

            elif cmd == "close":
                remote.close()
                break

            # Allow `SubprocessEnv` to ask any env what the obs and act spaces
            # are across the envs it is managing.
            elif cmd == "_spaces":
                remote.send((env.observation_space, env.action_space))

            else:
                raise NotImplementedError("Unknown command: {}".format(cmd))

    except KeyboardInterrupt:
        print("SubprocessEnv worker {}: KeyboardInterrupt".format(pid))

    finally:
        env.close()


def _flatten_obs(obs):
    """ 

    """
    raise NotImplementedError("Won't also nest agent IDs")

    assert isinstance(obs, list) or isinstance(obs, tuple)
    assert len(obs) > 0

    if isinstance(obs[0], dict):
        assert isinstance(obs, OrderedDict)
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return np.stack(obs)
    