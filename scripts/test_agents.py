from oai_agents.agents import OAIAgent
from overcooked_ai_py.mdp.overcooked_mdp import Action

import numpy as np
import torch as th
from torch.distributions.categorical import Categorical


class SingleActionAgent(OAIAgent):
    def __init__(self, action=Action.STAY):
        super(SingleActionAgent, self).__init__()
        self.action = Action.ACTION_TO_INDEX[action]
        self.name = f'single_action_agent_{Action.ACTION_TO_CHAR[action]}'

    def predict(self, obs: th.Tensor, state=None, episode_start=None, deterministic=False):
        return self.action, None

    def get_distribution(self, obs: th.Tensor):
        probs = np.zeros(len(Action.ALL_ACTIONS))
        probs[self.action] = 1
        return Categorical(probs=probs)

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass


class UniformActionAgent(OAIAgent):
    def __init__(self):
        super(UniformActionAgent, self).__init__()
        self.action = Action.ACTION_TO_INDEX[action]
        self.name = f'uniform_action_agent'

    def predict(self, obs: th.Tensor, state=None, episode_start=None, deterministic=False):
        probs = np.full(len(Action.ALL_ACTIONS), 1 / len(Action.ALL_ACTIONS))
        return Categorical(probs=probs).sample()

    def get_distribution(self, obs: th.Tensor):
        probs = np.full(len(Action.ALL_ACTIONS), 1 / len(Action.ALL_ACTIONS))
        return Categorical(probs=probs)

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass

