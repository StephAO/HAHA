from oai_agents.common.arguments import get_arguments
from overcooked_ai_py.mdp.overcooked_mdp import Action

from gym import spaces
import numpy as np
from pathlib import Path
import torch as th


# Load any agent
def load_agent(agent_path, args=None):
    args = args or get_arguments()
    agent_path = Path(agent_path)
    load_dict = th.load(agent_path / 'agent_file', map_location=args.device)
    agent = load_dict['agent_type'].load(agent_path, args)
    return agent

def is_held_obj(player, object):
    '''Returns True if the object that the "player" picked up / put down is the same as the "object"'''
    x, y = player.position[0] + player.orientation[0], player.position[1] + player.orientation[1]
    return player.held_object is not None and \
           ((object.name == player.held_object.name) or
            (object.name == 'soup' and player.held_object.name == 'onion'))\
           and object.position == (x, y)

class DummyPolicy:
    def __init__(self, obs_space):
        self.observation_space = obs_space

class DummyAgent():
    def __init__(self, action=Action.STAY):
        self.name = f'{action}_agent'
        self.action = action if 'random' in action else Action.ACTION_TO_INDEX[action]
        self.policy = DummyPolicy(spaces.Dict({'visual_obs': spaces.Box(0,1,(1,))}))
        self.encoding_fn = lambda *args, **kwargs: {}
        self.use_hrl_obs = False

    def predict(self, x, state=None, episode_start=None, deterministic=False):
        add_dim = len(x) == 1
        if self.action == 'random':
            action = np.random.randint(0, Action.NUM_ACTIONS)
        elif self.action == 'random_dir':
            action = np.random.randint(0, 4)
        else:
            action = self.action
        if add_dim:
            action = np.array([action])
        return action, None

    def set_encoding_params(self, *args, **kwargs):
        pass

    def set_obs_closure_fn(self, obs_closure_fn):
        pass