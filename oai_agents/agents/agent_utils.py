from oai_agents.common.arguments import get_arguments
from overcooked_ai_py.mdp.overcooked_mdp import Action, Direction

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
    x, y = np.array(player.position) + np.array(player.orientation)
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
        if self.action == 'random':
            action = np.random.randint(0, Action.NUM_ACTIONS)
        elif self.action == 'random_dir':
            action = np.random.randint(0, 4)
        else:
            action = self.action
        return action, None

    def set_idx(self, *args, **kwargs):
        pass

    def set_obs_closure_fn(self, obs_closure_fn):
        pass

class TutorialAgent():
    COOK_SOUP_LOOP = [
        # Grab first onion
        Direction.WEST,
        Direction.WEST,
        Direction.WEST,
        Action.INTERACT,

        # Place onion in pot
        Direction.EAST,
        Direction.NORTH,
        Action.INTERACT,

        # Grab second onion
        Direction.WEST,
        Action.INTERACT,

        # Place onion in pot
        Direction.EAST,
        Direction.NORTH,
        Action.INTERACT,

        # Grab third onion
        Direction.WEST,
        Action.INTERACT,

        # Place onion in pot
        Direction.EAST,
        Direction.NORTH,
        Action.INTERACT,

        # Grab plate
        Direction.EAST,
        Direction.SOUTH,
        Action.INTERACT,
        Direction.WEST,
        Direction.NORTH,

        # Wait for soup to cook
        Action.STAY, Action.STAY, Action.STAY, Action.STAY, Action.STAY, Action.STAY, Action.STAY,

        # Deliver soup
        Action.INTERACT,
        Direction.EAST,
        Direction.EAST,
        Direction.EAST,
        Action.INTERACT,
        Direction.WEST
    ]

    def __init__(self):
        self.name = 'tutorial agent'
        self.curr_phase = -1
        self.curr_tick = -1
        self.policy = DummyPolicy(spaces.Dict({'visual_obs': spaces.Box(0, 1, (1,))}))
        self.encoding_fn = lambda *args, **kwargs: {}
        self.use_hrl_obs = False
        self.slowdown = 2

    def predict(self, x, state=None, episode_start=None, deterministic=False):
        self.curr_tick += 1
        act = self.COOK_SOUP_LOOP[(self.curr_tick // self.slowdown) % len(self.COOK_SOUP_LOOP)] if (
                self.curr_tick % self.slowdown == 0) else Action.STAY
        # if self.curr_phase == 0:
        return Action.ACTION_TO_INDEX[act], None
        # elif self.curr_phase == 2:
            # return self.COOK_SOUP_COOP_LOOP[self.curr_tick % len(self.COOK_SOUP_COOP_LOOP)], None
        # return Action.STAY, None

    def reset(self):
        self.curr_tick = -1
        self.curr_phase += 1

    def set_encoding_params(self, mdp, horizon):
        pass

    def set_idx(self, *args, **kwargs):
        pass

    def set_obs_closure_fn(self, obs_closure_fn):
        pass