from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv
from oai_agents.common.subtasks import Subtasks, get_doable_subtasks

from overcooked_ai_py.mdp.overcooked_mdp import Action

from copy import deepcopy
from gym import spaces
import numpy as np
import torch as th


class OvercookedManagerGymEnv(OvercookedGymEnv):
    def __init__(self, worker=None, grid_shape=None, shape_rewards=False, randomize_start=False, args=None):
        self.worker = worker
        super(OvercookedManagerGymEnv, self).__init__(grid_shape=grid_shape, shape_rewards=shape_rewards,
                                                      ret_completed_subtasks=True, randomize_start=randomize_start, args=args)
        self.action_space = spaces.Discrete(Subtasks.NUM_SUBTASKS)
        self.terrain = self.mdp.terrain_mtx

    def get_low_level_obs(self, p_idx=None):
        obs = self.encoding_fn(self.env.mdp, self.state, self.grid_shape, self.args.horizon, p_idx=p_idx)
        if p_idx == self.p_idx:
            obs['curr_subtask'] = self.curr_subtask
        return obs

    def action_masks(self):
        return get_doable_subtasks(self.state, self.terrain, self.p_idx).astype(bool)

    def step(self, action):
        if self.teammate is None:
            raise ValueError('set_teammate must be set called before starting game unless play_both_players is True')
        # Action is the subtask for subtask agent to perform
        self.curr_subtask = action.cpu() if type(action) == th.tensor else action
        joint_action = [Action.STAY, Action.STAY]
        reward, done, info = 0, False, None
        while joint_action[self.p_idx] != Action.INTERACT and not done:
            joint_action[self.p_idx] = self.worker.predict(self.get_low_level_obs(p_idx=self.p_idx))[0]
            joint_action[self.t_idx] = self.teammate.predict(self.get_low_level_obs(p_idx=self.t_idx))[0]
            # joint_action = [self.agents[i].predict(self.get_obs(p_idx=i))[0] for i in range(2)]
            joint_action = [Action.INDEX_TO_ACTION[a] for a in joint_action]

            # If the state didn't change from the previous timestep and the agent is choosing the same action
            # then play a random action instead. Prevents agents from getting stuck
            if self.prev_state and self.state.time_independent_equal(self.prev_state) and tuple(joint_action) == self.prev_actions:
                joint_action = [np.random.choice(Action.ALL_ACTIONS), np.random.choice(Action.ALL_ACTIONS)]

            self.prev_state, self.prev_actions = deepcopy(self.state), joint_action
            next_state, r, done, info = self.env.step(joint_action)
            reward += r
            self.state = self.env.state

        return self.get_obs(self.p_idx), reward, done, info

    def reset(self):
        self.env.reset()
        self.state = self.env.state
        self.prev_state = None
        self.p_idx = np.random.randint(2)
        self.t_idx = 1 - self.p_idx
        self.curr_subtask = 0
        return self.get_obs(self.p_idx)
