from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv
from oai_agents.common.subtasks import Subtasks, get_doable_subtasks, calculate_completed_subtask

from overcooked_ai_py.mdp.overcooked_mdp import Action, Direction

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
        self.n_counters = len(self.find_free_counters_valid_for_both_players(self.env.state, self.env.mlam))

    def get_low_level_obs(self, p_idx=None):
        obs = self.encoding_fn(self.env.mdp, self.state, self.grid_shape, self.args.horizon, p_idx=p_idx)
        if p_idx == self.p_idx:
            obs['curr_subtask'] = self.curr_subtask
        return obs

    def action_masks(self):
        return get_doable_subtasks(self.state, self.terrain, self.p_idx, self.n_counters).astype(bool)

    def step(self, action):
        # if self.teammate is None:
        #     raise ValueError('set_teammate must be set called before starting game unless play_both_players is True')
        # Action is the subtask for subtask agent to perform
        self.curr_subtask = action.cpu() if type(action) == th.tensor else action
        joint_action = [Action.STAY, Action.STAY]
        reward, done, info = 0, False, None
        ready_for_next_subtask = False
        worker_steps = 0
        while (not ready_for_next_subtask and not done):
            joint_action[self.p_idx] = self.worker.predict(self.get_low_level_obs(p_idx=self.p_idx))[0]
            joint_action[self.t_idx] = self.teammate.predict(self.get_low_level_obs(p_idx=self.t_idx))[0]
            # joint_action = [self.agents[i].predict(self.get_obs(p_idx=i))[0] for i in range(2)]
            joint_action = [Action.INDEX_TO_ACTION[a] for a in joint_action]

            # If the state didn't change from the previous timestep and the agent is choosing the same action
            # then play a random action instead. Prevents agents from getting stuck
            if self.prev_state and self.state.time_independent_equal(self.prev_state) and tuple(joint_action) == self.prev_actions:
                joint_action = [np.random.choice(Direction.ALL_DIRECTIONS), np.random.choice(Direction.ALL_DIRECTIONS)]

            self.prev_state, self.prev_actions = deepcopy(self.state), deepcopy(joint_action)
            next_state, r, done, info = self.env.step(joint_action)
            if self.shape_rewards:
                ratio = min(self.step_count * self.args.n_envs / 2.5e6, 1)
                sparse_r = sum(info['sparse_r_by_agent'])
                shaped_r = info['shaped_r_by_agent'][self.p_idx] if self.p_idx else sum(info['shaped_r_by_agent'])
                reward += sparse_r * ratio + shaped_r * (1 - ratio)
            else:
                reward += r
            self.step_count += 1
            worker_steps += 1
            self.state = self.env.state

            if worker_steps % 5 == 0:
                if not get_doable_subtasks(self.state, self.terrain, self.p_idx, self.n_counters)[self.curr_subtask]:
                    ready_for_next_subtask = True
            if worker_steps % 40:
                ready_for_next_subtask = True

            if joint_action[self.p_idx] == Action.INTERACT:
                completed_subtask = calculate_completed_subtask(self.terrain, self.prev_state, self.state, self.p_idx)
                if completed_subtask != self.curr_subtask:
                    completed_subtask_str = Subtasks.IDS_TO_SUBTASKS[completed_subtask] if (completed_subtask is not None) else 'None'
                    print(f'Worker Failure! -> goal: {Subtasks.IDS_TO_SUBTASKS[self.curr_subtask]}, completed: {completed_subtask_str}', flush=True)
                ready_for_next_subtask = (completed_subtask is not None)

        return self.get_obs(self.p_idx), reward, done, info

    def reset(self):
        self.env.reset()
        self.state = self.env.state
        self.prev_state = None
        self.p_idx = np.random.randint(2)
        self.t_idx = 1 - self.p_idx
        self.curr_subtask = 0
        return self.get_obs(self.p_idx)
