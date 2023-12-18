from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv, USEABLE_COUNTERS
from oai_agents.common.subtasks import Subtasks, get_doable_subtasks, calculate_completed_subtask, facing

from overcooked_ai_py.mdp.overcooked_mdp import Action, Direction

from copy import deepcopy
from gym import spaces
import numpy as np
import torch as th


class OvercookedManagerGymEnv(OvercookedGymEnv):
    def __init__(self, worker=None, **kwargs):
        kwargs['ret_completed_subtasks'] = True
        # Use worker num encoding channel to get worker obs easier
        super(OvercookedManagerGymEnv, self).__init__(**kwargs)
        self.action_space = spaces.Discrete(Subtasks.NUM_SUBTASKS)
        self.worker = worker
        self.worker_failures = {k: 0 for k in range(Subtasks.NUM_SUBTASKS)}
        self.base_env_timesteps = 0

    def get_obs(self, p_idx, for_worker=False, **kwargs):
        goal_objects = Subtasks.IDS_TO_GOAL_MARKERS[self.curr_subtask] if (p_idx == self.p_idx and for_worker) else None
        return super().get_obs(p_idx, goal_objects=goal_objects, **kwargs)

    def get_base_env_timesteps(self):
        return self.base_env_timesteps

    def get_worker_failures(self):
        failures = self.worker_failures
        self.worker_failures = {k: 0 for k in range(Subtasks.NUM_SUBTASKS)}
        return (self.layout_name, failures)

    def action_masks(self, p_idx=None):
        p_idx = p_idx or self.p_idx
        return get_doable_subtasks(self.state, self.prev_subtask[p_idx], self.layout_name, self.terrain, p_idx, self.valid_counters, USEABLE_COUNTERS.get(self.layout_name, 5)).astype(bool)

    def step(self, action):
        reward = 0
        # Action is the subtask for subtask agent to perform
        self.curr_subtask = action.cpu() if type(action) == th.tensor else action
        joint_action = [Action.STAY, Action.STAY]
        ready_for_next_subtask, done, worker_steps = False, False, 0
        # while (not ready_for_next_subtask and not done):
        if self.curr_subtask != 11:
            obs = {k: v for k, v in self.get_obs(self.p_idx, for_worker=True).items() if k in self.worker.policy.observation_space.keys()}
            with th.no_grad():
                joint_action[self.p_idx] = Action.INDEX_TO_ACTION[self.worker.predict(obs, deterministic=False)[0]]
        else: # unknown subtask, just noop
            # Keep no-op action
            self.prev_subtask[self.p_idx] = Subtasks.SUBTASKS_TO_IDS['unknown']
            if not self.is_eval_env:
                reward -= 0.1
            ready_for_next_subtask = True

        tm_obs = self.get_obs(self.t_idx, enc_fn=self.teammate.encoding_fn)
            # if self.teammate.use_hrl_obs else self.get_low_level_obs(self.t_idx, enc_fn=self.teammate.encoding_fn)
        with th.no_grad():
            joint_action[self.t_idx] = Action.INDEX_TO_ACTION[self.teammate.predict(tm_obs, deterministic=False)[0]] # self.is_eval_env

        # If the state didn't change from the previous timestep and the agent is choosing the same action
        # then play a random action instead. Prevents agents from getting stuck
        if self.is_eval_env:
            if self.prev_state and self.state.time_independent_equal(self.prev_state) and tuple(joint_action) == tuple(self.prev_actions):
                joint_action = [np.random.choice(range(len(Direction.ALL_DIRECTIONS))),
                                np.random.choice(range(len(Direction.ALL_DIRECTIONS)))]
                joint_action = [Direction.INDEX_TO_DIRECTION[(a.squeeze() if type(a) != int else a)] for a in joint_action]
            self.prev_state, self.prev_actions = deepcopy(self.state), deepcopy(joint_action)

        tile_in_front = facing(self.mdp.terrain_mtx, self.env.state.players[self.p_idx])
        prev_obj = self.state.players[self.p_idx].held_object.name if self.state.players[self.p_idx].held_object else None

        self.state, r, done, info = self.env.step(joint_action)
        reward += r
        self.step_count += 1
        worker_steps += 1
        if worker_steps > 10:
            ready_for_next_subtask = True

        if joint_action[self.p_idx] == Action.INTERACT:
            ready_for_next_subtask = True
            curr_obj = self.state.players[self.p_idx].held_object.name if self.state.players[self.p_idx].held_object else None
            completed_subtask = calculate_completed_subtask(prev_obj, curr_obj, tile_in_front)
            if completed_subtask != self.curr_subtask:
                self.worker_failures[self.curr_subtask] += 1
                self.failures_in_a_row += 1
                if self.failures_in_a_row >= 5 and not self.is_eval_env:
                    done = True
            else:
                self.failures_in_a_row = 0

        return self.get_obs(self.p_idx, done=done), reward, done, info

    def reset(self, p_idx=None):
        self.env.reset()
        self.state = self.env.state
        self.prev_state = None
        self.prev_subtask = [Subtasks.SUBTASKS_TO_IDS['unknown'], Subtasks.SUBTASKS_TO_IDS['unknown']]

        if p_idx is not None:
            self.p_idx = p_idx
        elif self.reset_p_idx is not None:
            self.p_idx = self.reset_p_idx
        else:
            self.p_idx = np.random.randint(2)
        self.t_idx = 1 - self.p_idx
        self.stack_frames_need_reset = [True, True]
        self.curr_subtask = 0
        self.unknowns_in_a_row = 0
        self.failures_in_a_row = 0
        # Reset subtask counts
        self.completed_tasks = [np.zeros(Subtasks.NUM_SUBTASKS), np.zeros(Subtasks.NUM_SUBTASKS)]
        return self.get_obs(self.p_idx, on_reset=True)
