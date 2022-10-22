from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv, USEABLE_COUNTERS
from oai_agents.common.subtasks import Subtasks, get_doable_subtasks, calculate_completed_subtask

from overcooked_ai_py.mdp.overcooked_mdp import Action, Direction

from copy import deepcopy
from gym import spaces
import numpy as np
import torch as th


class OvercookedManagerGymEnv(OvercookedGymEnv):
    def __init__(self, worker=None, **kwargs):
        kwargs['ret_completed_subtasks'] = True
        super(OvercookedManagerGymEnv, self).__init__(**kwargs)
        self.action_space = spaces.Discrete(Subtasks.NUM_SUBTASKS)
        self.worker = worker
        self.worker_failures = {k: 0 for k in range(Subtasks.NUM_SUBTASKS)}

    def get_worker_failures(self):
        failures = self.worker_failures
        self.worker_failures = {k: 0 for k in range(Subtasks.NUM_SUBTASKS)}
        return failures

    def get_low_level_obs(self, p_idx, done=False, enc_fn=None):
        enc_fn = enc_fn or self.encoding_fn
        obs = enc_fn(self.env.mdp, self.state, self.grid_shape, self.args.horizon, p_idx=p_idx)
        if p_idx == self.p_idx:
            obs['curr_subtask'] = self.curr_subtask
        if self.stack_frames[p_idx]:
            obs['visual_obs'] = np.expand_dims(obs['visual_obs'], 0)
            if self.stack_frames_need_reset[p_idx]: # On reset
                obs['visual_obs'] = self.stackedobs[p_idx].reset(obs['visual_obs'])
                self.stack_frames_need_reset[p_idx] = False
            else:
                obs['visual_obs'], _ = self.stackedobs[p_idx].update(obs['visual_obs'], np.array([done]), [{}])
            obs['visual_obs'] = obs['visual_obs'].squeeze()
        return obs

    def action_masks(self):
        return get_doable_subtasks(self.state, self.prev_st, self.layout_name, self.terrain, self.p_idx, USEABLE_COUNTERS[self.layout_name] - 1).astype(bool)

    def step(self, action):
        # Action is the subtask for subtask agent to perform
        self.curr_subtask = action.cpu() if type(action) == th.tensor else action
        joint_action = [Action.STAY, Action.STAY]
        reward, done, info = 0, False, None
        ready_for_next_subtask = False
        worker_steps = 0
        while (not ready_for_next_subtask and not done):
            joint_action[self.p_idx] = self.worker.predict(self.get_low_level_obs(self.p_idx), deterministic=False)[0]
            tm_obs = self.get_obs(self.t_idx, enc_fn=self.teammate.encoding_fn) if self.teammate.use_hrl_obs else \
                     self.get_low_level_obs(self.t_idx, enc_fn=self.teammate.encoding_fn)
            joint_action[self.t_idx] = self.teammate.predict(tm_obs, deterministic=False)[0] # self.is_eval_env
            joint_action = [Action.INDEX_TO_ACTION[a] for a in joint_action]

            # If the state didn't change from the previous timestep and the agent is choosing the same action
            # then play a random action instead. Prevents agents from getting stuck
            if self.prev_state and self.state.time_independent_equal(self.prev_state) and tuple(joint_action) == self.prev_actions:
                joint_action = [np.random.choice(Direction.ALL_DIRECTIONS), np.random.choice(Direction.ALL_DIRECTIONS)]

            self.prev_state, self.prev_actions = deepcopy(self.state), deepcopy(joint_action)
            next_state, r, done, info = self.env.step(joint_action)
            if self.shape_rewards and not self.is_eval_env:
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
                if not get_doable_subtasks(self.state, self.prev_st, self.layout_name, self.terrain, self.p_idx, USEABLE_COUNTERS[self.layout_name])[self.curr_subtask]:
                    ready_for_next_subtask = True
            if worker_steps > 25:
                ready_for_next_subtask = True
            # If subtask equals unknown, HRL agent will just STAY. This essentially forces a recheck every timestep
            # to see if any other task is possible
            if self.curr_subtask == Subtasks.SUBTASKS_TO_IDS['unknown']:
                ready_for_next_subtask = True
                self.prev_st = Subtasks.SUBTASKS_TO_IDS['unknown']
                self.unknowns_in_a_row += 1
                # If no new subtask becomes available after 25 timesteps, end round
                if self.unknowns_in_a_row > 25:
                    done = True
            else:
                self.unknowns_in_a_row = 0

            if joint_action[self.p_idx] == Action.INTERACT:
                completed_subtask = calculate_completed_subtask(self.terrain, self.prev_state, self.state, self.p_idx)
                if completed_subtask != self.curr_subtask:
                    self.worker_failures[self.curr_subtask] += 1
                ready_for_next_subtask = (completed_subtask is not None)

        return self.get_obs(self.p_idx, done=done), reward, done, info

    def reset(self):
        if self.is_eval_env:
            ss_kwargs = {'random_pos': False, 'random_dir': False, 'max_random_objs': 0}
        else:
            random_pos = (self.layout_name == 'counter_circuit_o_1order')
            ss_kwargs = {'random_pos': random_pos, 'random_dir': True, 'max_random_objs': USEABLE_COUNTERS[self.layout_name]}
        self.env.reset(start_state_kwargs=ss_kwargs)
        self.state = self.env.state
        self.prev_state = None
        self.p_idx = np.random.randint(2)
        self.t_idx = 1 - self.p_idx
        # Setup correct agent observation stacking for agents that need it
        self.stack_frames[self.p_idx] = self.main_agent_stack_frames
        if self.teammate is not None:
            self.stack_frames[self.t_idx] = self.teammate.policy.observation_space['visual_obs'].shape[0] == \
                                            (self.enc_num_channels * self.args.num_stack)
        self.stack_frames_need_reset = [True, True]
        self.curr_subtask = 0
        self.unknowns_in_a_row = 0
        # Reset subtask counts
        self.completed_tasks = [np.zeros(Subtasks.NUM_SUBTASKS), np.zeros(Subtasks.NUM_SUBTASKS)]
        return self.get_obs(self.p_idx)
