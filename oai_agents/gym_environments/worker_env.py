from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv, USEABLE_COUNTERS
from oai_agents.common.subtasks import Subtasks, get_doable_subtasks, calculate_completed_subtask

from overcooked_ai_py.planning.planners import MediumLevelActionManager
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

from copy import deepcopy
from gym import spaces
import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import Action, Direction
import wandb


import math


class OvercookedSubtaskGymEnv(OvercookedGymEnv):
    def __init__(self, single_subtask_id=None, use_curriculum=False, **kwargs):
        self.use_curriculum = use_curriculum
        self.use_single_subtask = single_subtask_id is not None
        if self.use_single_subtask:
            self.single_subtask, self.single_subtask_id = Subtasks.IDS_TO_SUBTASKS[single_subtask_id], single_subtask_id
        elif self.use_curriculum:
            self.curr_lvl = 0
        super(OvercookedSubtaskGymEnv, self).__init__(**kwargs)
        self.obs_dict['curr_subtask'] = spaces.Discrete(Subtasks.NUM_SUBTASKS)
        self.observation_space = spaces.Dict(self.obs_dict)
        assert not (use_curriculum and self.use_single_subtask)  # only one of them can be true

    def init_base_env(self, env_index=None, **kwargs):
        assert env_index is not None
        self.mdp = OvercookedGridworld.from_layout_name(self.args.layout_names[env_index])
        all_counters = self.mdp.get_counter_locations()
        COUNTERS_PARAMS = {
            'start_orientations': False,
            'wait_allowed': False,
            'counter_goals': all_counters,
            'counter_drop': all_counters,
            'counter_pickup': all_counters,
            'same_motion_goals': True
        }
        self.mlam = MediumLevelActionManager.from_pickle_or_compute(self.mdp, COUNTERS_PARAMS, force_compute=False)
        ss_fn = self.mdp.get_subtask_start_state_fn(self.mlam)
        env = OvercookedEnv.from_mdp(self.mdp, horizon=100, start_state_fn=ss_fn)
        super(OvercookedSubtaskGymEnv, self).init_base_env(env_index=env_index, base_env=env, **kwargs)

    def get_obs(self, p_idx, done=False, enc_fn=None):
        enc_fn = enc_fn or self.encoding_fn
        obs = enc_fn(self.env.mdp, self.state, self.grid_shape, self.args.horizon, p_idx=p_idx)
        if p_idx == self.p_idx:
            obs['curr_subtask'] = self.goal_subtask_id
        if self.stack_frames[p_idx]:
            obs['visual_obs'] = np.expand_dims(obs['visual_obs'], 0)
            if self.stack_frames_need_reset[p_idx]:  # On reset
                obs['visual_obs'] = self.stackedobs[p_idx].reset(obs['visual_obs'])
                self.stack_frames_need_reset[p_idx] = False
            else:
                obs['visual_obs'], _ = self.stackedobs[p_idx].update(obs['visual_obs'], np.array([done]), [{}])
            obs['visual_obs'] = obs['visual_obs'].squeeze()
        return obs

    def get_putdown_proximity_reward(self, feature_locations):
        # Calculate bonus reward for putting an object down on the pass.
        # Reward should be proportional to how much time is saved by using the pass
        smallest_dist = float('inf')
        object_location = np.array(self.state.players[self.p_idx].position) + np.array(
            self.state.players[self.p_idx].orientation)
        for direction in [np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0])]:
            adj_tile = tuple(np.array(object_location) + direction)
            # Can't pick up from a terrain location that is not walkable
            if adj_tile not in self.mdp.get_valid_player_positions():
                continue
            if adj_tile == self.state.players[self.p_idx].position:
                curr_dist = self.mlam.motion_planner.min_cost_to_feature((adj_tile, Direction.NORTH), feature_locations)
            else:
                dist = self.mlam.motion_planner.min_cost_to_feature((adj_tile, Direction.NORTH), feature_locations)
                if dist < smallest_dist:
                    smallest_dist = dist

        if smallest_dist == float('inf'): # No other place to pick it up -> i.e. useless
            return -0.1

        # Impossible to reach the feature from our curr spot,
        # Reward should scale by how close the best pickup spot is to the feature
        if curr_dist == float('inf'):
            # Every spot further from the pot is -0.1 starting at 1. Max reward of 1
            reward = max(1, 1 - (smallest_dist * 0.1))
        else:
            # Reward proportional to how much time is saved from using the pass compared to walking ourselves
            # Only get additional reward if there is a worst spot that exists
            smallest_dist = min(smallest_dist, curr_dist)
            reward = (curr_dist - smallest_dist) * 0.1
        return max(0, reward) # No negative reward

    def get_pickup_proximity_reward(self, feature_locations):
        # Calculate bonus reward for picking up an object on the pass.
        # Reward should be proportional to how much time is saved by using the pass
        largest_dist = 0
        object_location = np.array(self.state.players[self.p_idx].position) + np.array(
            self.state.players[self.p_idx].orientation)
        for direction in [np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0])]:
            adj_tile = tuple(np.array(object_location) + direction)
            # Can't pick up from a terrain location that is not walkable
            if adj_tile not in self.mdp.get_valid_player_positions():
                continue
            if adj_tile == self.state.players[self.p_idx].position:
                curr_dist = self.mlam.motion_planner.min_cost_to_feature((adj_tile, Direction.NORTH), feature_locations)
            else:
                dist = self.mlam.motion_planner.min_cost_to_feature((adj_tile, Direction.NORTH), feature_locations)
                if dist != float('inf') and dist > largest_dist:
                    largest_dist = dist

        # If there is a further place that you could've picked it up form (i.e. worse place)
        # then get bonus reward for picking it up from the better place
        largest_dist = max(largest_dist, curr_dist)
        # Reward proportional to how much time is saved from using the pass
        reward = (largest_dist - curr_dist) * 0.1
        return max(0, reward) # No negative reward

    def get_fuller_pot_reward(self, state, terrain):
        """
        Returns a reward proportional to the difference in number of onions in each pot (incentivizes putting onions
        in the fuller pot to complete soups faster)
        Assumes 2 pots
        """
        chosen_pot_loc = np.array(state.players[self.p_idx].position) + np.array(state.players[self.p_idx].orientation)
        chosen_pot_num_onions, other_pot_num_onions = 0, 0
        for obj in state.objects.values():
            x, y = obj.position
            if obj.name == 'soup' and terrain[y][x] == 'P':
                if (obj.position == chosen_pot_loc).all():  # this is the pot the worker put the onion in
                    # -1 since one onion was just added to this pot, and we want the number before it was added
                    chosen_pot_num_onions = len(obj.ingredients) - 1
                else:  # this is the other pot
                    other_pot_num_onions = len(obj.ingredients)

        return max(0, (chosen_pot_num_onions - other_pot_num_onions) * 0.1)

    def step(self, action):
        if self.teammate is None:
            raise ValueError('set_teammate must be set called before starting game.')
        joint_action = [None, None]
        joint_action[self.p_idx] = action
        joint_action[self.t_idx] = self.teammate.predict(self.get_obs(self.t_idx, enc_fn=self.teammate.encoding_fn))[0]
        joint_action = [Action.INDEX_TO_ACTION[a] for a in joint_action]

        # If the state didn't change from the previous timestep and the agent is choosing the same action
        # then play a random action instead. Prevents agents from getting stuck
        if self.prev_state and self.state.time_independent_equal(self.prev_state) and tuple(joint_action) == self.prev_actions:
            joint_action = [np.random.choice(Direction.ALL_DIRECTIONS), np.random.choice(Direction.ALL_DIRECTIONS)]

        self.prev_state, self.prev_actions = deepcopy(self.state), deepcopy(joint_action)
        next_state, _, done, info = self.env.step(joint_action)
        self.state = deepcopy(next_state)

        reward = -0.01  # existence penalty
        if joint_action[self.p_idx] == Action.INTERACT:
            subtask = calculate_completed_subtask(self.mdp.terrain_mtx, self.prev_state, self.state, self.p_idx)
            done = True
            reward = 1 if subtask == self.goal_subtask_id else -1
            if reward == 1:
                # Extra rewards to incentivize petter placements
                if self.goal_subtask == 'put_onion_closer':
                    pot_locations = self.mdp.get_pot_locations()
                    reward += self.get_putdown_proximity_reward(pot_locations)
                elif self.goal_subtask == 'put_plate_closer':
                    pot_locations = self.mdp.get_pot_locations()
                    reward += self.get_putdown_proximity_reward(pot_locations)
                elif self.goal_subtask == 'put_soup_closer':
                    serving_locations = self.mdp.get_serving_locations()
                    reward += self.get_putdown_proximity_reward(serving_locations)
                elif self.goal_subtask == 'get_onion_from_counter':
                    pot_locations = self.mdp.get_pot_locations()
                    reward += self.get_pickup_proximity_reward(pot_locations)
                elif self.goal_subtask == 'get_plate_from_counter':
                    pot_locations = self.mdp.get_pot_locations()
                    reward += self.get_pickup_proximity_reward(pot_locations)
                elif self.goal_subtask == 'get_soup_from_counter':
                    serving_locations = self.mdp.get_serving_locations()
                    reward += self.get_pickup_proximity_reward(serving_locations)
                elif self.goal_subtask == 'put_onion_in_pot':
                    reward += self.get_fuller_pot_reward(self.state, self.mdp.terrain_mtx)

        return self.get_obs(self.p_idx, done=done), reward, done, info

    def reset(self, evaluation_trial_num=-1):
        if self.use_single_subtask:
            self.goal_subtask = self.single_subtask
        else:
            subtask_probs = np.ones(Subtasks.NUM_SUBTASKS)
            subtask_probs[-1] = 0
            if self.use_curriculum:
                # nothing past curr level can be selected
                subtask_probs[self.curr_lvl + 1:] = 0
            subtask_mask = get_doable_subtasks(self.env.state, Subtasks.SUBTASKS_TO_IDS['unknown'], self.layout_name,
                                               self.terrain, self.p_idx, USEABLE_COUNTERS[self.layout_name] + 2)
            subtask_probs = subtask_mask / np.sum(subtask_mask)
            self.goal_subtask = np.random.choice(Subtasks.SUBTASKS, p=subtask_probs)
        self.goal_subtask_id = Subtasks.SUBTASKS_TO_IDS[self.goal_subtask]

        # For layouts where there are restrictions on what player can do each subtask, set the right player for the
        # goal subtask. If certain subtasks are useless for certain layouts, raise errors if trying to learn on them
        if self.goal_subtask == 'unknown': # Just so unknown doesn't break other stuff.
            self.p_idx = np.random.randint(2)
        elif self.layout_name == 'forced_coordination':
            if self.goal_subtask in ['get_onion_from_dispenser', 'get_plate_from_dish_rack']:
                self.p_idx = 1
            elif self.goal_subtask in ['put_onion_in_pot', 'get_soup', 'serve_soup']:
                self.p_idx = 0
            else:
                self.p_idx = np.random.randint(2)
        elif self.layout_name == 'asymmetric_advantages':
            self.p_idx = np.random.randint(2)
            useless_subtasks = ['put_soup_closer', 'get_soup_from_counter', 'get_onion_from_counter', 'get_plate_from_counter']
            if self.goal_subtask in useless_subtasks:
                raise ValueError(f"{useless_subtasks} are not valid subtasks for asymmetric_advantages")
        else:
            self.p_idx = np.random.randint(2)

        self.t_idx = 1 - self.p_idx
        # Setup correct agent observation stacking for agents that need it
        self.stack_frames[self.p_idx] = self.main_agent_stack_frames
        if self.teammate is not None:
            self.stack_frames[self.t_idx] = self.teammate.policy.observation_space['visual_obs'].shape[0] == \
                                            (self.enc_num_channels * self.args.num_stack)
        self.stack_frames_need_reset = [True, True]

        if evaluation_trial_num >= 0:
            counters = evaluation_trial_num % max(USEABLE_COUNTERS[self.layout_name], 1)
            random_pos = (self.layout_name == 'counter_circuit_o_1order')
            ss_kwargs = {'p_idx': self.p_idx, 'random_pos': random_pos, 'random_dir': True,
                         'curr_subtask': self.goal_subtask, 'num_random_objects': counters}
        else:
            random_pos = (self.layout_name == 'counter_circuit_o_1order')
            ss_kwargs = {'p_idx': self.p_idx, 'random_pos': random_pos, 'random_dir': True,
                         'curr_subtask': self.goal_subtask, 'max_random_objs': USEABLE_COUNTERS[self.layout_name]}
        self.env.reset(start_state_kwargs=ss_kwargs)
        self.state = self.env.state
        self.prev_state = None
        if self.goal_subtask != 'unknown':
            unk_id = Subtasks.SUBTASKS_TO_IDS['unknown']
            assert get_doable_subtasks(self.state, unk_id, self.layout_name, self.terrain, self.p_idx, USEABLE_COUNTERS[self.layout_name] + 2)[
                self.goal_subtask_id]
        return self.get_obs(self.p_idx)

    def evaluate(self, agent):
        results = np.zeros((Subtasks.NUM_SUBTASKS, 2))
        mean_reward = []
        curr_trial, tot_trials = 0, 100
        avg_steps = []
        while curr_trial < tot_trials:
            invalid_trial = False
            cum_reward, reward, done, n_steps = 0, 0, False, 0
            obs = self.reset(evaluation_trial_num=curr_trial)
            while not done:
                # If the subtask is no longer possible (e.g. other agent picked the only onion up from the counter)
                # then stop the trial and don't count it
                unk_id = Subtasks.SUBTASKS_TO_IDS['unknown']
                if not get_doable_subtasks(self.state, unk_id, self.layout_name, self.terrain, self.p_idx, USEABLE_COUNTERS[self.layout_name] + 2)[
                    self.goal_subtask_id]:
                    invalid_trial = True
                    break

                action = agent.predict(obs, deterministic=True)[0]
                obs, reward, done, info = self.step(action)
                cum_reward += reward
                n_steps += 1

            if invalid_trial:
                tot_trials -= 1
                continue

            thresh = 0.4 if (self.layout_name == 'forced_coordination' and self.p_idx == 0 and self.goal_subtask_id in [3, 6, 9]) else 1
            if reward >= thresh:
                results[self.goal_subtask_id][0] += 1
                avg_steps.append(n_steps)
            else:
                results[self.goal_subtask_id][1] += 1
            mean_reward.append(cum_reward)
            curr_trial += 1

        mean_reward = np.mean(mean_reward)
        num_succ = np.sum(results[:, 0])

        print(f'Subtask eval results on layout {self.layout_name} with teammate {self.teammate.name}.')
        print(f'Steps taken, avg: {np.mean(avg_steps)}, min: {np.min(avg_steps)}, max: {np.max(avg_steps)}')
        # for subtask in Subtasks.SUBTASKS:
        subtask_id = self.goal_subtask_id
        print(f'Mean reward: {mean_reward}')
        print(f'{subtask_id} - successes: {results[subtask_id][0]}, failures: {results[subtask_id][1]}')
        if self.use_curriculum and np.sum(results[:, 0]) == num_trials and self.curr_lvl < Subtasks.NUM_SUBTASKS:
            print(f'Going from level {self.curr_lvl} to {self.curr_lvl + 1}')
            self.curr_lvl += 1
        # wandb.log({f'st_rew_{self.teammate.name}_{self.layout_name}': mean_reward, f'st_succ_{self.teammate.name}_{self.layout_name}': num_succ, 'timestep': agent.num_timesteps})
        return num_succ == tot_trials, np.sum(results[:, 1])# and num_succ > 10
