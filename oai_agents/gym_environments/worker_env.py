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
    def __init__(self, **kwargs):
        # Add enc channel one for goal layers
        super(OvercookedSubtaskGymEnv, self).__init__(**kwargs)

    def get_overcooked_from_mdp_kwargs(self, horizon=None):
        return {'start_state_fn': self.mdp.get_subtask_start_state_fn(self.mlam), 'horizon': 100}

    def get_obs(self, p_idx, **kwargs):
        go = self.goal_objects if p_idx == self.p_idx else None
        obs = super().get_obs(p_idx, goal_objects=go,**kwargs)
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
                curr_dist = self.mlam.motion_planner.min_cost_to_feature((adj_tile, tuple(-1 * direction)), feature_locations)
            else:
                dist = self.mlam.motion_planner.min_cost_to_feature((adj_tile, tuple(-1 * direction)), feature_locations)
                if dist < smallest_dist:
                    smallest_dist = dist

        if smallest_dist == float('inf'): # No other place to pick it up -> i.e. useless
            return -0.1

        # Impossible to reach the feature from our curr spot,
        # Reward should scale by how close the best pickup spot is to the feature
        if curr_dist == float('inf'):
            # Every spot further from the pot is -1 starting at 5. Max reward of 5
            reward = max(5, 1 - (smallest_dist))
        else:
            # Reward proportional to how much time is saved from using the pass compared to walking ourselves
            # Only get additional reward if there is a worst spot that exists
            smallest_dist = min(smallest_dist, curr_dist)
            reward = (curr_dist - smallest_dist)
        return max(0, reward) # No negative reward

    def get_pickup_proximity_reward(self, feature_locations):
        # Calculate bonus reward for picking up an object on the pass.
        # Reward should be proportional to how much time is saved by using the pass
        largest_dist = 0
        agent_dir = self.state.players[self.p_idx].orientation
        object_location = np.array(self.state.players[self.p_idx].position) + np.array(agent_dir)
        for direction in [np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0])]:
            adj_tile = tuple(np.array(object_location) + direction)
            # Can't pick up from a terrain location that is not walkable
            if adj_tile not in self.mdp.get_valid_player_positions():
                continue
            if adj_tile == self.state.players[self.p_idx].position:
                curr_dist = self.mlam.motion_planner.min_cost_to_feature((adj_tile, agent_dir), feature_locations)
            else:
                dist = self.mlam.motion_planner.min_cost_to_feature((adj_tile, agent_dir), feature_locations)
                if dist != float('inf') and dist > largest_dist:
                    largest_dist = dist

        # If there is a further place that you could've picked it up form (i.e. worse place)
        # then get bonus reward for picking it up from the better place
        largest_dist = max(largest_dist, curr_dist)
        # Reward proportional to how much time is saved from using the pass
        reward = (largest_dist - curr_dist)
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

        return max(0, (chosen_pot_num_onions - other_pot_num_onions))

    def get_non_full_pot_locations(self, state):
        pot_states = self.mdp.get_pot_states(state)
        return pot_states['empty'] + pot_states['1_items'] + pot_states['2_items']

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
                    pot_locations = self.get_non_full_pot_locations(self.state)
                    reward += self.get_putdown_proximity_reward(pot_locations)
                elif self.goal_subtask == 'put_plate_closer':
                    pot_locations = self.get_non_full_pot_locations(self.state)
                    reward += self.get_putdown_proximity_reward(pot_locations)
                elif self.goal_subtask == 'put_soup_closer':
                    serving_locations = self.mdp.get_serving_locations()
                    reward += self.get_putdown_proximity_reward(serving_locations)
                elif self.goal_subtask == 'get_onion_from_counter':
                    pot_locations = self.get_non_full_pot_locations(self.state)
                    reward += self.get_pickup_proximity_reward(pot_locations)
                elif self.goal_subtask == 'get_plate_from_counter':
                    pot_locations = self.get_non_full_pot_locations(self.state)
                    reward += self.get_pickup_proximity_reward(pot_locations)
                elif self.goal_subtask == 'get_soup_from_counter':
                    serving_locations = self.mdp.get_serving_locations()
                    reward += self.get_pickup_proximity_reward(serving_locations)
                elif self.goal_subtask == 'put_onion_in_pot':
                    reward += self.get_fuller_pot_reward(self.state, self.mdp.terrain_mtx)
        return self.get_obs(self.p_idx, done=done), reward, done, info

    def reset(self, evaluation_trial_num=-1, p_idx=None):
        if evaluation_trial_num < 0:
            self.goal_subtask = np.random.choice(Subtasks.SUBTASKS)
            self.goal_subtask_id = Subtasks.SUBTASKS_TO_IDS[self.goal_subtask]
        else:
            self.goal_subtask_id = evaluation_trial_num % (Subtasks.NUM_SUBTASKS - 1) # No need to test unknown subtask
            self.goal_subtask = Subtasks.IDS_TO_SUBTASKS[self.goal_subtask_id]
        self.goal_objects = Subtasks.IDS_TO_GOAL_MARKERS[self.goal_subtask_id]

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
                self.p_idx = p_idx or np.random.randint(2)
        elif self.layout_name == 'asymmetric_advantages':
            self.p_idx = p_idx or np.random.randint(2)
        else:
            self.p_idx = p_idx or np.random.randint(2)

        self.t_idx = 1 - self.p_idx
        self.stack_frames_need_reset = [True, True]

        if evaluation_trial_num >= 0:
            counters = evaluation_trial_num % max(USEABLE_COUNTERS[self.layout_name] - 1, 1)
            ss_kwargs = {'p_idx': self.p_idx, 'random_pos': True, 'random_dir': True,
                         'curr_subtask': self.goal_subtask, 'num_random_objects': counters}
        else:
            ss_kwargs = {'p_idx': self.p_idx, 'random_pos': True, 'random_dir': True,
                         'curr_subtask': self.goal_subtask, 'max_random_objs': USEABLE_COUNTERS[self.layout_name] - 1}
        self.env.reset(start_state_kwargs=ss_kwargs)
        self.state = self.env.state
        self.prev_state = None
        if self.goal_subtask != 'unknown':
            unk_id = Subtasks.SUBTASKS_TO_IDS['unknown']
            assert get_doable_subtasks(self.state, unk_id, self.layout_name, self.terrain, self.p_idx, USEABLE_COUNTERS[self.layout_name])[self.goal_subtask_id]
        return self.get_obs(self.p_idx, on_reset=True)

    def evaluate(self, agent):
        results = np.zeros((Subtasks.NUM_SUBTASKS, 2))
        mean_reward = {}
        curr_trial, tot_trials = 0, 100 * (Subtasks.NUM_SUBTASKS - 1) # No need to test unknown subtask
        avg_steps = []
        while curr_trial < tot_trials:
            invalid_trial = False
            cum_reward, reward, done, n_steps = 0, 0, False, 0
            obs = self.reset(evaluation_trial_num=curr_trial)
            while not done:
                # If the subtask is no longer possible (e.g. other agent picked the only onion up from the counter)
                # then stop the trial and don't count it
                unk_id = Subtasks.SUBTASKS_TO_IDS['unknown']
                if not get_doable_subtasks(self.state, unk_id, self.layout_name, self.terrain, self.p_idx, USEABLE_COUNTERS[self.layout_name])[
                    self.goal_subtask_id]:
                    invalid_trial = True
                    break

                action = agent.predict(obs, deterministic=False)[0]
                obs, reward, done, info = self.step(action)
                cum_reward += reward
                n_steps += 1

            if invalid_trial:
                tot_trials -= 1
                continue

            thresh = 0.4 if (self.layout_name == 'forced_coordination' and self.p_idx == 0 and self.goal_subtask_id in [3, 6, 9]) else 0.8
            if reward >= thresh:
                results[self.goal_subtask_id][0] += 1
                avg_steps.append(n_steps)
            else:
                results[self.goal_subtask_id][1] += 1
            mean_reward[self.goal_subtask_id] = mean_reward.get(self.goal_subtask_id, []) + [cum_reward]
            curr_trial += 1

        num_succ = np.sum(results[:, 0])

        print(f'Subtask eval results on layout {self.layout_name} with teammate {self.teammate.name}.')
        for subtask_id in range(Subtasks.NUM_SUBTASKS - 1):
            print(f'{subtask_id}: mean reward of {np.mean(mean_reward[subtask_id])} -- successes: {results[subtask_id][0]}, failures: {results[subtask_id][1]}')
        return num_succ == tot_trials, np.sum(results[:, 1])# and num_succ > 10
