from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv, USEABLE_COUNTERS
from oai_agents.common.subtasks import Subtasks, get_doable_subtasks, calculate_completed_subtask, facing

from overcooked_ai_py.planning.planners import MediumLevelActionManager
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

from copy import deepcopy
from gym import spaces
import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import Action, Direction
import torch as th
import wandb


import math


class OvercookedSubtaskGymEnv(OvercookedGymEnv):
    def __init__(self, manager=None, **kwargs):
        self.state = None
        self.prev_state = None
        self.curr_timestep = 0
        self.manager = manager
        self.goal_subtask_id = Subtasks.SUBTASKS_TO_IDS['unknown']
        self.goal_objects = None
        self.subtask_counts = np.ones(Subtasks.NUM_SUBTASKS)
        # Add enc channel one for goal layers
        self.requires_hard_reset = True
        super(OvercookedSubtaskGymEnv, self).__init__(**kwargs)

    def set_manager(self, manager):
        self.manager = manager
        print(f'Manager set to: {self.manager}')

    # def get_overcooked_from_mdp_kwargs(self, horizon=None):
    #     return {'start_state_fn': self.mdp.get_subtask_start_state_fn(self.mlam), 'horizon': 100}

    def get_obs(self, p_idx, for_manager=False, **kwargs):
        go = self.goal_objects if (p_idx == self.p_idx and not for_manager) else None
        obs = super().get_obs(p_idx, goal_objects=go,**kwargs)
        return obs

    def get_putdown_proximity_reward(self, feature_locations):
        # Calculate bonus reward for putting an object down on the pass.
        # Reward should be proportional to how much time is saved by using the pass
        smallest_dist = float('inf')
        object_location = (self.state.players[self.p_idx].position[0] + self.state.players[self.p_idx].orientation[0],
                           self.state.players[self.p_idx].position[1] + self.state.players[self.p_idx].orientation[1])
        for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            adj_tile = (object_location[0] + direction[0], object_location[1] + direction[1])
            # Can't pick up from a terrain location that is not walkable
            if adj_tile not in self.mdp.get_valid_player_positions():
                continue
            if adj_tile == self.state.players[self.p_idx].position:
                curr_dist = self.mlam.motion_planner.min_cost_to_feature((adj_tile, self.state.players[self.p_idx].orientation), feature_locations)
            else:
                dist = self.mlam.motion_planner.min_cost_to_feature((adj_tile, (-1 * direction[0], -1 * direction[1])), feature_locations)
                if dist < smallest_dist:
                    smallest_dist = dist

        if smallest_dist == float('inf'): # No other place to pick it up -> i.e. useless
            return -0.5

        # Impossible to reach the feature from our curr spot,
        # Reward should scale by how close the best pickup spot is to the feature
        if curr_dist == float('inf'):
            # Every spot further from the pot is -.1 starting at 1. Max reward of .5
            reward = 1 - (smallest_dist * 0.1)
        else:
            # Reward proportional to how much time is saved from using the pass compared to walking ourselves
            reward = (curr_dist - smallest_dist) * 0.1
        return np.clip(reward, -0.9, 2)

    def get_pickup_proximity_reward(self, feature_locations):
        # Calculate bonus reward for picking up an object on the pass.
        # Reward should be proportional to how much time is saved by using the pass
        smallest_dist = float('inf')
        agent_dir = self.state.players[self.p_idx].orientation
        object_location = (self.state.players[self.p_idx].position[0] + agent_dir[0],
                           self.state.players[self.p_idx].position[1] + agent_dir[1])
        for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            adj_tile = (object_location[0] + direction[0], object_location[1] + direction[1])
            # Can't pick up from a terrain location that is not walkable
            if adj_tile not in self.mdp.get_valid_player_positions():
                continue
            if adj_tile == self.state.players[self.p_idx].position:
                curr_dist = self.mlam.motion_planner.min_cost_to_feature((adj_tile, agent_dir), feature_locations)
            else:
                dist = self.mlam.motion_planner.min_cost_to_feature((adj_tile, (-1 * direction[0], -1 * direction[1])), feature_locations)
                if dist < smallest_dist:
                    smallest_dist = dist

        if smallest_dist == float('inf'):
            # No other spot to pick it up
            reward = 0 
        elif curr_dist == float('inf'):
            # Impossible to reach feature anyway, distance doesn't matter
            reward = 0
        else:
            # Reward proportional to how much time is saved from using the pass compared to walking ourselves
            reward = (smallest_dist - curr_dist) * 0.1

        return np.clip(reward, -0.9, 2)

    def get_fuller_pot_reward(self, state, terrain):
        """
        Returns a reward proportional to the difference in number of onions in each pot (incentivizes putting onions
        in the fuller pot to complete soups faster)
        Assumes 2 pots
        """
        chosen_pot_loc = (state.players[self.p_idx].position[0] + state.players[self.p_idx].orientation[0],
                          state.players[self.p_idx].position[1] + state.players[self.p_idx].orientation[1])
        chosen_pot_num_onions, other_pot_num_onions = 0, 0
        for obj in state.objects.values():
            x, y = obj.position
            if obj.name == 'soup' and terrain[y][x] == 'P':
                if chosen_pot_loc[0] == x and chosen_pot_loc[1] == y:  # this is the pot the worker put the onion in
                    # -1 since one onion was just added to this pot, and we want the number before it was added
                    chosen_pot_num_onions = len(obj.ingredients) - 1
                else:  # this is the other pot
                    other_pot_num_onions = len(obj.ingredients)

        return max(0, (chosen_pot_num_onions - other_pot_num_onions) * 0.1)

    def get_non_full_pot_locations(self, state):
        pot_states = self.mdp.get_pot_states(state)
        return pot_states['empty'] + pot_states['1_items'] + pot_states['2_items']

    def base_step(self, action):
        self.joint_action = [None, None]
        self.joint_action[self.p_idx] = action
        with th.no_grad():
            if self.teammate is None:
                self.joint_action[self.t_idx] = Action.STAY
            else:
                self.joint_action[self.t_idx] = self.teammate.predict(self.get_obs(self.t_idx, enc_fn=self.teammate.encoding_fn))[0]
                self.joint_action[self.t_idx] = Action.INDEX_TO_ACTION[self.joint_action[self.t_idx]]

        # If the state didn't change from the previous timestep and the agent is choosing the same action
        # then play a random action instead. Prevents agents from getting stuck
        if self.is_eval_env:
            if self.prev_state and self.state.time_independent_equal(self.prev_state) and tuple(self.joint_action) == tuple(self.prev_actions):
                self.joint_action = [np.random.choice(range(len(Direction.ALL_DIRECTIONS))),
                                     np.random.choice(range(len(Direction.ALL_DIRECTIONS)))]
                self.joint_action = [Direction.INDEX_TO_DIRECTION[(a.squeeze() if type(a) != int else a)] for a in self.joint_action]

            self.prev_state, self.prev_actions = deepcopy(self.state), deepcopy(self.joint_action)

        self.state, reward, self.requires_hard_reset, info = self.env.step(self.joint_action)

    def get_new_subtask(self):
        doable_subtasks = get_doable_subtasks(self.state, self.goal_subtask_id, self.layout_name, self.terrain,
                                              self.p_idx, self.valid_counters,
                                              USEABLE_COUNTERS.get(self.layout_name, 5))
        while np.sum(doable_subtasks[:-1]) < 1:
            self.base_step(Action.STAY)
            if self.requires_hard_reset:
                self.base_reset()
            doable_subtasks = get_doable_subtasks(self.state, 'unknown', self.layout_name, self.terrain,
                                                  self.p_idx, self.valid_counters,
                                                  USEABLE_COUNTERS.get(self.layout_name, 5))
        doable_subtasks[-1] = 0
        if self.manager is not None:
            man_obs = self.get_obs(self.p_idx, for_manager=True)
            man_obs['subtask_mask'] = doable_subtasks
            self.goal_subtask_id = int(self.manager.predict(man_obs, deterministic=False)[0].squeeze())
        else:
            if np.sum(doable_subtasks[:-1]) >= 1:
                doable_subtasks[-1] = 0
            probs = (1 / self.subtask_counts) * doable_subtasks
            probs = probs / np.sum(probs)
            self.goal_subtask_id = np.random.choice(Subtasks.NUM_SUBTASKS, p=probs)
            self.subtask_counts[self.goal_subtask_id] += 1
        self.goal_subtask = Subtasks.IDS_TO_SUBTASKS[self.goal_subtask_id]
        self.goal_objects = Subtasks.IDS_TO_GOAL_MARKERS[self.goal_subtask_id]

    def step(self, action):
        tile_in_front = [facing(self.mdp.terrain_mtx, self.env.state.players[i]) for i in range(2)]
        prev_obj = [(self.state.players[i].held_object.name if self.state.players[i].held_object else None) for i in range(2)]

        #self.state, env_reward, self.requires_hard_reset, info = self.env.step(joint_action)
        agent_action = Action.INDEX_TO_ACTION[action] 
        self.base_step(agent_action)

        done = self.goal_subtask_id == Subtasks.SUBTASKS_TO_IDS['unknown'] or self.requires_hard_reset
        self.curr_timestep += 1

        reward = -0.01
        curr_obj = [(self.state.players[i].held_object.name if self.state.players[i].held_object else None) for i in range(2)]
        if agent_action == Action.INTERACT:
            player_completed_task = calculate_completed_subtask(prev_obj[self.p_idx], curr_obj[self.p_idx], tile_in_front[self.p_idx])

            reward = 1 if player_completed_task == self.goal_subtask_id else -1
            done = True
            if reward == 1:
                # Extra rewards to incentivize better placements
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
                elif self.goal_subtask == 'serve_soup':
                    reward += 1

        elif self.curr_timestep >= 40:
            reward = -1
            done = True

        elif self.joint_action[self.t_idx] == Action.INTERACT:
            tm_completed_task = calculate_completed_subtask(prev_obj[self.t_idx], curr_obj[self.t_idx], tile_in_front[self.t_idx])
            #if tm_completed_task is not None:
            #    reward += 0.1

            unk_id = Subtasks.SUBTASKS_TO_IDS['unknown']
            if not get_doable_subtasks(self.state, unk_id, self.layout_name, self.terrain, self.p_idx, self.valid_counters, USEABLE_COUNTERS.get(self.layout_name, 5))[self.goal_subtask_id]:
                done = True

        return self.get_obs(self.p_idx, done=done), reward, done, {}

    def base_reset(self, p_idx=None):
        if p_idx is not None:
            self.p_idx = p_idx
        elif self.reset_p_idx is not None:
            self.p_idx = self.reset_p_idx
        else:
            self.p_idx = np.random.randint(2)
        self.t_idx = 1 - self.p_idx

        self.env.reset()
        self.state = self.env.state
        self.prev_state = None
        self.stack_frames_need_reset = [True, True]

    def reset(self, p_idx=None):
        if self.requires_hard_reset:
            self.base_reset(p_idx=p_idx)
        self.get_new_subtask()
        self.curr_timestep = 0

        return self.get_obs(self.p_idx)

    def evaluate(self, agent):
        results = np.zeros((Subtasks.NUM_SUBTASKS, 2))
        mean_reward = {i: [] for i in range(Subtasks.NUM_SUBTASKS)}
        curr_trial, tot_trials = 0, 50 * (Subtasks.NUM_SUBTASKS - 1) # No need to test unknown subtask
        # avg_steps = []
        while curr_trial < tot_trials:
            invalid_trial = False
            cum_reward, reward, done, n_steps = 0, 0, False, 0
            obs = self.reset(p_idx=(curr_trial % 2))
            while not done:
                # If the subtask is no longer possible (e.g. other agent picked the only onion up from the counter)
                # then stop the trial and don't count it
                unk_id = Subtasks.SUBTASKS_TO_IDS['unknown']
                if not get_doable_subtasks(self.state, unk_id, self.layout_name, self.terrain, self.p_idx, self.valid_counters, USEABLE_COUNTERS.get(self.layout_name, 5))[
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

            if reward > 0:
                results[self.goal_subtask_id][0] += 1
            else:
                results[self.goal_subtask_id][1] += 1
            mean_reward[self.goal_subtask_id].append(reward)

            curr_trial += 1

        avg_succ_rate = []
        print(f'Subtask eval results on layout {self.layout_name} with teammate {self.teammate.name}.')
        print(f'Total completed subtasks: {np.sum(results[:, 0])}, Total failed subtasks: {np.sum(results[:-1, 1])}, Total unknown: {np.sum(results[-1])}')
        for subtask_id in range(Subtasks.NUM_SUBTASKS - 1):
            print(f'{subtask_id}: mean reward of {np.mean(mean_reward[subtask_id])} -- successes: {results[subtask_id][0]}, failures: {results[subtask_id][1]}')
            if np.sum(results[subtask_id]) != 0:
                avg_succ_rate.append( np.sum(results[subtask_id, 0]) / np.sum(results[subtask_id]) )
        return np.mean(avg_succ_rate)
