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


class OvercookedSubtaskGymEnv(OvercookedGymEnv):
    def __init__(self, grid_shape=None, single_subtask_id=None, use_curriculum=False, args=None):
        self.use_curriculum = use_curriculum
        self.use_single_subtask = single_subtask_id is not None
        assert not (use_curriculum and self.use_single_subtask)  # only one of them can be true
        if self.use_single_subtask:
            self.single_subtask, self.single_subtask_id = Subtasks.IDS_TO_SUBTASKS[single_subtask_id], single_subtask_id
        elif self.use_curriculum:
            self.curr_lvl = 0

        self.mdp = OvercookedGridworld.from_layout_name(args.layout_name)
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
        ss_fn = self.mdp.get_subtask_start_state_fn(self.mlam, random_start_pos=True, random_orientation=True,
                                                    max_objects=USEABLE_COUNTERS)
        env = OvercookedEnv.from_mdp(self.mdp, horizon=100, start_state_fn=ss_fn)
        super(OvercookedSubtaskGymEnv, self).__init__(grid_shape=grid_shape, base_env=env, args=args)
        self.obs_dict['curr_subtask'] = spaces.Discrete(Subtasks.NUM_SUBTASKS)
        self.observation_space = spaces.Dict(self.obs_dict)

    def get_obs(self, p_idx=None):
        obs = self.encoding_fn(self.env.mdp, self.state, self.grid_shape, self.args.horizon, p_idx=p_idx)
        if p_idx == self.p_idx:
            obs['curr_subtask'] = self.goal_subtask_id
        return obs

    def get_proximity_reward(self, feature_locations):
        # Calculate reward for using the pass.
        # Reward should be proportional to how much time is saved from using the pass
        smallest_dist = float('inf')
        object_location = np.array(self.state.players[self.p_idx].position) + np.array(self.state.players[self.p_idx].orientation)
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
        smallest_dist = min(smallest_dist, curr_dist + 2)
        # Reward proportional to how much time is saved from using the pass
        return (curr_dist - smallest_dist) * 0.1

    def step(self, action):
        if self.teammate is None:
            raise ValueError('set_teammate must be set called before starting game unless play_both_players is True')
        joint_action = [None, None]
        joint_action[self.p_idx] = action
        joint_action[self.t_idx] = self.teammate.predict(self.get_obs(p_idx=self.t_idx))[0]
        joint_action = [Action.INDEX_TO_ACTION[a] for a in joint_action]

        # If the state didn't change from the previous timestep and the agent is choosing the same action
        # then play a random action instead. Prevents agents from getting stuck
        if self.prev_state and self.state.time_independent_equal(self.prev_state) and tuple(joint_action) == self.prev_actions:
            joint_action = [np.random.choice(Direction.ALL_DIRECTIONS), np.random.choice(Direction.ALL_DIRECTIONS)]

        self.prev_state, self.prev_actions = deepcopy(self.state), joint_action
        next_state, _, done, info = self.env.step(joint_action)
        self.state = self.env.state

        reward = -0.01 # existence penalty
        if joint_action[self.p_idx] == Action.INTERACT:
            subtask = calculate_completed_subtask(self.mdp.terrain_mtx, self.prev_state, self.state, self.p_idx)
            done = True
            reward = 1 if subtask == self.goal_subtask_id else -1
            if self.goal_subtask == 'put_onion_closer':
                pot_locations = self.mdp.get_pot_locations()
                reward += self.get_proximity_reward(pot_locations)
            elif self.goal_subtask == 'put_plate_closer':
                pot_locations = self.mdp.get_pot_locations()
                reward += self.get_proximity_reward(pot_locations)
            elif self.goal_subtask == 'put_soup_closer':
                serving_locations = self.mdp.get_serving_locations()
                reward += self.get_proximity_reward(serving_locations)

        return self.get_obs(self.p_idx), reward, done, info

    def reset(self, evaluate=False):
        self.p_idx = np.random.randint(2)
        self.t_idx = 1 - self.p_idx
        # TODO randomly set p_idx
        if self.use_single_subtask:
            self.goal_subtask = self.single_subtask
        else:
            subtask_probs = np.ones(Subtasks.NUM_SUBTASKS)
            subtask_probs[-1] = 0
            if self.use_curriculum:
                # nothing past curr level can be selected
                subtask_probs[self.curr_lvl + 1:] = 0
            subtask_probs = subtask_mask / np.sum(subtask_mask)
            self.goal_subtask = np.random.choice(Subtasks.SUBTASKS, p=subtask_probs)
        self.goal_subtask_id = Subtasks.SUBTASKS_TO_IDS[self.goal_subtask]
        self.env.reset(start_state_kwargs={'p_idx': self.p_idx, 'curr_subtask': self.goal_subtask})
        self.state = self.env.state
        self.prev_state = None
        if self.goal_subtask != 'unknown':
            assert get_doable_subtasks(self.state, self.terrain, self.p_idx, USEABLE_COUNTERS)[self.goal_subtask_id]
        return self.get_obs(self.p_idx)

    def evaluate(self, agent):
        results = np.zeros((Subtasks.NUM_SUBTASKS, 2))
        mean_reward = []
        curr_trial, tot_trials = 0, 25
        while curr_trial < tot_trials:
            invalid_trial = False
            cum_reward, reward, done = 0, 0, False
            obs = self.reset(evaluate=True)
            while not done:
                # If the subtask is no longer possible (e.g. other agent picked the only onion up from the counter)
                # then stop the trial and don't count it
                if not get_doable_subtasks(self.state, self.terrain, self.p_idx, USEABLE_COUNTERS)[self.goal_subtask_id]:
                    invalid_trial = True
                    break
                
                action = agent.predict(obs)[0]
                obs, reward, done, info = self.step(action)
                cum_reward += reward
            
            if invalid_trial:
                tot_trials -= 1
                continue
            
            if reward >= 1:
                results[self.goal_subtask_id][0] += 1
            else:
                results[self.goal_subtask_id][1] += 1
            mean_reward.append(cum_reward)
            curr_trial += 1

        mean_reward = np.mean(mean_reward)
        num_succ = np.sum(results[:, 0])

        for subtask in Subtasks.SUBTASKS:
            subtask_id = Subtasks.SUBTASKS_TO_IDS[subtask]
            print(f'{subtask_id} - successes: {results[subtask_id][0]}, failures: {results[subtask_id][1]}')
        if self.use_curriculum and np.sum(results[:, 0]) == num_trials and self.curr_lvl < Subtasks.NUM_SUBTASKS:
            print(f'Going from level {self.curr_lvl} to {self.curr_lvl + 1}')
            self.curr_lvl += 1
        wandb.log({'subtask_reward': mean_reward, 'subtask_success': num_succ, 'timestep': agent.num_timesteps})
        return num_succ == tot_trials and num_succ > 10


