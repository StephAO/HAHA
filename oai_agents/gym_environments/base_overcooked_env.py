from oai_agents.common.state_encodings import ENCODING_SCHEMES
from oai_agents.common.subtasks import Subtasks, calculate_completed_subtask, get_doable_subtasks

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelActionManager
from overcooked_ai_py.utils import read_layout_dict
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

from copy import deepcopy
from gym import Env, spaces, register
import numpy as np
import pygame
from pygame.locals import HWSURFACE, DOUBLEBUF, RESIZABLE
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations

# DEPRECATED NOTE: For counter circuit, trained workers with 8, but trained manager with 4. Only 4 spots are useful add
# more during subtask worker training for robustness
# Max number of counters the agents should use
# USEABLE_COUNTERS = {'counter_circuit_o_1order': 8, 'forced_coordination': 5, 'asymmetric_advantages': 2, 'cramped_room': 5, 'coordination_ring': 5} # FOR WORKER TRAINING
# USEABLE_COUNTERS = {'counter_circuit_o_1order': 6, 'forced_coordination': 3, 'asymmetric_advantages': 2, 'cramped_room': 4, 'coordination_ring': 4} # FOR MANAGER TRAINING
USEABLE_COUNTERS = {'counter_circuit_o_1order': 4, 'forced_coordination': 3, 'asymmetric_advantages': 2, 'cramped_room': 3, 'coordination_ring': 3}  # FOR EVALUATION AND SP TRAINING


# Calculated by dividing the average overall reward by the average reward on each layout in the 2019 human trials
SCALING_FACTORS = {
    'asymmetric_advantages': 0.612,
    'coordination_ring': 1.116,
    'cramped_room': 0.946,
    'forced_coordination': 1.371,
    'counter_circuit_o_1order': 1.461
}

class OvercookedGymEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_shape=None, ret_completed_subtasks=False, stack_frames=False, is_eval_env=False,
                 shape_rewards=False, enc_fn=None, full_init=True, args=None, **kwargs):
        self.is_eval_env = is_eval_env
        self.args = args
        self.device = args.device
        # Observation encoding setup
        enc_fn = enc_fn or args.encoding_fn
        self.encoding_fn =  ENCODING_SCHEMES[enc_fn]
        if enc_fn == 'OAI_egocentric':
            # Override grid shape to make it egocentric
            assert grid_shape is None, 'Grid shape cannot be used when egocentric encodings are used!'
            self.grid_shape = (7, 7)
        elif grid_shape is None:
            base_layout_params = read_layout_dict(args.layout_names[0])
            grid = [layout_row.strip() for layout_row in base_layout_params['grid'].split("\n")]
            self.grid_shape = (len(grid[0]), len(grid))
        else:
            self.grid_shape = grid_shape

        # Set Sp Observation Space
        # Currently 20 is the default value for recipe time (which I believe is the largest value used in encoding)
        self.enc_num_channels = 26 # Default channels of OAI_Lossless encoding
        self.obs_dict = {}
        if enc_fn == 'OAI_feats':
            self.obs_dict['agent_obs'] = spaces.Box(0, 400, (96,), dtype=np.int)
        else:
            self.obs_dict['visual_obs'] = spaces.Box(0, 20, (self.enc_num_channels, *self.grid_shape), dtype=np.int)
            # Stacked obs for main player (index 0) and teammate (index 1)
            self.stackedobs = [StackedObservations(1, args.num_stack, self.obs_dict['visual_obs'], 'first'),
                               StackedObservations(1, args.num_stack, self.obs_dict['visual_obs'], 'first')]
        if stack_frames:
            self.obs_dict['visual_obs'] = self.stackedobs[0].stack_observation_space(self.obs_dict['visual_obs'])

        if ret_completed_subtasks:
            self.obs_dict['player_completed_subtasks'] = spaces.Box(-1, 100, (Subtasks.NUM_SUBTASKS,), dtype=np.int)
            self.obs_dict['teammate_completed_subtasks'] = spaces.Box(-1, 100, (Subtasks.NUM_SUBTASKS,), dtype=np.int)
            self.obs_dict['subtask_mask'] = spaces.MultiBinary(Subtasks.NUM_SUBTASKS)
        self.obs_dict['layout_idx'] = spaces.MultiBinary(5)
        self.obs_dict['p_idx'] = spaces.MultiBinary(2)
        self.observation_space = spaces.Dict(self.obs_dict)
        self.return_completed_subtasks = ret_completed_subtasks
        # Default stack frames to false since we don't currently know who is playing what - properly set in reset
        self.main_agent_stack_frames = stack_frames
        self.stack_frames = [False, False]
        self.stack_frames_need_reset = [True, True]
        # Set up Action Space
        self.action_space = spaces.Discrete(len(Action.ALL_ACTIONS))

        self.shape_rewards = shape_rewards
        self.visualization_enabled = False
        self.step_count = 0
        self.reset_p_idx = None
        self.teammate = None
        self.p_idx = None
        self.joint_action = [None, None]
        if full_init:
            self.init_base_env(**kwargs)

    # TODO rename
    def init_base_env(self, env_index=None, layout_name=None, base_env=None, horizon=None):
        '''
        :param shape_rewards: Shape rewards for RL
        :param base_env: Base overcooked environment. If None, create env from layout name. Useful if special parameters
                         are required when creating the environment
        :param horizon: How many steps to run the env for. If None, default to args.horizon value
        :param args: Experiment arguments (see arguments.py)
        '''
        assert env_index is not None or layout_name is not None or base_env is not None

        if base_env is None:
            self.env_idx = env_index
            self.layout_name = layout_name or self.args.layout_names[env_index]
            self.mdp = OvercookedGridworld.from_layout_name(self.layout_name)
            horizon = horizon or self.args.horizon
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
            # To ensure that an agent is on both sides of the counter, remove random starts for forced coord
            ss_fn = self.mdp.get_fully_random_start_state_fn(self.mlam)
            self.env = OvercookedEnv.from_mdp(self.mdp, horizon=horizon, start_state_fn=ss_fn)
        else:
            self.env = base_env
            self.layout_name = self.env.mdp.layout_name
            self.env_idx = self.args.layout_names.index(self.layout_name)


        self.terrain = self.mdp.terrain_mtx
        self.prev_subtask = [Subtasks.SUBTASKS_TO_IDS['unknown'], Subtasks.SUBTASKS_TO_IDS['unknown']]
        obs = self.reset()

    def get_layout_name(self):
        return self.layout_name

    def get_joint_action(self):
        return self.joint_action

    def set_teammate(self, teammate):
        # TODO asser has attribute observation space
        self.teammate = teammate
        self.stack_frames[self.p_idx] = self.main_agent_stack_frames
        # TODO Get rid of magic numbers
        self.stack_frames[self.t_idx] = self.teammate.policy.observation_space['visual_obs'].shape[0] == \
                                        (self.enc_num_channels * self.args.num_stack)
        self.stack_frames_need_reset = [True, True]

    def setup_visualization(self):
        self.visualization_enabled = True
        pygame.init()
        surface = StateVisualizer().render_state(self.state, grid=self.env.mdp.terrain_mtx)
        self.window = pygame.display.set_mode(surface.get_size(), HWSURFACE | DOUBLEBUF | RESIZABLE)
        self.window.blit(surface, (0, 0))
        pygame.display.flip()

    def action_masks(self, p_idx):
        return get_doable_subtasks(self.state, self.prev_subtask[p_idx], self.layout_name, self.terrain, p_idx, USEABLE_COUNTERS[self.layout_name]).astype(bool)

    def get_obs(self, p_idx, done=False, enc_fn=None, on_reset=False):
        enc_fn = enc_fn or self.encoding_fn
        obs = enc_fn(self.env.mdp, self.state, self.grid_shape, self.args.horizon, p_idx=p_idx)
        obs['layout_idx'] = np.eye(5)[self.env_idx or 0]
        obs['layout_idx'] = obs['layout_idx'].astype(bool)
        obs['p_idx'] = np.eye(2)[self.env_idx]
        obs['p_idx'] = obs['p_idx'].astype(bool)

        if self.stack_frames[p_idx]:
            obs['visual_obs'] = np.expand_dims(obs['visual_obs'], 0)
            if self.stack_frames_need_reset[p_idx]: # On reset
                obs['visual_obs'] = self.stackedobs[p_idx].reset(obs['visual_obs'])
                self.stack_frames_need_reset[p_idx] = False
            else:
                obs['visual_obs'], _ = self.stackedobs[p_idx].update(obs['visual_obs'], np.array([done]), [{}])
            obs['visual_obs'] = obs['visual_obs'].squeeze()
        if self.return_completed_subtasks or (self.teammate is not None and p_idx == self.t_idx and 'player_completed_subtasks' in self.teammate.policy.observation_space.keys()):
            # If this isn't the first step of the game, see if a subtask has been completed
            comp_st = None
            if on_reset:
                comp_st = Subtasks.NUM_SUBTASKS - 1
            elif self.prev_state is not None:
                comp_st = calculate_completed_subtask(self.terrain, self.prev_state, self.state, p_idx)
                # If a subtask has been completed, update counts
                if comp_st is not None:
                    self.completed_tasks[p_idx] = np.eye(Subtasks.NUM_SUBTASKS)[comp_st]
                    self.prev_subtask[p_idx] = comp_st
                else:
                    self.completed_tasks[p_idx] = np.zeros(Subtasks.NUM_SUBTASKS)
                    self.prev_subtask[p_idx] = Subtasks.SUBTASKS_TO_IDS['unknown']
            obs['player_completed_subtasks'] =  self.completed_tasks[p_idx]
            obs['teammate_completed_subtasks'] = self.completed_tasks[1 - p_idx]
            obs['subtask_mask'] = self.action_masks(p_idx)
            if p_idx == self.t_idx:
                obs = {k: v for k, v in obs.items() if k in self.teammate.policy.observation_space.keys()}
            else:
                obs = {k: v for k, v in obs.items() if k in self.observation_space.keys()}

        return obs

    def step(self, action):
        if self.teammate is None:
            raise ValueError('set_teammate must be set called before starting game.')

        joint_action = [None, None]
        joint_action[self.p_idx] = action
        tm_obs = self.get_obs(p_idx=self.t_idx, enc_fn=self.teammate.encoding_fn)
        joint_action[self.t_idx] = self.teammate.predict(tm_obs, deterministic=False)[0]
        joint_action = [Action.INDEX_TO_ACTION[(a.squeeze() if type(a) != int else a)] for a in joint_action]
        self.joint_action = joint_action

        # If the state didn't change from the previous timestep and the agent is choosing the same action
        # then play a random action instead. Prevents agents from getting stuck
        if self.prev_state and self.state.time_independent_equal(self.prev_state) and tuple(joint_action) == self.prev_actions:
            joint_action = [np.random.choice(Direction.ALL_DIRECTIONS), np.random.choice(Direction.ALL_DIRECTIONS)]

        self.prev_state, self.prev_actions = deepcopy(self.state), deepcopy(joint_action)

        next_state, reward, done, info = self.env.step(joint_action)
        self.state = self.env.state
        if self.shape_rewards and not self.is_eval_env:
            ratio = min(self.step_count * self.args.n_envs / 1e7, 1)
            sparse_r = sum(info['sparse_r_by_agent'])
            shaped_r = info['shaped_r_by_agent'][self.p_idx] if self.p_idx else sum(info['shaped_r_by_agent'])
            reward = sparse_r * ratio + shaped_r * (1 - ratio)

        self.step_count += 1
        return self.get_obs(self.p_idx, done=done), reward, done, info

    def set_reset_p_idx(self, p_idx):
        self.reset_p_idx = p_idx

    def reset(self, p_idx=None):
        if p_idx is not None:
            self.p_idx = p_idx
        elif self.reset_p_idx is not None:
            self.p_idx = self.reset_p_idx
        else:
            self.p_idx = np.random.randint(2)
        self.t_idx = 1 - self.p_idx
        # Setup correct agent observation stacking for agents that need it
        self.stack_frames[self.p_idx] = self.main_agent_stack_frames
        if self.teammate is not None:
            self.stack_frames[self.t_idx] = self.teammate.policy.observation_space['visual_obs'].shape[0] == \
                                            (self.enc_num_channels * self.args.num_stack)
        self.stack_frames_need_reset = [True, True]

        if self.is_eval_env:
            ss_kwargs = {'random_pos': False, 'random_dir': False, 'max_random_objs': 0}
        else:
            ss_kwargs = {'random_pos': True, 'random_dir': True, 'max_random_objs': USEABLE_COUNTERS[self.layout_name]}
        self.env.reset(start_state_kwargs=ss_kwargs)
        self.prev_state = None
        self.state = self.env.state
        # Reset subtask counts
        self.completed_tasks = [np.zeros(Subtasks.NUM_SUBTASKS), np.zeros(Subtasks.NUM_SUBTASKS)]
        return self.get_obs(self.p_idx, on_reset=True)

    def render(self, mode='human', close=False):
        if self.visualization_enabled:
            surface = StateVisualizer().render_state(self.state, grid=self.env.mdp.terrain_mtx)
            self.window = pygame.display.set_mode(surface.get_size(), HWSURFACE | DOUBLEBUF | RESIZABLE)
            self.window.blit(surface, (0, 0))
            pygame.display.flip()
            pygame.time.wait(200)

    def close(self):
        pygame.quit()


register(
    id='OvercookedGymEnv-v0',
    entry_point='OvercookedGymEnv'
)


if __name__ == '__main__':
    from oai_agents.common.arguments import get_arguments
    args = get_arguments()
    env = OvercookedGymEnv(p1=DummyAgent(), args=args) #make('overcooked_ai.agents:OvercookedGymEnv-v0', layout='asymmetric_advantages', encoding_fn=encode_state, args=args)
    print(check_env(env))
    env.setup_visualization()
    env.reset()
    env.render()
    done = False
    while not done:
        obs, reward, done, info = env.step( Action.ACTION_TO_INDEX[np.random.choice(Action.ALL_ACTIONS)] )
        env.render()
