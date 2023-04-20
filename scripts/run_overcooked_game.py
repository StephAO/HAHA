import json
import numpy as np
import pandas as pd
from pathlib import Path
import pygame
import pylsl
from pygame import K_UP, K_LEFT, K_RIGHT, K_DOWN, K_SPACE, K_s
from pygame.locals import HWSURFACE, DOUBLEBUF, RESIZABLE, FULLSCREEN
import matplotlib
matplotlib.use('TkAgg')

from os import listdir, environ, system

from os.path import isfile, join
import re

# Windows path
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# LSL testing
from pylsl import StreamInfo, StreamOutlet, local_clock

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

from pylsl import StreamInfo, StreamOutlet, local_clock


from oai_agents.agents.base_agent import OAIAgent
from oai_agents.agents.il import BehaviouralCloningAgent
from oai_agents.agents.rl import MultipleAgentsTrainer
from oai_agents.agents.hrl import MultiAgentSubtaskWorker, HierarchicalRL
# from oai_agents.agents import Manager
from oai_agents.common.arguments import get_arguments
from oai_agents.common.subtasks import Subtasks, get_doable_subtasks
from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv
from oai_agents.agents.agent_utils import load_agent, DummyAgent
# from oai_agents.gym_environments import OvercookedSubtaskGymEnv
from oai_agents.common.state_encodings import ENCODING_SCHEMES
from overcooked_ai_py.mdp.overcooked_mdp import Direction, Action, OvercookedState, OvercookedGridworld
# from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.planning.planners import MediumLevelActionManager



def pause():
    programPause = input("Press the <Enter> key to continue...")


no_counters_params = {
    'start_orientations': False,
    'wait_allowed': False,
    'counter_goals': [],
    'counter_drop': [],
    'counter_pickup': [],
    'same_motion_goals': True
}

valid_counters = [(5, 3)]
one_counter_params = {
    'start_orientations': False,
    'wait_allowed': False,
    'counter_goals': valid_counters,
    'counter_drop': valid_counters,
    'counter_pickup': [],
    'same_motion_goals': True
}



class App:
    """Class to run an Overcooked Gridworld game, leaving one of the agents as fixed.
    Useful for debugging. Most of the code from http://pygametutorials.wikidot.com/tutorials-basic."""
    def __init__(self, args, agent=None, teammate=None, fps=5):
        self._running = True
        self._display_surf = None
        self.args = args
        self.layout_name = 'asymmetric_advantages' # counter_circuit_o_1order,coordination_ring,forced_coordination,asymmetric_advantages,cramped_room # args.layout_names[0]

        self.use_subtask_env = False
        if self.use_subtask_env:
            kwargs = {'single_subtask_id': 10, 'args': args, 'is_eval_env': True}
            self.env = OvercookedSubtaskGymEnv(**p_kwargs, **kwargs)
        else:
            self.env = OvercookedGymEnv(layout_name=self.layout_name, args=args, ret_completed_subtasks=True, is_eval_env=True)
        self.env.set_teammate(teammate)
        self.env.reset(p_idx=0)
        self.env.teammate.set_idx(self.env.t_idx, self.layout_name, False, True, False)

        self.grid_shape = self.env.grid_shape
        self.agent = agent
        self.human_action = None

        self.fps = fps

        self.score = 0
        self.curr_tick = 0
        self.data_path = args.base_dir / args.data_path
        self.data_path.mkdir(parents=True, exist_ok=True)

        self.info_stream = StreamInfo(name="GameData", type="GameData", channel_count=1, nominal_srate=self.fps,
                                      channel_format='string', source_id='game')
        self.outlet = StreamOutlet(self.info_stream)


        print("\n\nRefresh LSL streams\nSelect GameData stream\n Start LSL recording")

        pause()


        self.collect_trajectory = True
        if self.collect_trajectory:
            self.trajectory = []
            trial_file = re.compile('^.*\.[0-9]+\.pickle$')
            trial_ids = []
            for file in listdir(self.data_path):
                if isfile(join(self.data_path, file)) and trial_file.match(file):
                    trial_ids.append(int(file.split('.')[-2]))
            self.trial_id = max(trial_ids) + 1 if len(trial_ids) > 0 else 1

    def on_init(self):
        pygame.init()
        surface = StateVisualizer().render_state(self.env.state, grid=self.env.env.mdp.terrain_mtx, hud_data={"timestep": 0})
        self.window = pygame.display.set_mode(surface.get_size(), HWSURFACE | DOUBLEBUF | RESIZABLE)
        self.window.blit(surface, (0, 0))
        pygame.display.flip()
        self._running = True


    def on_event(self, event):
        if event.type == pygame.KEYDOWN:
            pressed_key = event.dict['key']
            action = None

            if pressed_key == K_UP:
                action = Direction.NORTH
            elif pressed_key == K_RIGHT:
                action = Direction.EAST
            elif pressed_key == K_DOWN:
                action = Direction.SOUTH
            elif pressed_key == K_LEFT:
                action = Direction.WEST
            elif pressed_key == K_SPACE:
                action = Action.INTERACT
            elif pressed_key == K_s:
                action = Action.STAY
            else:
                action = Action.STAY
            self.human_action = Action.ACTION_TO_INDEX[action]

        if event.type == pygame.QUIT:
            self._running = False


    def step_env(self, agent_action):
        prev_state = self.env.state

        obs, reward, done, info = self.env.step(agent_action)

        # pygame.image.save(self.window, f"screenshots/screenshot_{self.curr_tick}.png")


        # Log data to send to psiturk client
        curr_reward = sum(info['sparse_r_by_agent'])
        self.score += curr_reward
        transition = {

            "state" : json.dumps(prev_state.to_dict()),
            "joint_action" : json.dumps(self.env.get_joint_action()), # TODO get teammate action from env to create joint_action json.dumps(joint_action.item()),
            "reward" : curr_reward,
            "time_left" : max((1200 - self.curr_tick) / self.fps, 0),
            "score" : self.score,
            "time_elapsed" : self.curr_tick / self.fps,
            "cur_gameloop" : self.curr_tick,
            "layout" : self.env.env.mdp.terrain_mtx,
            "layout_name" : self.layout_name,
            "trial_id" : 100, # TODO this is just for testing self.trial_id,
            "dimension": (self.x, self.y, self.surface_size, self.tile_size, self.grid_shape, self.hud_size),
            "timestamp": time.time(),
            "user_id": 100,
        }


        trans_str = json.dumps(transition)
        self.outlet.push_sample([trans_str])

        if self.collect_trajectory:
            self.trajectory.append(transition)
        return done

    def on_render(self, pidx=None):
        surface = StateVisualizer().render_state(self.env.state, grid=self.env.env.mdp.terrain_mtx, hud_data={"timestep": self.curr_tick})
        self.window = pygame.display.set_mode(surface.get_size(), HWSURFACE | DOUBLEBUF | RESIZABLE)
        self.window.blit(surface, (0, 0))
        pygame.display.flip()
        # save = input('press y to save')
        # if save.lower() == 'y':
        #     pygame.image.save(self.window, "screenshot.png")

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False
        sleep_time = 1000 // (self.fps or 5)

        on_reset = True
        while (self._running):
            if self.agent == 'human':

                if self.human_action is None:
                    for event in pygame.event.get():
                        self.on_event(event)
                    pygame.event.pump()

                action = self.human_action if self.human_action is not None else Action.ACTION_TO_INDEX[Action.STAY]
            else:
                obs = self.env.get_obs(self.env.p_idx, on_reset=False)
                action = self.agent.predict(obs, state=self.env.state, deterministic=True)[0]
                pygame.time.wait(sleep_time)
            done = self.step_env(action)
            self.human_action = None
            pygame.time.wait(sleep_time)
            self.on_render()
            self.curr_tick += 1

            if done:
                self._running = False

        self.on_cleanup()
        print(f'Trial finished in {self.curr_tick} steps with total reward {self.score}')

    def save_trajectory(self):
        df = pd.DataFrame(self.trajectory)
        df.to_pickle(self.data_path / f'{self.layout_name}.{self.trial_id}.pickle')

    @staticmethod
    def combine_df(data_path):
        trial_file = re.compile('^.*\.[0-9]+\.pickle$')
        df = pd.concat([pd.read_pickle(data_path / f) for f in listdir(data_path) if trial_file.match(f)])
        print(f'Combined df has a length of {len(df)}')
        df.to_pickle(data_path / f'all_trials.pickle')

    @staticmethod
    def fix_files_df(data_path):
        trial_file = re.compile('^.*\.[0-9]+\.pickle$')
        for f in listdir(data_path):
            if trial_file.match(f):
                df = pd.read_pickle(data_path / f)
                def joiner(list_of_lists):
                    for i in range(len(list_of_lists)):
                        list_of_lists[i] = ''.join(list_of_lists[i])
                    return str(list_of_lists)
                df['layout'] = df['layout'].apply(joiner)
                df.to_pickle(data_path / f)

class HumanManagerHRL(OAIAgent):
    def __init__(self, worker, args):
        super(HumanManagerHRL, self).__init__('hierarchical_rl', args)
        self.worker = worker
        self.curr_subtask_id = 11
        self.prev_pcs = None

    def get_distribution(self, obs, sample=True):
        if obs['player_completed_subtasks'] is not None:
            # Completed previous subtask, set new subtask
            print(f'GOAL: {Subtasks.IDS_TO_SUBTASKS[self.curr_subtask_id]}, DONE: {obs["player_completed_subtasks"]}')
            next_st = input("Enter next subtask (0-10): ")
            self.curr_subtask_id = int(next_st)
        obs['curr_subtask'] = self.curr_subtask_id
        return self.worker.get_distribution(obs, sample=sample)

    def predict(self, obs, state=None, episode_start=None, deterministic: bool=False):
        print(obs['player_completed_subtasks'])
        if np.sum(obs['player_completed_subtasks']) == 1:
            comp_st = np.argmax(obs["player_completed_subtasks"], axis=0)
            print(f'GOAL: {Subtasks.IDS_TO_SUBTASKS[self.curr_subtask_id]}, DONE: {Subtasks.IDS_TO_SUBTASKS[comp_st]}')
            doable_st = [Subtasks.IDS_TO_SUBTASKS[idx] for idx, doable in enumerate(obs['subtask_mask']) if doable == 1]
            print('DOABLE SUBTASKS:', doable_st)
            next_st = input("Enter next subtask (0-10): ")
            self.curr_subtask_id = int(next_st)
        obs['curr_subtask'] = self.curr_subtask_id
        obs.pop('player_completed_subtasks')
        obs.pop('teammate_completed_subtasks')
        return self.worker.predict(obs, state=state, episode_start=episode_start, deterministic=True)

from oai_agents.agents.agent_utils import DummyPolicy
from gym import spaces
class HumanPlayer(OAIAgent):
    def __init__(self, name, args):
        super(HumanPlayer, self).__init__(name, args)
        self.policy = DummyPolicy(spaces.Dict({'visual_obs': spaces.Box(0,1,(1,))}))

    def get_distribution(self, obs, sample=True):
        key = input(f'{self.name} enter action:')
        if key == 'w':
            action = Direction.NORTH
        elif key == 'd':
            action = Direction.EAST
        elif key == 's':
            action = Direction.SOUTH
        elif key == 'a':
            action = Direction.WEST
        elif key == ' ':
            action = Action.INTERACT
        else:
            action = Action.STAY
        return np.array([Action.ACTION_TO_INDEX[action]])

    def predict(self, obs, state=None, episode_start=None, deterministic: bool=False):
        return self.get_distribution(obs)


if __name__ == "__main__":
    """
    Sample commands
    -> pbt
    python overcooked_interactive.py -t pbt -r pbt_simple -a 0 -s 8015
    ->
    python overcooked_interactive.py -t ppo -r ppo_sp_simple -s 386
    -> BC
    python overcooked_interactive.py -t bc -r simple_bc_test_seed4
    """

    # parser.add_argument("-t", "--type", dest="type",
    #                     help="type of run, (i.e. pbt, bc, ppo, etc)", required=True)
    # parser.add_argument("-r", "--run_dir", dest="run",
    #                     help="tag of run dir in data/*_runs/", required=True)
    additional_args = [
        ('--combine', {'action': 'store_true', 'help': 'Combine all previous trials'}),
        ('--traj-file', {'type': str, 'default': None, 'help': 'trajectory file to run'}),
        ('--agent-file', {'type': str, 'default': None, 'help': 'agent file to load'}),
    ]
    # parser.add_argument("-no_slowed", "--no_slowed_down", dest="slow",
    #                     help="Slow down time for human to simulate actual test time", action='store_false')
    # parser.add_argument("-s", "--seed", dest="seed", required=False, default=0)
    # parser.add_argument("-a", "--agent_num", dest="agent_num", default=0)
    # parser.add_argument("-i", "--idx", dest="idx", default=0)
    # parser.add_argument('--combine', action='store_true', help='Combine all previous trials')
    # parser.add_argument('--traj-file', type=str, default=None, help='trajectory file to run') # '2019_hh_trials_all.pickle'
    # parser.add_argument('--agent-file', type=str, default=None, help='trajectory file to run')

    args = get_arguments(additional_args)

    # args.layout_names = ['tf_test_4', 'tf_test_4']
    #
    # data_path = args.base_dir / args.data_path
    #
    # mat = MultipleAgentsTrainer(args, num_agents=0)
    # mat.load_agents(path=Path('./agent_models/fcp_pop/ego_pop'), tag='test')
    # teammates = mat.get_agents()
    #
    # worker = MultiAgentSubtaskWorker.load(
    #         Path('./agent_models/multi_agent_subtask_worker/final/'), args)
    # #
    # hm_hrl = HumanManagerHRL(worker, args)
    #

    #tm = load_agent(Path('agent_models/2l_hd128_s1997/ck_0/agents_dir/agent_0'), args) # 'agent_models/HAHA'
    tm = load_agent(Path('agent_models/HAHA'), args) # 'agent_models/HAHA'
    #tm = DummyAgent()
    agent = 'human' #load_agent(Path('agent_models/SP'), args) #'human' #HumanPlayer('agent', args)


    dc = App(args, agent=agent, teammate=tm)
    dc.on_execute()














    # if args.agent_file is not None:
    #     env = OvercookedGymEnv(args=args)
    #     obs = env.get_obs()
    #     visual_obs_shape = obs['visual_obs'][0].shape
    #     agent_obs_shape = obs['agent_obs'][0].shape
    #     agents = [Manager(visual_obs_shape, agent_obs_shape, 0, args),
    #               Manager(visual_obs_shape, agent_obs_shape, 1, args)]
    #     for i, agent in enumerate(agents):
    #         path = args.base_dir / 'agent_models' / 'IL_agents' / (args.agent_file + f'_p{i + 1}')
    #         agent.load(path)
    #
    #     agents[0] = 'human'
    #
    # if args.combine:
    #     App.combine_df(data_path)
    # else:
    #     layout_name = 'asymmetric_advantages'
    #     dc = App(args, traj_file=args.traj_file, agents=agents, slowmo_rate=8, )
    #     dc.on_execute()