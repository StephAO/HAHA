import json
import numpy as np
import pandas as pd
import pygame
import pylsl
from pygame import K_UP, K_LEFT, K_RIGHT, K_DOWN, K_SPACE, K_s
from pygame.locals import HWSURFACE, DOUBLEBUF, RESIZABLE, FULLSCREEN
import matplotlib
import time

matplotlib.use('TkAgg')

from os import listdir, environ, system, name
from os.path import isfile, join
import re
import time

from pathlib import Path
import pathlib
USING_WINDOWS = (name == 'nt')
# Windows path


#Tobii Pro SDK
import tobii_research as tr


if USING_WINDOWS:
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath


# Lab streaming layer
from pylsl import StreamInfo, StreamOutlet, local_clock

# Used to activate game window at game start for immediate game play
if USING_WINDOWS:
    import pygetwindow as gw

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

# from run_eyetracking_study_AJR1 import run_study

def pause():
    programPause = input("\n\n\nPress the <Enter> key to continue...")


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

class OvercookedGUI:
    """Class to run an Overcooked Gridworld game, leaving one of the agents as fixed.
    Useful for debugging. Most of the code from http://pygametutorials.wikidot.com/tutorials-basic."""
    def __init__(self, args, layout_name=None, agent=None, teammate=None, trial_id=None, user_id=None, stream=None, outlet=None, fps=5):
        self.x = None
        self._running = True
        self._display_surf = None
        self.args = args
        self.layout_name = layout_name or 'asymmetric_advantages'

        self.use_subtask_env = False
        if self.use_subtask_env:
            kwargs = {'single_subtask_id': 10, 'args': args, 'is_eval_env': True}
            self.env = OvercookedSubtaskGymEnv(**p_kwargs, **kwargs)
        else:
            self.env = OvercookedGymEnv(layout_name=self.layout_name, args=args, ret_completed_subtasks=True,
                                        is_eval_env=True, horizon=25)
        self.env.set_teammate(teammate)
        self.teammate_name=teammate.name
        self.env.reset(p_idx=0)
        self.env.teammate.set_idx(self.env.t_idx, self.layout_name, False, True, False)

        self.grid_shape = self.env.grid_shape
        self.agent = agent
        self.trial_id = trial_id
        self.user_id = user_id
        self.fps = fps

        self.score = 0
        self.curr_tick = 0
        self.tile_size = 150
        self.human_action = None
        self.data_path = args.base_dir / args.data_path
        self.data_path.mkdir(parents=True, exist_ok=True)

       # self.info_stream = StreamInfo(name="GameData", type="GameData", channel_count=1, nominal_srate=self.fps,
                                    #  channel_format='string', source_id='game')
       # self.outlet = StreamOutlet(self.info_stream)

        self.info_stream = stream
        self.outlet = outlet

        pause()


        self.collect_trajectory = False
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
        surface = StateVisualizer(tile_size=self.tile_size).render_state(self.env.state,
                                                                         grid=self.env.env.mdp.terrain_mtx,
                                                                         hud_data={"timestep": 0})
        self.surface_size = surface.get_size()
        self.x, self.y = (1920 - self.surface_size[0]) // 2, (1080 - self.surface_size[1]) // 2
        self.grid_shape = self.env.mdp.shape
        self.hud_size = self.surface_size[1] - (self.grid_shape[1] * self.tile_size)
        environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (self.x, self.y)

        self.window = pygame.display.set_mode(self.surface_size, HWSURFACE | DOUBLEBUF | RESIZABLE)
        self.window.blit(surface, (0, 0))
        print(self.x, self.y, self.surface_size, self.tile_size, self.grid_shape, self.hud_size)
        pygame.display.flip()
        self._running = True

        if USING_WINDOWS:
            win = gw.getWindowsWithTitle('pygame window')[0]
            win.activate()

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

            "state": json.dumps(prev_state.to_dict()),
            "joint_action": json.dumps(self.env.get_joint_action()),
            # TODO get teammate action from env to create joint_action json.dumps(joint_action.item()),
            "reward": curr_reward,
            "time_left": max((1200 - self.curr_tick) / self.fps, 0),
            "score": self.score,
            "time_elapsed": self.curr_tick / self.fps,
            "cur_gameloop": self.curr_tick,
            "layout": self.env.env.mdp.terrain_mtx,
            "layout_name": self.layout_name,
            "trial_id": self.trial_id,
            "user_id": self.user_id,
            "dimension": (self.x, self.y, self.surface_size, self.tile_size, self.grid_shape, self.hud_size),
            "timestamp": time.time(),
            "agent": self.teammate_name,
        }

        trans_str = json.dumps(transition)
        self.outlet.push_sample([trans_str])

        if self.collect_trajectory:
            self.trajectory.append(transition)
        return done

    def on_render(self, pidx=None):
        surface = StateVisualizer(tile_size=self.tile_size).render_state(self.env.state,
                                                                         grid=self.env.env.mdp.terrain_mtx,
                                                                         hud_data={"timestep": self.curr_tick})
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

    def predict(self, obs, state=None, episode_start=None, deterministic: bool = False):
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
        self.policy = DummyPolicy(spaces.Dict({'visual_obs': spaces.Box(0, 1, (1,))}))

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

    def predict(self, obs, state=None, episode_start=None, deterministic: bool = False):
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

    args = get_arguments(additional_args)

    # tm = load_agent(Path('agent_models/2l_hd128_s1997/ck_0/agents_dir/agent_0'), args) # 'agent_models/HAHA'
    tm = load_agent(Path('agent_models/HAHA'), args)  # 'agent_models/HAHA'
    # tm = DummyAgent()
    agent = 'human'  # load_agent(Path('agent_models/SP'), args) #'human' #HumanPlayer('agent', args)

    dc = OvercookedGUI(args, agent=agent, teammate=tm)
    dc.on_execute()
