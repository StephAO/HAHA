from gym import spaces
import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import Direction, Action

from oai_agents.agents.base_agent import OAIAgent
from oai_agents.agents.agent_utils import DummyPolicy
from oai_agents.common.subtasks import Subtasks

class HumanManagerHRL(OAIAgent):
    def __init__(self, worker, args):
        super(HumanManagerHRL, self).__init__('hierarchical_rl', args)
        self.worker = worker
        self.policy = self.worker.policy
        self.curr_subtask_id = 11
        self.prev_pcs = None
        self.action_id = 0

    def get_distribution(self, obs, sample=True):
        # Completed previous subtask, set new subtask
        next_st = input("Enter next subtask (0-10): ")
        self.curr_subtask_id = int(next_st)
        obs['curr_subtask'] = self.curr_subtask_id
        return self.worker.get_distribution(obs, sample=sample)

    def predict(self, obs, state=None, episode_start=None, deterministic: bool = False):
        if (self.action_id and Action.INDEX_TO_ACTION[int(self.action_id.squeeze())] == Action.INTERACT) or self.curr_subtask_id == 11:
            doable_st = [Subtasks.IDS_TO_SUBTASKS[idx] for idx, doable in enumerate(obs['subtask_mask']) if doable == 1]
            print('DOABLE SUBTASKS:', [(dst, Subtasks.SUBTASKS_TO_IDS[dst]) for dst in doable_st])
            next_st = input("Enter next subtask (0-10): ")
            self.curr_subtask_id = int(next_st)
        worker_obs = self.obs_fn(p_idx=self.p_idx, goal_objects=Subtasks.IDS_TO_GOAL_MARKERS[self.curr_subtask_id])
        self.action_id, _ = self.worker.predict(worker_obs, state=state, episode_start=episode_start, deterministic=True)
        return self.action_id, None


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
