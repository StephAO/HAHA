from oai_agents.agents.base_agent import OAIAgent, PolicyClone
from oai_agents.agents.il import BehavioralCloningTrainer
from oai_agents.agents.rl import RLAgentTrainer, SB3Wrapper, SB3LSTMWrapper, VEC_ENV_CLS
from oai_agents.agents.agent_utils import DummyAgent, is_held_obj, load_agent
from oai_agents.common.arguments import get_arguments, get_args_to_save, set_args_from_load
from oai_agents.common.subtasks import Subtasks
from oai_agents.gym_environments.worker_env import OvercookedSubtaskGymEnv
from oai_agents.gym_environments.manager_env import OvercookedManagerGymEnv

from overcooked_ai_py.mdp.overcooked_mdp import Action, OvercookedGridworld

from copy import deepcopy
import numpy as np
from pathlib import Path
from stable_baselines3.common.env_util import make_vec_env
import torch as th
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from typing import Tuple, List

class RLManagerTrainer(RLAgentTrainer):
    ''' Train an RL agent to play with a provided agent '''
    def __init__(self, worker, teammates, args, use_frame_stack=False, use_subtask_counts=False,
                 inc_sp=False, use_policy_clone=False, name=None, seed=None):
        name = name or 'hrl_manager'
        name += ('_sp' if inc_sp else '') + ('_pc' if use_policy_clone else '')
        n_layouts = len(args.layout_names)
        env_kwargs = {'worker': worker, 'shape_rewards': False, 'stack_frames': use_frame_stack, 'full_init': False, 'args': args}
        env = make_vec_env(OvercookedManagerGymEnv, n_envs=args.n_envs, env_kwargs=env_kwargs, vec_env_cls=VEC_ENV_CLS)

        eval_envs_kwargs = {'worker': worker, 'shape_rewards': False, 'stack_frames': use_frame_stack,
                            'is_eval_env': True, 'horizon': 400, 'args': args}
        eval_envs = [OvercookedManagerGymEnv(**{'env_index': i, **eval_envs_kwargs}) for i in range(n_layouts)]

        self.worker = worker
        super(RLManagerTrainer, self).__init__(teammates, args, name=name, env=env,
                                               eval_envs=eval_envs, use_subtask_counts=use_subtask_counts,
                                               use_hrl=True, use_policy_clone=use_policy_clone,
                                               seed=seed)
        # COMMENTED CODE BELOW IS TO ADD SELFPLAY
        # However, currently it's just a reference to the agent, so the "self-teammate" could update the main agents subtask
        # To do this correctly, the "self-teammate" would have to be cloned before every epoch
        if inc_sp:
            playable_self = HierarchicalRL(self.worker, self.learning_agent, self.args, name=f'playable_self')
            for i in range(3):
                if self.use_policy_clone:
                    manager = PolicyClone(self.learning_agent, self.args)
                    playable_self = HierarchicalRL(self.worker, manager, self.args, name=f'playable_self_{i}')
                self.teammates.append(playable_self)

            if type(self.eval_teammates) == dict :
                if self.use_policy_clone:
                    manager = PolicyClone(self.learning_agent, self.args)
                    playable_self = HierarchicalRL(self.worker, manager, self.args, name=f'playable_self')
                for k in self.eval_teammates:
                    self.eval_teammates[k].append(playable_self)
            elif self.eval_teammates is not None:
                if self.use_policy_clone:
                    manager = PolicyClone(self.learning_agent, self.args)
                    playable_self = HierarchicalRL(self.worker, manager, self.args, name=f'playable_self')
                self.eval_teammates.append(playable_self)

        if self.eval_teammates is None:
            self.eval_teammates = self.teammates

    def update_pc(self, epoch):
        if not self.use_policy_clone:
            return
        for tm in self.teammates:
            idx = epoch % 3
            if tm.name == f'playable_self_{idx}':
                tm.manager = PolicyClone(self.learning_agent, self.args)
        if type(self.eval_teammates) == dict:
            pc = PolicyClone(self.learning_agent, self.args, name=f'playable_self')
            for k in self.eval_teammates:
                for tm in self.eval_teammates[k]:
                    if tm.name == 'playable_self':
                        tm.manager = pc
        elif self.eval_teammates is not None:
            for i, tm in enumerate(self.eval_teammates):
                if tm.name == 'playable_self':
                    tm.manager = PolicyClone(self.learning_agent, self.args, name=f'playable_self')


class HierarchicalRL(OAIAgent):
    def __init__(self, worker, manager, args, name=None):
        name = name or 'haha'
        super(HierarchicalRL, self).__init__(name, args)
        self.worker = worker
        self.manager = manager
        self.policy = self.manager.policy
        self.num_steps_since_new_subtask = 0
        self.use_hrl_obs = True
        self.layout_name = None
        self.subtask_step = 0
        self.output_message = True
        self.tune_subtasks = None
        self.curr_subtask_id = Subtasks.SUBTASKS_TO_IDS['unknown']

    def set_play_params(self, output_message, tune_subtasks):
        self.output_message = output_message
        self.tune_subtasks = tune_subtasks
        self.subtask_step = 0
        self.waiting_steps = 0

    def get_distribution(self, obs, sample=True):
        if np.sum(obs['player_completed_subtasks']) == 1:
            # Completed previous subtask, set new subtask
            self.curr_subtask_id = self.manager.predict(obs, sample=sample)[0]
        worker_obs = self.obs_closure_fn(p_idx=self.p_idx, goal_objects=Subtasks.IDS_TO_GOAL_MARKERS[self.curr_subtask_id])
        obs.update(worker_obs)
        return self.worker.get_distribution(obs, sample=sample)

    def adjust_distributions(self, probs, indices, weights):
        new_probs = np.copy(probs.cpu()) if type(probs) == th.Tensor else np.copy(probs)
        if np.sum(new_probs[indices]) > (1 - 1e-12) or np.sum(new_probs[indices]) < 1e-12:
            # print("Agent is too decisive, no behavior changed", flush=True)
            return new_probs
        original_values = np.zeros_like(new_probs)
        adjusted_values = np.zeros_like(new_probs)
        for i, idx in enumerate(indices):
            original_values = new_probs[idx]
            adjusted_values[idx] = new_probs[idx] * weights[i]
            new_probs[idx] = 0
        if np.sum(adjusted_values) > 1:
            adjusted_values = adjusted_values / np.sum(adjusted_values)
        if np.sum(original_values) > 1:
            original_values = original_values / np.sum(original_values)
        new_probs = new_probs - (np.sum(adjusted_values) - np.sum(original_values)) * new_probs / np.sum(new_probs)
        new_probs = np.clip(new_probs, 0, None)
        for idx in indices:
            new_probs[idx] = adjusted_values[idx]
        return new_probs

    def other_player_has_plate(self, obs):
        other_player_loc_idx = 1
        dish_locations_idx= 22
        if len(obs['visual_obs'].shape) == 4:
            return obs['visual_obs'][0][dish_locations_idx][np.nonzero(obs['visual_obs'][0][other_player_loc_idx])] == 1
        else:
            return obs['visual_obs'][dish_locations_idx][np.nonzero(obs['visual_obs'][other_player_loc_idx])] == 1

    def non_full_pot_exists(self, obs):
        pot_locations_idx = 10
        onions_in_pot_idx= 16
        onions_in_soup_idx = 18
        if len(obs['visual_obs'].shape) == 4:
            for loc in zip(*np.nonzero(obs['visual_obs'][0][pot_locations_idx])):
                if obs['visual_obs'][0][onions_in_pot_idx][loc] < 3 and obs['visual_obs'][0][onions_in_soup_idx][loc] == 0:
                    return True
        else:
            for loc in zip(*np.nonzero(obs['visual_obs'][pot_locations_idx])):
                if obs['visual_obs'][onions_in_pot_idx][loc] < 3 and obs['visual_obs'][onions_in_soup_idx][loc] == 0:
                    return True
        return False

    def a_soup_is_almost_done(self, obs, time_left_thresh=10):
        pot_locations_idx = 10
        onions_in_soup_idx = 18
        cooking_time_left_idx= 20
        if len(obs['visual_obs'].shape) == 4:
            for loc in zip(*np.nonzero(obs['visual_obs'][0][pot_locations_idx])):
                if obs['visual_obs'][0][onions_in_soup_idx][loc] == 3 and obs['visual_obs'][0][cooking_time_left_idx][loc] <= time_left_thresh:
                    return True
        else:
            for loc in zip(*np.nonzero(obs['visual_obs'][pot_locations_idx])):
                if obs['visual_obs'][onions_in_soup_idx][loc] == 3 and obs['visual_obs'][cooking_time_left_idx][loc] <= time_left_thresh:
                    return True
        return False

    def is_urgent(self, obs):
        urgent_idx = 25
        if len(obs['visual_obs'].shape) == 4:
            return np.sum(obs['visual_obs'][0][urgent_idx]) > 0
        else:
            return np.sum(obs['visual_obs'][urgent_idx]) > 0

    def get_manually_tuned_action(self, obs, deterministic=False):
        dist = self.manager.get_distribution(obs)
        probs = dist.distribution.probs
        probs = probs[0]
        assert np.isclose(np.sum(probs.numpy()), 1)
        if self.layout_name == None:
            raise ValueError("Set current layout using set_curr_layout before attempting manual adjustment")
        elif self.layout_name == 'counter_circuit_o_1order':
            # if self.p_idx == 0:
                # Up weight supporting tasks
            subtasks_to_weigh = [Subtasks.SUBTASKS_TO_IDS['unknown']]
            subtask_weighting = [0.1]
            new_probs = self.adjust_distributions(probs, subtasks_to_weigh, subtask_weighting)
            if self.non_full_pot_exists(obs) and not self.is_urgent(obs):
                    subtasks_to_weigh = [Subtasks.SUBTASKS_TO_IDS['get_onion_from_counter'],
                                         Subtasks.SUBTASKS_TO_IDS['get_plate_from_dish_rack']]
                    subtask_weighting = [200, 0.1]
                    new_probs = self.adjust_distributions(new_probs, subtasks_to_weigh, subtask_weighting)


        elif self.layout_name == 'forced_coordination':
            # NOTE: THIS ASSUMES BEING P2
            # Since tasks are very limited, we use a different change instead of support and coordinated.
            if self.p_idx == 1:
                if (self.subtask_step + 2) % 16 == 0 or (self.subtask_step + 4) % 16 == 0:
                    subtasks_to_weigh = Subtasks.SUBTASKS_TO_IDS['get_plate_from_dish_rack']
                elif (self.subtask_step + 1) % 16 == 0 or (self.subtask_step + 3) % 16 == 0:
                    subtasks_to_weigh = Subtasks.SUBTASKS_TO_IDS['put_plate_closer']
                else:
                    if self.subtask_step % 2 == 0:
                        subtasks_to_weigh = Subtasks.SUBTASKS_TO_IDS['get_onion_from_dispenser']
                    else:
                        subtasks_to_weigh = Subtasks.SUBTASKS_TO_IDS['put_onion_closer']
                subtasks_to_weigh = [subtasks_to_weigh]
                subtask_weighting = [1e8 for _ in subtasks_to_weigh]
                new_probs = self.adjust_distributions(probs, subtasks_to_weigh, subtask_weighting)
                # print(self.subtask_step, [Subtasks.IDS_TO_SUBTASKS[s] for s in subtasks_to_weigh])
            else:
                new_probs = np.copy(probs.cpu()) if type(probs) == th.Tensor else np.copy(probs)
            # self.subtask_step += 1
        elif self.layout_name == 'asymmetric_advantages':
            #
            if self.p_idx == 0:
                if self.non_full_pot_exists(obs):
                    subtasks_to_weigh = [Subtasks.SUBTASKS_TO_IDS['get_onion_from_dispenser']]
                    subtask_weighting = [1e12 for _ in subtasks_to_weigh]
                    new_probs = self.adjust_distributions(probs, subtasks_to_weigh, subtask_weighting)

                elif (not self.a_soup_is_almost_done(obs, time_left_thresh=2) or self.other_player_has_plate(obs))\
                        and self.waiting_steps < 5:
                    subtasks_to_weigh = [Subtasks.SUBTASKS_TO_IDS['unknown'],
                                         Subtasks.SUBTASKS_TO_IDS['get_onion_from_dispenser']]
                    subtask_weighting = [1e12 for _ in subtasks_to_weigh]
                    new_probs = self.adjust_distributions(probs, subtasks_to_weigh, subtask_weighting)
                    self.waiting_steps += 1
                else:
                    new_probs = np.copy(probs.cpu()) if type(probs) == th.Tensor else np.copy(probs)
                    self.waiting_steps = 0

            elif self.p_idx == 1:
                if self.non_full_pot_exists(obs) and not self.a_soup_is_almost_done(obs, time_left_thresh=14) and not self.is_urgent(obs):
                    subtasks_to_weigh = [Subtasks.SUBTASKS_TO_IDS['get_onion_from_dispenser'], Subtasks.SUBTASKS_TO_IDS['put_onion_in_pot']]
                    subtask_weighting = [1e8 for _ in subtasks_to_weigh]
                    new_probs = self.adjust_distributions(probs, subtasks_to_weigh, subtask_weighting)
                    self.waiting_steps = 0
                elif self.other_player_has_plate(obs) and self.waiting_steps < 5:
                    subtasks_to_weigh = [Subtasks.SUBTASKS_TO_IDS['get_plate_from_dish_rack']]
                    subtask_weighting = [1e-12 for _ in subtasks_to_weigh]
                    new_probs = self.adjust_distributions(probs, subtasks_to_weigh, subtask_weighting)
                    self.waiting_steps += 1
                else:
                    new_probs = np.copy(probs.cpu()) if type(probs) == th.Tensor else np.copy(probs)
                    self.waiting_steps = 0

        else:
            new_probs = np.copy(probs.cpu()) if type(probs) == th.Tensor else np.copy(probs)
        while not np.isclose(np.sum(new_probs), 1, rtol=1e-3, atol=1e-3):
            new_probs /= np.sum(new_probs)
            # print('--------------\n', new_probs, '\n--->\n', probs)
        subtask = np.argmax(new_probs, axis=-1) if deterministic else Categorical(probs=th.tensor(new_probs)).sample()
        return np.expand_dims(np.array(subtask), 0)

    def predict(self, obs, state=None, episode_start=None, deterministic: bool=False):
        # TODO consider forcing new subtask if none has been completed in x timesteps
        # print(obs['player_completed_subtasks'],  self.prev_player_comp_st, (obs['player_completed_subtasks'] != self.prev_player_comp_st).any(), flush=True)
        if np.sum(obs['player_completed_subtasks']) == 1 or np.sum(obs['teammate_completed_subtasks']) == 1 or \
            self.num_steps_since_new_subtask >= 2 or self.curr_subtask_id == Subtasks.SUBTASKS_TO_IDS['unknown']:
            if np.sum(obs['player_completed_subtasks'][:-1]) == 1:
                self.subtask_step += 1
            # Completed previous subtask, set new subtask
            if self.tune_subtasks:
                self.curr_subtask_id = self.get_manually_tuned_action(obs, deterministic=deterministic)
            else:
                self.curr_subtask_id = self.manager.predict(obs, state=state, episode_start=episode_start,
                                                            deterministic=deterministic)[0]
        # if Subtasks.IDS_TO_SUBTASKS[int(self.curr_subtask_id)] == 'unknown':
        #     print(f'SUBTASK: {Subtasks.IDS_TO_SUBTASKS[int(self.curr_subtask_id)]}')
        # self.num_steps_since_new_subtask = 0
        if not isinstance(self.curr_subtask_id, int):
            self.curr_subtask_id = int(self.curr_subtask_id.squeeze())
        # print(Subtasks.IDS_TO_SUBTASKS[self.curr_subtask_id])
        if self.curr_subtask_id == Subtasks.SUBTASKS_TO_IDS['unknown']:
            return np.array([Action.ACTION_TO_INDEX[Action.STAY]]), None
        self.num_steps_since_new_subtask += 1
        worker_obs = self.obs_closure_fn(p_idx=self.p_idx, goal_objects=Subtasks.IDS_TO_GOAL_MARKERS[self.curr_subtask_id])
        worker_obs = {k: np.expand_dims(v, 0) for k, v in worker_obs.items()}
        return self.worker.predict(worker_obs, state=state, episode_start=episode_start, deterministic=False)

    def get_agent_output(self):
        return Subtasks.IDS_TO_HR_SUBTASKS[int(self.curr_subtask_id)] if self.output_message else ' '

    def save(self, path: Path) -> None:
        """
        Save model to a given location.
        :param path:
        """
        save_path = path / 'agent_file'
        worker_save_path = path / 'worker'
        manager_save_path = path / 'manager'
        self.worker.save(worker_save_path)
        self.manager.save(manager_save_path)
        args = get_args_to_save(self.args)
        th.save({'worker_type': type(self.worker), 'manager_type': type(self.manager),
                 'agent_type': type(self), 'const_params': self._get_constructor_parameters(), 'args': args}, save_path)

    @classmethod
    def load(cls, path: Path, args) -> 'OAIAgent':
        """
        Load model from path.
        :param path: path to save to
        :param device: Device on which the policy should be loaded.
        :return:
        """
        device = args.device
        load_path = path / 'agent_file'
        saved_variables = th.load(load_path, map_location=device)
        set_args_from_load(saved_variables['args'], args)
        worker = saved_variables['worker_type'].load(path / 'worker', args)
        manager = saved_variables['manager_type'].load(path / 'manager', args)
        saved_variables['const_params']['args'] = args

        # Create agent object
        model = cls(manager=manager, worker=worker, args=args)  # pytype: disable=not-instantiable
        model.to(device)
        return model

