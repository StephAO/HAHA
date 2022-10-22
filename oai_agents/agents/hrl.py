from oai_agents.agents.base_agent import OAIAgent, load_agent
from oai_agents.agents.il import BehavioralCloningTrainer
from oai_agents.agents.rl import MultipleAgentsTrainer, SingleAgentTrainer, SB3Wrapper, SB3LSTMWrapper, VEC_ENV_CLS
from oai_agents.common.arguments import get_arguments, get_args_to_save, set_args_from_load
from oai_agents.common.subtasks import Subtasks
from oai_agents.gym_environments.worker_env import OvercookedSubtaskGymEnv
from oai_agents.gym_environments.manager_env import OvercookedManagerGymEnv

from overcooked_ai_py.mdp.overcooked_mdp import Action, OvercookedGridworld

from copy import deepcopy
from gym import spaces
import numpy as np
from pathlib import Path
from stable_baselines3.common.env_util import make_vec_env
import torch as th
import torch.nn.functional as F
from typing import Tuple, List


# TODO Move to util
def is_held_obj(player, object):
    '''Returns True if the object that the "player" picked up / put down is the same as the "object"'''
    x, y = np.array(player.position) + np.array(player.orientation)
    return player.held_object is not None and \
           ((object.name == player.held_object.name) or
            (object.name == 'soup' and player.held_object.name == 'onion'))\
           and object.position == (x, y)

### UTILS ###
class DummyPolicy:
    def __init__(self, obs_space):
        self.observation_space = obs_space

class DummyAgent:
    def __init__(self, action=Action.STAY):
        self.action = action if action == 'random' else Action.ACTION_TO_INDEX[action]
        self.name = f'{action}_agent'
        self.policy = DummyPolicy(spaces.Dict({'visual_obs': spaces.Box(0,1,(1,))}))
        self.encoding_fn = lambda *args, **kwargs: {}
        self.use_hrl_obs = False

    def predict(self, x, state=None, episode_start=None, deterministic=False):
        if self.action == 'random':
            action = np.random.randint(0, Action.NUM_ACTIONS)
        else:
            action = self.action
        return action, None



class MultiAgentSubtaskWorker(OAIAgent):
    def __init__(self, agents, args):
        super(MultiAgentSubtaskWorker, self).__init__('multi_agent_subtask_worker', args)
        assert len(agents) == Subtasks.NUM_SUBTASKS
        self.agents = agents
        self.agents[-1] = DummyAgent(action=Action.STAY) # Make unknown subtask equivalent to stay

    def predict(self, obs: th.Tensor, state=None, episode_start=None, deterministic: bool=False):
        assert 'curr_subtask' in obs.keys()
        obs = {k: v for k, v in obs.items() if k in ['visual_obs', 'agent_obs', 'curr_subtask']}
        try: # curr_subtask is iterable because this is a training batch
            preds = [self.agents[st].predict(obs, state=state, episode_start=episode_start, deterministic=deterministic)
                     for st in obs['curr_subtask']]
            actions, states = zip(*preds)

        except TypeError: # curr_subtask is not iterable because this is regular run
            actions, states = self.agents[obs['curr_subtask']].predict(obs, state=state, episode_start=episode_start,
                                                                       deterministic=deterministic)
        return actions, states

    def get_distribution(self, obs: th.Tensor):
        assert 'curr_subtask' in obs.keys()
        return self.agents[obs['curr_subtask']].get_distribution(obs)

    def _get_constructor_parameters(self):
        return dict(name=self.name)

    def save(self, path: Path) -> None:
        args = get_args_to_save(self.args)
        save_path = path / 'agent_file'
        agent_dir = path / 'subtask_agents_dir'
        Path(agent_dir).mkdir(parents=True, exist_ok=True)

        save_dict = {'agent_type': type(self), 'agent_fns': [],
                     'const_params': self._get_constructor_parameters(), 'args': args}
        for i, agent in enumerate(self.agents):
            agent_path_i = agent_dir / f'subtask_{i}_agent'
            agent.save(agent_path_i)
            save_dict['agent_fns'].append(f'subtask_{i}_agent')
        th.save(save_dict, save_path)

    @classmethod
    def load(cls, path: Path, args):
        device = args.device
        load_path = path / 'agent_file'
        agent_dir = path / 'subtask_agents_dir'
        saved_variables = th.load(load_path, map_location=device)
        set_args_from_load(saved_variables['args'], args)
        saved_variables['const_params']['args'] = args

        # Load weights
        agents = []
        for agent_fn in saved_variables['agent_fns']:
            agent = load_agent(agent_dir / agent_fn, args)
            agent.to(device)
            agents.append(agent)
        print(f'Loaded {len(agents)} subtask worker agents')
        return cls(agents=agents, args=args)

    @classmethod
    def create_model_from_scratch(cls, args, teammates=None, eval_tms=None, dataset_file=None) -> Tuple['OAIAgent', List['OAIAgent']]:
        # Define teammates
        if teammates is not None:
            tms = teammates
        elif dataset_file is not None:
            bct = BehavioralCloningTrainer(dataset_file, args)
            bct.train_agents(epochs=100)
            tms = bct.get_agents()
        else:
            tsa = MultipleAgentsTrainer(args)
            tsa.train_agents(total_timesteps=1e7)
            tms = tsa.get_agents()

        # Train 12 individual agents, each for a respective subtask
        agents = []
        original_layout_names = deepcopy(args.layout_names)
        for i in range(Subtasks.NUM_SUBTASKS):
            print(f'Starting subtask {i} - {Subtasks.IDS_TO_SUBTASKS[i]}')
            # RL single subtask agents trained with teammeates
            # Make necessary envs

            # Don't bother training an agent on a subtask if that subtask is useless for that layout
            layouts_to_use = deepcopy(original_layout_names)
            if Subtasks.IDS_TO_SUBTASKS[i] in ['get_soup_from_counter', 'put_soup_closer']:
                layouts_to_use.remove('forced_coordination')
            if Subtasks.IDS_TO_SUBTASKS[i] in ['put_soup_closer', 'put_onion_closer', 'put_plate_closer',
                                               'get_soup_from_counter', 'get_onion_from_counter', 'get_plate_from_counter']:
                layouts_to_use.remove('asymmetric_advantages')
            args.layout_names = layouts_to_use
            n_layouts = len(args.layout_names)

            env_kwargs = {'single_subtask_id': i, 'stack_frames': False, 'full_init': False, 'args': args}
            env = make_vec_env(OvercookedSubtaskGymEnv, n_envs=args.n_envs, env_kwargs=env_kwargs,
                               vec_env_cls=VEC_ENV_CLS)

            env_kwargs['full_init'] = True
            eval_envs = [OvercookedSubtaskGymEnv(**{'env_index': n, 'is_eval_env': True, **env_kwargs})
                         for n in range(n_layouts)]
            # Create trainer
            name = f'subtask_worker_{i}'
            rl_sat = SingleAgentTrainer(tms, args, eval_tms=eval_tms, name=name, env=env, eval_envs=eval_envs, use_subtask_eval=True)
            # Train if it makes sense to (can't train on an unknown task)
            if i != Subtasks.SUBTASKS_TO_IDS['unknown']:
                rl_sat.train_agents(total_timesteps=1e7)
            agents.extend(rl_sat.get_agents())

        args.layout_names = original_layout_names
        model = cls(agents=agents, args=args)
        path = args.base_dir / 'agent_models' / model.name
        Path(path).mkdir(parents=True, exist_ok=True)
        model.save(path / args.exp_name)
        return model, tms

    @classmethod
    def create_model_from_pretrained_subtask_workers(cls, args):
        agents = []
        for i in range(11):
            agents.append(SB3Wrapper.load(args.base_dir / 'agent_models' / f'subtask_worker_{i}' / 'best' / 'agents_dir' / 'agent_0', args))

        # All this logic is to get an unknown agent
        env_kwargs = {'single_subtask_id': 11, 'env_index': 0, 'full_init': True, 'stack_frames': False, 'args': args}
        env = make_vec_env(OvercookedSubtaskGymEnv, n_envs=args.n_envs, env_kwargs=env_kwargs,
                           vec_env_cls=VEC_ENV_CLS)
        # Create trainer
        name = f'subtask_worker_11'
        rl_sat = SingleAgentTrainer(agents, args, name=name, env=env, eval_envs=None, use_subtask_eval=True)
        agents.extend(rl_sat.get_agents())

        print(len(agents))

        model = cls(agents=agents, args=args)
        path = args.base_dir / 'agent_models' / model.name
        Path(path).mkdir(parents=True, exist_ok=True)
        model.save(path / args.exp_name)

class RLManagerTrainer(SingleAgentTrainer):
    ''' Train an RL agent to play with a provided agent '''
    def __init__(self, worker, teammates, args, eval_tms=None, use_frame_stack=False, use_subtask_counts=False,
                 inc_sp=False, name=None):
        name = name or 'hrl_manager'
        n_layouts = len(args.layout_names)
        env_kwargs = {'worker': worker, 'shape_rewards': False, 'stack_frames': use_frame_stack, 'full_init': False, 'args': args}
        env = make_vec_env(OvercookedManagerGymEnv, n_envs=args.n_envs, env_kwargs=env_kwargs, vec_env_cls=VEC_ENV_CLS)

        eval_envs_kwargs = {'worker': worker, 'shape_rewards': False, 'stack_frames': use_frame_stack,
                            'is_eval_env': True, 'horizon': 400, 'args': args}
        eval_envs = [OvercookedManagerGymEnv(**{'env_index': i, **eval_envs_kwargs}) for i in range(n_layouts)]

        self.worker = worker
        super(RLManagerTrainer, self).__init__(teammates, args, eval_tms=eval_tms, name=name, env=env,
                                               eval_envs=eval_envs, inc_sp=False, use_subtask_counts=use_subtask_counts,
                                               use_hrl=True, use_maskable_ppo=True)
        if inc_sp:
            playable_self = HierarchicalRL(self.worker, self.learning_agent, args)
            self.teammates.append(playable_self)

        if type(self.eval_teammates) == dict:
            playable_self = HierarchicalRL(self.worker, self.learning_agent, args)
            for k in self.eval_teammates:
                self.eval_teammates[k].append(playable_self)
        elif self.eval_teammates is not None:
            playable_self = HierarchicalRL(self.worker, self.learning_agent, args)
            self.eval_teammates.append(playable_self)
        else:
            self.eval_teammates = self.teammates


class HierarchicalRL(OAIAgent):
    def __init__(self, worker, manager, args, name=None):
        name = name or 'hrl'
        super(HierarchicalRL, self).__init__(name, args)
        self.worker = worker
        self.manager = manager
        self.prev_player_comp_st = None
        self.policy = self.manager.policy
        self.num_steps_since_new_subtask = 0
        self.use_hrl_obs = True
        self.curr_layout = None
        self.subtask_step = 0

    def get_distribution(self, obs, sample=True):
        if obs['player_completed_subtasks'] != self.prev_player_comp_st:
            # Completed previous subtask, set new subtask
            self.curr_subtask_id = self.manager.predict(obs, sample=sample)[0]
            self.prev_player_comp_st = obs['player_completed_subtasks']
        obs['curr_subtask'] = self.curr_subtask_id
        return self.worker.get_distribution(obs, sample=sample)

    def set_curr_layout(self, layout_name):
        self.curr_layout = layout_name
        self.subtask_step = 0

    def adjust_distributions(self, probs, indices, weights):
        new_probs = np.copy(probs)
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

    def get_manually_tuned_action(self, obs):
        # Currently assumes p2 for hrl agent
        dist = self.manager.get_distribution(obs)
        probs = dist.distribution.probs
        if self.curr_layout == None:
            raise ValueError("Set current layout using set_curr_layout before attempting manual adjustment")
        elif self.curr_layout == 'counter_circuit_o_1order':
            subtasks_to_weigh = [Subtasks.SUBTASKS_TO_IDS['get_onion_from_counter'],
                                 Subtasks.SUBTASKS_TO_IDS['put_onion_closer']]
            subtask_weighting = [2, 2]
            new_probs = self.adjust_distributions(probs, subtasks_to_weigh, subtask_weighting)
        elif self.curr_layout == 'forced_coordination':
            # 3 onions then one plate
            subtask_weighting = [2]
            if (self.subtask_step + 2) % 8 == 0:
                subtasks_to_weigh = Subtasks.SUBTASKS_TO_IDS['get_plate_from_counter']
            elif (self.subtask_step + 1) % 8 == 0:
                subtasks_to_weigh = Subtasks.SUBTASKS_TO_IDS['put_plate_closer']
            else:
                if self.subtask_step % 2 == 0:
                    subtasks_to_weigh = [Subtasks.SUBTASKS_TO_IDS['get_onion_from_dispenser']]
                else:
                    subtasks_to_weigh = [Subtasks.SUBTASKS_TO_IDS['put_onion_closer']]
                subtask_weighting = [2]
            new_probs = self.adjust_distributions(probs, subtasks_to_weigh, [2, 2])
        elif self.curr_layout == 'asymmetric_advantages':
            subtasks_to_weigh = [Subtasks.SUBTASKS_TO_IDS['get_plate_from_dish_rack'],
                                 Subtasks.SUBTASKS_TO_IDS['get_soup'],
                                 Subtasks.SUBTASKS_TO_IDS['serve_soup']]
            new_probs = self.adjust_distributions(probs, subtasks_to_weigh, [2, 2, 2])
        else:
            new_probs = probs
        new_dist = Categorical(probs=new_probs)
        return new_dist.sample()

    def predict(self, obs, state=None, episode_start=None, deterministic: bool=False):
        # TODO consider forcing new subtask if none has been completed in x timesteps
        # print(obs['player_completed_subtasks'],  self.prev_player_comp_st, (obs['player_completed_subtasks'] != self.prev_player_comp_st).any(), flush=True)
        if (obs['player_completed_subtasks'] != self.prev_player_comp_st).any() or \
                self.num_steps_since_new_subtask > 25 or self.curr_subtask_id == Subtasks.SUBTASKS_TO_IDS['unknown']:
            # Completed previous subtask, set new subtask
            self.curr_subtask_id = self.manager.predict(obs, state=state, episode_start=episode_start,
                                                        deterministic=deterministic)[0]
            self.prev_player_comp_st = deepcopy(obs['player_completed_subtasks'])
            self.num_steps_since_new_subtask = 0
        obs['curr_subtask'] = self.curr_subtask_id
        self.num_steps_since_new_subtask += 1
        return self.worker.predict(obs, state=state, episode_start=episode_start, deterministic=False)

    def get_agent_output(self):
        return Subtasks.IDS_TO_HR_SUBTASKS[int(self.curr_subtask_id)]

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


### EVERYTHING BELOW IS A DRAFT AT BEST
class ValueBasedManager():
    """
    Follows a few basic rules. (tm = teammate)
    1. All independent tasks values:
       a) Get onion from dispenser = (3 * num_pots) - 0.5 * num_onions
       b) Get plate from dish rack = num_filled_pots * (2
       c)
       d)
    2. Supporting tasks
       a) Start at a value of zero
       b) Always increases in value by a small amount (supporting is good)
       c) If one is performed and the tm completes the complementary task, then the task value is increased
       d) If the tm doesn't complete the complementary task, after a grace period the task value starts decreasing
          until the object is picked up
    3. Complementary tasks:
       a) Start at a value of zero
       b) If a tm performs a supporting task, then its complementary task value increases while the object remains
          on the counter.
       c) If the object is removed from the counter, the complementary task value is reset to zero (the
          complementary task cannot be completed if there is no object to pick up)
    :return:
    """
    def __init__(self, worker, p_idx, args):
        super(ValueBasedManager, self).__init__(worker,'value_based_subtask_adaptor', p_idx, args)
        self.worker = worker
        assert worker.p_idx == p_idx
        self.trajectory = []
        self.terrain = OvercookedGridworld.from_layout_name(args.layout_names[index]).terrain_mtx
        # for i in range(len(self.terrain)):
        #     self.terrain[i] = ''.join(self.terrain[i])
        # self.terrain = str(self.terrain)
        self.worker_subtask_counts = np.zeros((2, Subtasks.NUM_SUBTASKS))
        self.subtask_selection = args.subtask_selection


        self.init_subtask_values()

    def init_subtask_values(self):
        self.subtask_values = np.zeros(Subtasks.NUM_SUBTASKS)
        # 'unknown' subtask is always set to 0 since it is more a relic of labelling than a useful subtask
        # Independent subtasks
        self.ind_subtask = ['get_onion_from_dispenser', 'put_onion_in_pot', 'get_plate_from_dish_rack', 'get_soup', 'serve_soup']
        # Supportive subtasks
        self.sup_subtask = ['put_onion_closer', 'put_plate_closer', 'put_soup_closer']
        self.sup_obj_to_subtask = {'onion': 'put_onion_closer', 'dish': 'put_plate_closer', 'soup': 'put_soup_closer'}
        # Complementary subtasks
        self.com_subtask = ['get_onion_from_counter', 'get_plate_from_counter', 'get_soup_from_counter']
        self.com_obj_to_subtask = {'onion': 'get_onion_from_counter', 'dish': 'get_plate_from_counter', 'soup': 'get_soup_from_counter'}
        for i_s in self.ind_subtask:
            # 1.a
            self.subtask_values[Subtasks.SUBTASKS_TO_IDS[i_s]] = 1
        for s_s in self.sup_subtask:
            # 2.a
            self.subtask_values[Subtasks.SUBTASKS_TO_IDS[s_s]] = 0
        for c_s in self.com_subtask:
            # 3.a
            self.subtask_values[Subtasks.SUBTASKS_TO_IDS[c_s]] = 0

        self.acceptable_wait_time = 10  # 2d
        self.sup_base_inc = 0.05  # 2b
        self.sup_success_inc = 1  # 2c
        self.sup_waiting_dec = 0.1  # 2d
        self.com_waiting_inc = 0.2  # 3d
        self.successful_support_task_reward = 1
        self.agent_objects = {}
        self.teammate_objects = {}

    def update_subtask_values(self, prev_state, curr_state):
        prev_objects = prev_state.objects.values()
        curr_objects = curr_state.objects.values()
        # TODO objects are only tracked by name and position, so checking equality fails because picking something up changes the objects position
        # 2.b
        for s_s in self.sup_subtask:
            self.subtask_values[Subtasks.SUBTASKS_TO_IDS[s_s]] += self.sup_base_inc

        # Analyze objects that are on counters
        for object in curr_objects:
            x, y = object.position
            if object.name == 'soup' and self.terrain[y][x] == 'P':
                # Soups while in pots can change without agent intervention
                continue
            # Objects that have been put down this turn
            if object not in prev_objects:
                if is_held_obj(prev_state.players[self.p_idx], object):
                    print(f'Agent placed {object}')
                    self.agent_objects[object] = 0
                elif is_held_obj(prev_state.players[self.t_idx], object):
                    print(f'Teammate placed {object}')
                    self.teammate_objects[object] = 0
                # else:
                #     raise ValueError(f'Object {object} has been put down, but did not belong to either player')
            # Objects that have not moved since the previous time step
            else:
                if object in self.agent_objects:
                    self.agent_objects[object] += 1
                    if self.agent_objects[object] > self.acceptable_wait_time:
                        # 2.d
                        subtask_id = Subtasks.SUBTASKS_TO_IDS[self.sup_obj_to_subtask[object.name]]
                        self.subtask_values[subtask_id] -= self.sup_waiting_dec
                elif object in self.teammate_objects:
                    # 3.b
                    self.teammate_objects[object] += 1
                    subtask_id = Subtasks.SUBTASKS_TO_IDS[self.com_obj_to_subtask[object.name]]
                    self.subtask_values[subtask_id] += self.com_waiting_inc

        for object in prev_objects:
            x, y = object.position
            if object.name == 'soup' and self.terrain[y][x] == 'P':
                # Soups while in pots can change without agent intervention
                continue
            # Objects that have been picked up this turn
            if object not in curr_objects:
                if is_held_obj(curr_state.players[self.p_idx], object):
                    print(f'Agent picked up {object}')
                    if object in self.agent_objects:
                        del self.agent_objects[object]
                    else:
                        del self.teammate_objects[object]

                elif is_held_obj(curr_state.players[self.t_idx], object):
                    print(f'Teammate picked up {object}')
                    if object in self.agent_objects:
                        # 2.c
                        subtask_id = Subtasks.SUBTASKS_TO_IDS[self.sup_obj_to_subtask[object.name]]
                        self.subtask_values[subtask_id] += self.sup_success_inc
                        del self.agent_objects[object]
                    else:
                        del self.teammate_objects[object]
                # else:
                #     raise ValueError(f'Object {object} has been picked up, but does not belong to either player')

                # Find out if there are any remaining objects of the same type left
                last_object_of_this_type = True
                for rem_objects in list(self.agent_objects) + list(self.teammate_objects):
                    if object.name == rem_objects.name:
                        last_object_of_this_type = False
                        break
                # 3.c
                if last_object_of_this_type:
                    subtask_id = Subtasks.SUBTASKS_TO_IDS[self.com_obj_to_subtask[object.name]]
                    self.subtask_values[subtask_id] = 0

        self.subtask_values = np.clip(self.subtask_values, 0, 10)

    def get_subtask_values(self, curr_state):
        assert self.subtask_values is not None
        return self.subtask_values * self.doable_subtasks(curr_state, self.terrain, self.p_idx)

    def select_next_subtask(self, curr_state):
        subtask_values = self.get_subtask_values(curr_state)
        subtask_id = np.argmax(subtask_values.squeeze(), dim=-1)
        self.curr_subtask_id = subtask_id
        print('new subtask', Subtasks.IDS_TO_SUBTASKS[subtask_id.item()])

    def reset(self, state):
        super().reset(state)
        self.init_subtask_values()

class DistBasedManager():
    def __init__(self, agent, p_idx, args):
        super(DistBasedManager, self).__init__(agent, p_idx, args)
        self.name = 'dist_based_subtask_agent'

    def distribution_matching(self, subtask_logits, egocentric=False):
        """
        Try to match some precalculated 'optimal' distribution of subtasks.
        If egocentric look only at the individual player distribution, else look at the distribution across both players
        """
        assert self.optimal_distribution is not None
        if egocentric:
            curr_dist = self.worker_subtask_counts[self.p_idx]
            best_dist = self.optimal_distribution[self.p_idx]
        else:
            curr_dist = self.worker_subtask_counts.sum(axis=0)
            best_dist = self.optimal_distribution.sum(axis=0)
        curr_dist = curr_dist / np.sum(curr_dist)
        dist_diff = best_dist - curr_dist

        pred_subtask_probs = F.softmax(subtask_logits).detach().numpy()
        # TODO investigate weighting
        # TODO should i do the above softmax?
        # Loosely based on Bayesian inference where prior is the difference in distributions, and the evidence is
        # predicted probability of what subtask should be done
        adapted_probs = pred_subtask_probs * dist_diff
        adapted_probs = adapted_probs / np.sum(adapted_probs)
        return adapted_probs

    def select_next_subtask(self, curr_state):
        # TODO
        pass

    def reset(self, state, player_idx):
        # TODO
        pass

if __name__ == '__main__':
    args = get_arguments()
    
    mat = MultipleAgentsTrainer(args, num_agents=0)
    mat.load_agents(path=Path('/projects/star7023/oai/agent_models/fcp/counter_circuit_o_1order/12_pop'), tag='test')
    teammates = mat.get_agents()

    # worker, teammates = MultiAgentSubtaskWorker.create_model_from_scratch(args, teammates=teammates)

    worker = MultiAgentSubtaskWorker.load(
            Path('/projects/star7023/oai/agent_models/multi_agent_subtask_worker/counter_circuit_o_1order/test/'), args)

    rlmt = RLManagerTrainer(worker, teammates, args)
    #rlmt.train_agents(total_timesteps=2e6, exp_name=args.exp_name + '_manager')
    rlmt.load_agents(path=Path('/projects/star7023/oai/agent_models/rl_manager/counter_circuit_o_1order'), tag='test')
    managers = rlmt.get_agents()
    manager = managers[0]
    hrl = HierarchicalRL(worker, manager, args)
    hrl.save(Path('/projects/star7023/oai/agent_models/hier_rl/counter_circuit_o_1order/test/'))
    # hrl.save('test_data/test')
    # del hrl
    # hrl = HierarchicalRL.load('test_data/test', args)
    print('done')


