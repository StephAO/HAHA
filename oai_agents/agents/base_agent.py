from oai_agents.agents.agent_utils import load_agent
from oai_agents.common.arguments import get_args_to_save, set_args_from_load, get_arguments
from oai_agents.common.state_encodings import ENCODING_SCHEMES
from oai_agents.common.subtasks import calculate_completed_subtask, get_doable_subtasks, Subtasks
from oai_agents.gym_environments.base_overcooked_env import USEABLE_COUNTERS

from overcooked_ai_py.mdp.overcooked_mdp import Action

from abc import ABC, abstractmethod
import argparse
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import numpy as np
import torch as th
import torch.nn as nn
from typing import List, Tuple, Union
import stable_baselines3.common.distributions as sb3_distributions
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations
import wandb


class OAIAgent(nn.Module, ABC):
    """
    A smaller version of stable baselines Base algorithm with some small changes for my new agents
    https://stable-baselines3.readthedocs.io/en/master/modules/base.html#stable_baselines3.common.base_class.BaseAlgorithm
    Ensures that all agents play nicely with the environment
    """

    def __init__(self, name, args):
        super(OAIAgent, self).__init__()
        self.name = name
        # Player index and Teammate index
        self.encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
        self.args = args
        # Must define a policy. The policy must implement a get_distribution(obs) that returns the action distribution
        self.policy = None
        # Used in overcooked-demo code
        self.p_idx = None
        self.mdp = None
        self.horizon = None
        self.prev_subtask = Subtasks.SUBTASKS_TO_IDS['unknown']
        self.use_hrl_obs = False

    @abstractmethod
    def predict(self, obs: th.Tensor, state=None, episode_start=None, deterministic: bool = False) -> Tuple[
        int, Union[th.Tensor, None]]:
        """
        Given an observation return the index of the action and the agent state if the agent is recurrent.
        Structure should be the same as agents created using stable baselines:
        https://stable-baselines3.readthedocs.io/en/master/modules/base.html#stable_baselines3.common.base_class.BaseAlgorithm.predict
        """

    @abstractmethod
    def get_distribution(self, obs: th.Tensor) -> Union[th.distributions.Distribution, sb3_distributions.Distribution]:
        """
        Given an observation return the index of the action and the agent state if the agent is recurrent.
        Structure should be the same as agents created using stable baselines:
        https://stable-baselines3.readthedocs.io/en/master/modules/base.html#stable_baselines3.common.base_class.BaseAlgorithm.predict
        """

    def set_idx(self, p_idx, env, is_hrl=False, output_message=True, tune_subtasks=False):
        self.p_idx = p_idx
        self.layout_name = env.layout_name
        self.obs_closure_fn = env.get_obs
        self.prev_state = None
        self.stack_frames = self.policy.observation_space['visual_obs'].shape[0] == (27 * self.args.num_stack)
        self.stackedobs = StackedObservations(1, self.args.num_stack, self.policy.observation_space['visual_obs'], 'first')
        if is_hrl:
            self.set_play_params(output_message, tune_subtasks)
        self.valid_counters = env.valid_counters

    def set_encoding_params(self, mdp, horizon):
        self.mdp = mdp
        self.horizon = horizon
        self.terrain = self.mdp.terrain_mtx
        self.grid_shape = (7, 7)

    def action(self, state, deterministic=False):
        if self.p_idx is None or self.mdp is None or self.horizon is None:
            raise ValueError('Please call set_idx() and set_encoding_params() before action. '
                             'Or, call predict with agent specific obs')

        obs = self.encoding_fn(self.mdp, state, self.grid_shape, self.horizon, p_idx=self.p_idx)
        if self.stack_frames:
            obs['visual_obs'] = np.expand_dims(obs['visual_obs'], 0)
            if self.prev_state is not None:
                obs['visual_obs'] = self.stackedobs.reset(obs['visual_obs'])
            else:
                obs['visual_obs'], _ = self.stackedobs.update(obs['visual_obs'], np.array([False]), [{}])
            obs['visual_obs'] = obs['visual_obs'].squeeze()

        if 'subtask_mask' in self.policy.observation_space.keys():
            obs['subtask_mask'] = get_doable_subtasks(state, self.prev_subtask, self.layout_name, self.terrain, self.p_idx, self.valid_counters, USEABLE_COUNTERS.get(self.layout_name, 5)).astype(bool)

        self.prev_state = deepcopy(state)
        obs = {k: v for k, v in obs.items() if k in self.policy.observation_space.keys()}

        try:
            agent_msg = self.get_agent_output()
        except AttributeError as e:
            agent_msg = ' '

        action, _ = self.predict(obs, deterministic=deterministic)
        return Action.INDEX_TO_ACTION[int(action)], agent_msg

    def _get_constructor_parameters(self):
        return dict(name=self.name, args=self.args)

    def step(self):
        pass

    def reset(self):
        pass

    def save(self, path: Path) -> None:
        """
        Save model to a given location.
        :param path:
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        save_path = path / 'agent_file'
        args = get_args_to_save(self.args)
        th.save({'agent_type': type(self), 'state_dict': self.state_dict(),
                 'const_params': self._get_constructor_parameters(), 'args': args}, save_path)

    @classmethod
    def load(cls, path: str, args: argparse.Namespace) -> 'OAIAgent':
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
        saved_variables['const_params']['args'] = args
        # Create agent object
        model = cls(**saved_variables['const_params'])  # pytype: disable=not-instantiable
        # Load weights
        model.load_state_dict(saved_variables['state_dict'])
        model.to(device)
        return model


class SB3Wrapper(OAIAgent):
    def __init__(self, agent, name, args):
        super(SB3Wrapper, self).__init__(name, args)
        self.agent = agent
        self.policy = self.agent.policy
        self.num_timesteps = 0

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        # Based on https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py#L305
        # Updated to include action masking
        obs = {k: v for k, v in obs.items() if k in self.policy.observation_space.keys()}
        self.policy.set_training_mode(False)
        obs, vectorized_env = self.policy.obs_to_tensor(obs)
        with th.no_grad():
            if 'subtask_mask' in obs and np.prod(obs['subtask_mask'].shape) == np.prod(self.agent.action_space.n):
                dist = self.policy.get_distribution(obs, action_masks=obs['subtask_mask'])
            else:
                dist = self.policy.get_distribution(obs)

            actions = dist.get_actions(deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1,) + self.agent.action_space.shape)
        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)
        return actions, state

    def get_distribution(self, obs: th.Tensor):
        self.policy.set_training_mode(False)
        obs, vectorized_env = self.policy.obs_to_tensor(obs)
        with th.no_grad():
            if 'subtask_mask' in obs and np.prod(obs['subtask_mask'].shape) == np.prod(self.policy.action_space.n):
                dist = self.policy.get_distribution(obs, action_masks=obs['subtask_mask'])
            else:
                dist = self.policy.get_distribution(obs)
        return dist

    def learn(self, total_timesteps):
        self.agent.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
        self.num_timesteps = self.agent.num_timesteps

    def save(self, path: Path) -> None:
        """
        Save model to a given location.
        :param path:
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        save_path = path / 'agent_file'
        args = get_args_to_save(self.args)
        th.save({'agent_type': type(self), 'sb3_model_type': type(self.agent),
                 'const_params': self._get_constructor_parameters(), 'args': args}, save_path)
        self.agent.save(str(save_path) + '_sb3_agent')

    @classmethod
    def load(cls, path: Path, args: argparse.Namespace, **kwargs) -> 'SB3Wrapper':
        """
        Load model from path.
        :param path: path to save to
        :param device: Device on which the policy should be loaded.
        :return:
        """
        device = args.device
        load_path = path / 'agent_file'
        saved_variables = th.load(load_path)
        set_args_from_load(saved_variables['args'], args)
        saved_variables['const_params']['args'] = args
        # Create agent object
        agent = saved_variables['sb3_model_type'].load(str(load_path) + '_sb3_agent')
        # Create wrapper object
        model = cls(agent=agent, **saved_variables['const_params'], **kwargs)  # pytype: disable=not-instantiable
        model.to(device)
        return model


class SB3LSTMWrapper(SB3Wrapper):
    ''' A wrapper for a stable baselines 3 agents that uses an lstm and controls a single player '''
    def __init__(self, agent, name, args):
        super(SB3LSTMWrapper, self).__init__(agent, name, args)
        self.lstm_states = None

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        episode_start = episode_start or np.ones((1,), dtype=bool)
        action, self.lstm_states = self.agent.predict(obs, state=state, episode_start=episode_start,
                                                      deterministic=deterministic)
        return action, self.lstm_states

    def get_distribution(self, obs: th.Tensor, state=None, episode_start=None):
        # TODO I think i need to store lstm states here
        episode_start = episode_start or np.ones((1,), dtype=bool)
        return self.agent.get_distribution(obs, lstm_states=state, episode_start=episode_start)

class PolicyClone(OAIAgent):
    """
    Policy Clones are copies of other agents policies (and nothing else). They can only play the game.
    They do not support training, saving, or loading, just playing.
    """
    def __init__(self, source_agent, args, device=None, name=None):
        """
        Given a source agent, create a new agent that plays identically.
        WARNING: This just copies the replica's policy, not all the necessary training code
        """
        name = name or 'policy_clone'
        super(PolicyClone, self).__init__(name, args)
        device = device or args.device
        # Create policy object
        policy_cls = type(source_agent.policy)
        const_params = deepcopy(source_agent.policy._get_constructor_parameters())
        self.policy = policy_cls(**const_params)  # pytype: disable=not-instantiable
        # Load weights
        state_dict = deepcopy(source_agent.policy.state_dict())
        self.policy.load_state_dict(state_dict)
        self.policy.to(device)

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        # Based on https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py#L305
        # Updated to include action masking
        self.policy.set_training_mode(False)
        obs, vectorized_env = self.policy.obs_to_tensor(obs)
        with th.no_grad():
            if 'subtask_mask' in obs and np.prod(obs['subtask_mask'].shape) == np.prod(self.policy.action_space.n):
                dist = self.policy.get_distribution(obs, action_masks=obs['subtask_mask'])
            else:
                dist = self.policy.get_distribution(obs)

            actions = dist.get_actions(deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1,) + self.policy.action_space.shape)
        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)
        return actions, state

    def get_distribution(self, obs: th.Tensor):
        self.policy.set_training_mode(False)
        obs, vectorized_env = self.policy.obs_to_tensor(obs)
        with th.no_grad():
            if 'subtask_mask' in obs and np.prod(obs['subtask_mask'].shape) == np.prod(self.policy.action_space.n):
                dist = self.policy.get_distribution(obs, action_masks=obs['subtask_mask'])
            else:
                dist = self.policy.get_distribution(obs)
        return dist

    def update_policy(self, source_agent):
        """
        Update the current agents policy using the source agents weights
        WARNING: This just copies the replica's policy fully, not all the necessary training code
        """
        state_dict = deepcopy(source_agent.policy.state_dict())
        self.policy.load_state_dict(state_dict)

    def learn(self):
        raise NotImplementedError('Learning is not supported for cloned policies')

    def save(self, path):
        raise NotImplementedError('Saving is not supported for cloned policies')

    def load(self, path, args):
        raise NotImplementedError('Loading is not supported for cloned policies')

class OAITrainer(ABC):
    """
    An abstract base class for trainer classes.
    Trainer classes must have two agents that they can train using some paradigm
    """
    def __init__(self, name, args, seed=None):
        super(OAITrainer, self).__init__()
        self.name = name
        self.args = args
        self.ck_list = []
        if seed is not None:
            th.manual_seed(seed)
            np.random.seed(seed)

        self.eval_teammates = None

        # For environment splits while training
        self.n_layouts = len(self.args.layout_names)
        self.splits = []
        self.curr_split = 0
        for split_size in range(self.n_layouts):
            for split in combinations(range(self.n_layouts), split_size + 1):
                self.splits.append(split)
        self.env_setup_idx, self.weighted_ratio = 0, 0.9

    def _get_constructor_parameters(self):
        return dict(name=self.name, args=self.args)

    def get_linear_schedule(self, start=1e-3, end=1e-4, end_fraction=0.8, name='lr'):
        def linear_anneal(progress_remaining: float) -> float:
            training_completed = self.agents_timesteps[0] / 2e7
            if training_completed > end_fraction:
                lr = end
            else:
                lr = start + training_completed * (end - start) / end_fraction
            if self.agents_timesteps[0] > 0:
                wandb.log({'timestep': self.agents_timesteps[0], name: lr})
            return lr
        return linear_anneal

    def evaluate(self, eval_agent, num_eps_per_layout_per_tm=5, visualize=False, timestep=None, log_wandb=True,
                 deterministic=False):
        tot_mean_reward = []
        rew_per_layout = {}
        use_layout_specific_tms = type(self.eval_teammates) == dict
        timestep = timestep if timestep is not None else eval_agent.num_timesteps
        for i, env in enumerate(self.eval_envs):
            tms = self.eval_teammates[env.get_layout_name()] if use_layout_specific_tms else self.eval_teammates
            rew_per_layout[env.layout_name] = []
            for tm in tms:
                env.set_teammate(tm)
                for p_idx in [0, 1]:
                    env.set_reset_p_idx(p_idx)
                    mean_reward, std_reward = evaluate_policy(eval_agent, env, n_eval_episodes=num_eps_per_layout_per_tm,
                                                              deterministic=deterministic, warn=False, render=visualize)
                    tot_mean_reward.append(mean_reward)
                    rew_per_layout[env.layout_name].append(mean_reward)
                    print(f'Eval at timestep {timestep} for layout {env.layout_name} at p{p_idx+1} with tm {tm.name}: {mean_reward}')
            rew_per_layout[env.layout_name] = np.mean(rew_per_layout[env.layout_name])
            if log_wandb:
                wandb.log({f'eval_mean_reward_{env.layout_name}': rew_per_layout[env.layout_name], 'timestep': timestep})

        print(f'Eval at timestep {timestep}: {np.mean(tot_mean_reward)}')
        if log_wandb:
            wandb.log({f'eval_mean_reward': np.mean(tot_mean_reward), 'timestep': timestep})
        return np.mean(tot_mean_reward), rew_per_layout

    def set_new_teammates(self):
        for i in range(self.args.n_envs):
            # each layout has different potential teammates
            if type(self.teammates) == dict:
                layout_name = self.env.env_method('get_layout_name', indices=i)[0]
                teammates = self.teammates[layout_name]
            else: # all layouts share teammates
                teammates = self.teammates
            teammate = teammates[np.random.randint(len(teammates))]
            self.env.env_method('set_teammate', teammate, indices=i)

    def get_agents(self) -> List[OAIAgent]:
        """
        Structure should be the same as agents created using stable baselines:
        https://stable-baselines3.readthedocs.io/en/master/modules/base.html#stable_baselines3.common.base_class.BaseAlgorithm.predict
        """
        return self.agents

    def save_agents(self, path: Union[Path, None] = None, tag: Union[str, None] = None):
        ''' Saves each agent that the trainer is training '''
        path = path or self.args.base_dir / 'agent_models' / self.name
        tag = tag or self.args.exp_name
        save_path = path / tag / 'trainer_file'
        agent_path = path / tag / 'agents_dir'
        Path(agent_path).mkdir(parents=True, exist_ok=True)
        save_dict = {'agent_fns': []}
        for i, agent in enumerate(self.agents):
            agent_path_i = agent_path / f'agent_{i}'
            agent.save(agent_path_i)
            save_dict['agent_fns'].append(f'agent_{i}')
        th.save(save_dict, save_path)
        return path, tag

    @staticmethod
    def load_agents(args, name: str=None, path: Union[Path, None] = None, tag: Union[str, None] = None):
        ''' Loads each agent that the trainer is training '''
        path = path or args.base_dir / 'agent_models' / name
        tag = tag or args.exp_name
        load_path = path / tag / 'trainer_file'
        agent_path = path / tag / 'agents_dir'
        device = args.device
        saved_variables = th.load(load_path, map_location=device)

        # Load weights
        agents = []
        for agent_fn in saved_variables['agent_fns']:
            agent = load_agent(agent_path / agent_fn, args)
            agent.to(device)
            agents.append(agent)
        return agents
