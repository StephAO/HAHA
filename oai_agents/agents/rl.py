from oai_agents.agents.base_agent import SB3Wrapper, SB3LSTMWrapper, OAITrainer, PolicyClone
from oai_agents.common.arguments import get_arguments
from oai_agents.common.networks import OAISinglePlayerFeatureExtractor
from oai_agents.common.state_encodings import ENCODING_SCHEMES
from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from sb3_contrib import RecurrentPPO, MaskablePPO
import wandb

EPOCH_TIMESTEPS = 10000
VEC_ENV_CLS = DummyVecEnv #SubprocVecEnv

class SingleAgentTrainer(OAITrainer):
    ''' Train an RL agent to play with a provided agent '''
    def __init__(self, teammates, args, eval_tms=None, name=None, env=None, eval_envs=None, use_lstm=False,
                 use_frame_stack=False, use_subtask_counts=False, use_maskable_ppo=False, inc_sp=False, hidden_dim=64,
                 use_subtask_eval=False, use_hrl=False, seed=None):
        name = name or 'rl_singleagent'
        super(SingleAgentTrainer, self).__init__(name, args, seed=seed)
        self.args = args
        self.device = args.device
        self.use_lstm = use_lstm
        self.use_frame_stack = use_frame_stack
        self.use_subtask_eval = use_subtask_eval
        self.using_hrl = use_hrl
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
        self.teammates = teammates
        self.eval_teammates = eval_tms
        if env is None:
            env_kwargs = {'shape_rewards': True, 'ret_completed_subtasks': use_subtask_counts,
                          'stack_frames': use_frame_stack, 'full_init': False, 'args': args}
            self.env = make_vec_env(OvercookedGymEnv, n_envs=args.n_envs, vec_env_cls=VEC_ENV_CLS,
                                    env_kwargs=env_kwargs)

            eval_envs_kwargs = {'is_eval_env': True, 'ret_completed_subtasks': use_subtask_counts,
                                'stack_frames': use_frame_stack, 'horizon': 400, 'args': args}
            self.eval_envs = [OvercookedGymEnv(**{'env_index': i, **eval_envs_kwargs}) for i in range(self.n_layouts)]
        else:
            self.env = env
            self.eval_envs = eval_envs
        self.set_new_envs()

        policy_kwargs = dict(
            net_arch=[dict(pi=[hidden_dim, hidden_dim], vf=[hidden_dim, hidden_dim])]
        )
        if use_lstm:
            policy_kwargs['n_lstm_layers'] = 2
            policy_kwargs['lstm_hidden_size'] = hidden_dim
            sb3_agent = RecurrentPPO('MultiInputLstmPolicy', self.env, policy_kwargs=policy_kwargs, verbose=1,
                                     n_steps=2048, batch_size=64)
            agent_name = f'{name}_lstm'
        elif use_maskable_ppo:
            sb3_agent = MaskablePPO('MultiInputPolicy', self.env, policy_kwargs=policy_kwargs, verbose=1)
            agent_name = f'{name}'
        else:
            sb3_agent = PPO('MultiInputPolicy', self.env, policy_kwargs=policy_kwargs, verbose=1)
            agent_name = f'{name}'
        self.learning_agent = self.wrap_agent(sb3_agent, agent_name)
        self.agents = [self.learning_agent]

        if inc_sp:
            self.teammates.append(self.learning_agent)

        if not self.use_subtask_eval and not self.using_hrl:
            if type(self.eval_teammates) == dict:
                for k in self.eval_teammates:
                    self.eval_teammates[k].append(self.learning_agent)
            elif self.eval_teammates is not None:
                self.eval_teammates.append(self.learning_agent)
            else:
                self.eval_teammates = self.teammates

    def _get_constructor_parameters(self):
        return dict(args=self.args, name=self.name, use_lstm=self.use_lstm, use_frame_stack=self.use_frame_stack,
                    hidden_dim=self.hidden_dim, seed=self.seed)

    def wrap_agent(self, sb3_agent, name):
        if self.use_lstm:
            agent = SB3LSTMWrapper(sb3_agent, name, self.args)
        else:
            agent = SB3Wrapper(sb3_agent, name, self.args)
        return agent

    def train_agents(self, total_timesteps=2e6, exp_name=None):
        exp_name = exp_name or self.args.exp_name
        run = wandb.init(project="overcooked_ai_test", entity=self.args.wandb_ent,
                                  dir=str(self.args.base_dir / 'wandb'),
                                  reinit=True, name=exp_name + '_' + self.learning_agent.name, mode=self.args.wandb_mode)
        best_path, best_tag = None, None
        fewest_failures = float('inf')
        best_score = -1
        best_training_rew = float('-inf')

        epoch = 0
        while self.learning_agent.num_timesteps < total_timesteps:
            # Set new env distribution if training on multiple envs
            if self.n_layouts > 1 and self.args.multi_env_mode != 'uniform':
                self.set_new_envs()
            self.set_new_teammates()
            self.learning_agent.learn(total_timesteps=EPOCH_TIMESTEPS)
            if self.using_hrl:
                failure_dicts = self.env.env_method('get_worker_failures')
                tot_failure_dict = {ln: {k: 0 for k in failure_dicts[0][1].keys()} for ln in self.args.layout_names}
                for ln, fd in failure_dicts:
                    for k in tot_failure_dict[ln]:
                        tot_failure_dict[ln][k] += fd[k]
                for ln in self.args.layout_names:
                    wandb.log({f'num_worker_failures_{ln}': sum(tot_failure_dict[ln].values()), 'timestep': self.learning_agent.num_timesteps})
                    print(f'Number of worker failures on {ln}: {tot_failure_dict[ln]}')

            mean_training_rew = np.mean([ep_info["r"] for ep_info in self.learning_agent.agent.ep_info_buffer])
            best_training_rew *= 0.99
            if epoch % 20 == 0 or (mean_training_rew > best_training_rew and self.learning_agent.num_timesteps > 2e5):
                if mean_training_rew >= best_training_rew:
                    best_training_rew = mean_training_rew
                if self.use_subtask_eval:
                    env_success = []
                    use_layout_specific_tms = type(self.eval_teammates) == dict
                    tot_failures = 0
                    for env in self.eval_envs:
                        tms = self.eval_teammates[env.get_layout_name()] if use_layout_specific_tms else self.eval_teammates
                        for tm in tms:
                            env.set_teammate(tm)
                            fully_successful, num_failures = env.evaluate(self.learning_agent)
                            tot_failures += num_failures
                            env_success.append(fully_successful)
                    wandb.log({f'num_tm_layout_successes': np.sum(env_success), 'total_failures': tot_failures,
                               'timestep': self.learning_agent.num_timesteps})
                    if tot_failures <= fewest_failures:
                        best_path, best_tag = self.save_agents(tag='best')
                        fewest_failures = tot_failures
                        print(f'New fewest failures of {fewest_failures} reached, model saved to {best_path}/{best_tag}')
                    if all(env_success):
                        break
                else:
                    mean_reward = self.evaluate(self.learning_agent)
                    if mean_reward >= best_score:
                        best_path, best_tag = self.save_agents(tag='best')
                        print(f'New best score of {mean_reward} reached, model saved to {best_path}/{best_tag}')
                        best_score = mean_reward
            epoch += 1
        path, tag = self.save_agents()
        self.load_agents(best_path, best_tag)
        run.finish()


class MultipleAgentsTrainer(OAITrainer):
    ''' Train two independent RL agents to play with each other '''

    def __init__(self, args, name=None, eval_tms=None,num_agents=1, use_lstm=False, use_frame_stack=False,
                 use_subtask_counts=False, hidden_dim=64, use_policy_clone=False, fcp_ck_rate=None, seed=None):
        '''
        Train multiple agents with each other.
        :param num_agents: Number of agents to train. num_agents=1 mean self-play, num_agents > 1 is population play
        :param args: Experiment arguments. See arguments.py
        :param use_lstm: Whether agents should use an lstm policy or not
        :param hidden_dim: hidden dimensions for agents
        :param fcp_ck_rate: If not none, rate to save agents. Used primarily to get agents for Fictitous Co-Play
        :param seed: Random see
        '''
        name = name or 'rl_multiagents'
        super(MultipleAgentsTrainer, self).__init__(name, args, seed=seed)
        self.device = args.device
        self.args = args
        self.fcp_ck_rate = fcp_ck_rate
        self.use_lstm = use_lstm
        self.use_frame_stack = use_frame_stack
        self.use_policy_clones = use_policy_clone

        env_kwargs = {'shape_rewards': True, 'ret_completed_subtasks': use_subtask_counts,
                      'stack_frames': use_frame_stack, 'full_init': False, 'args': args}
        self.env = make_vec_env(OvercookedGymEnv, n_envs=args.n_envs, vec_env_cls=VEC_ENV_CLS,
                                env_kwargs=env_kwargs)
        self.set_new_envs()

        eval_envs_kwargs = {'ret_completed_subtasks': use_subtask_counts, 'stack_frames': use_frame_stack,
                            'is_eval_env': True, 'horizon': 400, 'args': args}
        self.eval_envs = [OvercookedGymEnv(**{'env_index': i, **eval_envs_kwargs}) for i in range(self.n_layouts)]

        policy_kwargs = dict(
            # features_extractor_class=OAISinglePlayerFeatureExtractor,
            # features_extractor_kwargs=dict(features_dim=hidden_dim),
            net_arch=[dict(pi=[hidden_dim, hidden_dim], vf=[hidden_dim, hidden_dim])]
        )

        self.agents = []
        if use_lstm:
            policy_kwargs['n_lstm_layers'] = 2
            policy_kwargs['lstm_hidden_size'] = hidden_dim
            for i in range(num_agents):
                sb3_agent = RecurrentPPO('MultiInputLstmPolicy', self.env, policy_kwargs=policy_kwargs, verbose=1,
                                         n_steps=2048, batch_size=64)
                agent_name = f'{name}_lstm_{i + 1}'
                self.agents.append(SB3LSTMWrapper(sb3_agent, agent_name, args))
        else:
            for i in range(num_agents):
                sb3_agent = PPO("MultiInputPolicy", self.env, policy_kwargs=policy_kwargs, verbose=1)
                agent_name = f'{name}_{i + 1}'
                self.agents.append(SB3Wrapper(sb3_agent, agent_name, args))

        self.teammates = [PolicyClone(agent, args) for agent in self.agents] if self.use_policy_clones else self.agents
        self.agents_in_training = np.ones(len(self.agents))
        self.agents_timesteps = np.zeros(len(self.agents))

    def _get_constructor_parameters(self):
        return dict(args=self.args, name=self.name, use_lstm=self.use_lstm, use_frame_stack=self.use_frame_stack,
                    hidden_dim=self.hidden_dim, seed=self.seed)

    def set_agents(self, agents):
        self.agents = agents
        self.teammates = [PolicyClone(agent, args) for agent in self.agents] if self.use_policy_clones else self.agents
        self.agents_in_training = np.ones(len(self.agents))
        self.agents_timesteps = np.zeros(len(self.agents))

    def train_agents(self, total_timesteps=1e8, exp_name=None):
        if self.fcp_ck_rate is not None:
            self.ck_list = []
            path, tag = self.save_agents(tag=f'ck_{len(self.ck_list)}')
            self.ck_list.append((0, path, tag))
        best_path, best_tag = None, None
        best_score = -1
        exp_name = exp_name or self.args.exp_name
        run = wandb.init(project="overcooked_ai_test", entity=self.args.wandb_ent,
                                  dir=str(self.args.base_dir / 'wandb'),
                                  reinit=True, name=exp_name + '_' + self.name, mode=self.args.wandb_mode)

        epoch = 0
        # Each agent should learn for `total_timesteps` steps. Keep training until all agents hit this threshold
        while any(self.agents_in_training):
            # Set new env distribution if training on multiple envs
            if self.n_layouts > 1 and self.args.multi_env_mode != 'uniform':
                self.set_new_envs()
            # Randomly select new teammates from population (can include learner)
            self.set_new_teammates()
            # Randomly choose agent that will learn this time
            learner_idx = np.random.choice(len(self.agents), p=self.agents_in_training)
            # Learn and update recoded timesteps for that agent
            self.agents[learner_idx].learn(total_timesteps=EPOCH_TIMESTEPS)
            self.agents_timesteps[learner_idx] = self.agents[learner_idx].num_timesteps
            if self.agents_timesteps[learner_idx] > total_timesteps:
                self.agents_in_training[learner_idx] = 0
            if self.use_policy_clones:
                self.teammates[learner_idx].update_policy(self.agents[learner_idx])
            # Evaluate
            if epoch % 10 == 0:
                mean_reward = self.evaluate(self.agents[learner_idx], timestep=np.sum(self.agents_timesteps))
                # FCP checkpoint saving
                if self.fcp_ck_rate and len(self.agents) == 1:
                    if self.agents_timesteps[0] // self.fcp_ck_rate > (len(self.ck_list) - 1):
                        path, tag = self.save_agents(tag=f'ck_{len(self.ck_list)}')
                        self.ck_list.append((mean_reward, path, tag))
                # Saving best model
                if mean_reward >= best_score:
                    best_path, best_tag = self.save_agents(tag='best')
                    print(f'New best score of {mean_reward} reached, model saved to {best_path}/{best_tag}')
                    best_score = mean_reward
            epoch += 1
        self.load_agents(best_path, best_tag)
        run.finish()

    def get_fcp_agents(self):
        if len(self.ck_list) < 3:
            raise ValueError('Must have at least 3 checkpoints saved. Increase fcp_ck_rate or training length')
        agents = []
        best_score = -1
        best_path, best_tag = None, None
        for score, path, tag in self.ck_list:
            if score > best_score:
                best_score = score
                best_path, best_tag = path, tag
        best = self.load_agents(best_path, best_tag)
        agents.extend(best)
        del best
        _, worst_path, worst_tag = self.ck_list[0]
        worst = self.load_agents(worst_path, worst_tag)
        agents.extend(worst)
        del worst

        closest_to_mid_score = float('inf')
        mid_path, mid_tag = None, None
        for i, (score, path, tag) in enumerate(self.ck_list):
            if abs((best_score / 2) - score) < closest_to_mid_score:
                closest_to_mid_score = score
                mid_path, mid_tag = path, tag
        mid = self.load_agents(mid_path, mid_tag)
        agents.extend(mid)
        del mid
        return agents

