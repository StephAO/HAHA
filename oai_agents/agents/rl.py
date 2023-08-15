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

VEC_ENV_CLS = DummyVecEnv #

class RLAgentTrainer(OAITrainer):
    ''' Train an RL agent to play with a provided agent '''
    def __init__(self, teammates, args, selfplay=False, name=None, env=None, eval_envs=None,
                 use_cnn=False, use_lstm=False, use_frame_stack=False, taper_layers=False, use_subtask_counts=False,
                 use_policy_clone=False, num_layers=2, hidden_dim=256, use_subtask_eval=False, use_hrl=False,
                 fcp_ck_rate=None, seed=None):
        name = name or 'rl_agent'
        super(RLAgentTrainer, self).__init__(name, args, seed=seed)
        if not teammates and not selfplay:
            raise ValueError('Either a list of teammates with len > 0 must be passed in or selfplay must be true')
        self.args = args
        self.device = args.device
        self.use_lstm = use_lstm
        self.use_frame_stack = use_frame_stack
        self.use_subtask_eval = use_subtask_eval
        self.using_hrl = use_hrl
        self.use_policy_clone = use_policy_clone
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.fcp_ck_rate = fcp_ck_rate
        self.encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
        self.run_id = None
        if env is None:
            env_kwargs = {'shape_rewards': True, 'ret_completed_subtasks': use_subtask_counts,
                          'stack_frames': use_frame_stack, 'full_init': False, 'args': args}
            self.env = make_vec_env(OvercookedGymEnv, n_envs=args.n_envs, seed=seed,
                                    vec_env_cls=VEC_ENV_CLS, env_kwargs=env_kwargs)

            eval_envs_kwargs = {'is_eval_env': True, 'ret_completed_subtasks': use_subtask_counts,
                                'stack_frames': use_frame_stack, 'horizon': 400, 'args': args}
            self.eval_envs = [OvercookedGymEnv(**{'env_index': i, **eval_envs_kwargs}) for i in range(self.n_layouts)]
        else:
            self.env = env
            self.eval_envs = eval_envs
        for i in range(self.args.n_envs):
            self.env.env_method('set_env_layout', indices=i, env_index=i % self.n_layouts)

        layers = [hidden_dim // (2**i) for i in range(num_layers)] if taper_layers else [hidden_dim] * num_layers
        policy_kwargs = dict(net_arch=[dict(pi=layers, vf=layers)])
        if use_cnn:
            print('USING CNN')
            policy_kwargs.update(
                features_extractor_class=OAISinglePlayerFeatureExtractor,
                features_extractor_kwargs=dict(hidden_dim=hidden_dim)
            )

        self.epoch_timesteps = 1e6
        if use_lstm:
            policy_kwargs['n_lstm_layers'] = 2
            policy_kwargs['lstm_hidden_size'] = hidden_dim
            sb3_agent = RecurrentPPO('MultiInputLstmPolicy', self.env, policy_kwargs=policy_kwargs, verbose=1,
                                     n_steps=500, n_epochs=4, batch_size=500)
            agent_name = f'{name}_lstm'
        elif use_hrl:
            sb3_agent = MaskablePPO('MultiInputPolicy', self.env, policy_kwargs=policy_kwargs, verbose=1, n_steps=500,
                                    n_epochs=4, learning_rate=0.0003, batch_size=500, ent_coef=0.001, vf_coef=0.3,
                                    gamma=0.99, gae_lambda=0.95)
            agent_name = f'{name}'
        else:
            sb3_agent = PPO("MultiInputPolicy", self.env, policy_kwargs=policy_kwargs, verbose=1, n_steps=500,
                            n_epochs=4, learning_rate=0.0003, batch_size=500, ent_coef=0.001, vf_coef=0.3,
                            gamma=0.99, gae_lambda=0.95)
            agent_name = f'{name}'
        self.learning_agent = self.wrap_agent(sb3_agent, agent_name)
        self.agents = [self.learning_agent]

        self.teammates = teammates if teammates else []
        if selfplay:
            self.teammates += self.agents
        self.eval_teammates = self.teammates
        self.epoch, self.total_game_steps = 0, 0
        self.best_score, self.fewest_failures, self.best_training_rew = -1, float('inf'), float('-inf')

    def _get_constructor_parameters(self):
        return dict(args=self.args, name=self.name, use_lstm=self.use_lstm, use_frame_stack=self.use_frame_stack,
                    hidden_dim=self.hidden_dim, seed=self.seed)

    def wrap_agent(self, sb3_agent, name):
        if self.use_lstm:
            agent = SB3LSTMWrapper(sb3_agent, name, self.args)
        else:
            agent = SB3Wrapper(sb3_agent, name, self.args)
        return agent

    def train_agents(self, train_timesteps=2e6, exp_name=None):
        exp_name = exp_name or self.args.exp_name
        run = wandb.init(project="overcooked_ai", entity=self.args.wandb_ent, dir=str(self.args.base_dir / 'wandb'),
                         reinit=True, name=exp_name + '_' + self.name, mode=self.args.wandb_mode, id=self.run_id,
                         resume="allow")
        if self.run_id is None:
            self.run_id = run.id

        if self.fcp_ck_rate is not None:
            self.ck_list = []
            path, tag = self.save_agents(tag=f'ck_{len(self.ck_list)}')
            self.ck_list.append(({k: 0 for k in self.args.layout_names}, path, tag))
        best_path, best_tag = None, None

        curr_timesteps = 0
        prev_timesteps = self.learning_agent.num_timesteps
        while curr_timesteps < train_timesteps:
            print(curr_timesteps, self.learning_agent.num_timesteps)
            self.set_new_teammates()
            self.learning_agent.learn(total_timesteps=self.epoch_timesteps)
            curr_timesteps += (self.learning_agent.num_timesteps - prev_timesteps)
            prev_timesteps = curr_timesteps

            if self.use_policy_clone:
                self.update_pc(self.epoch)
            if self.using_hrl:
                failure_dicts = self.env.env_method('get_worker_failures')
                tot_failure_dict = {ln: {k: 0 for k in failure_dicts[0][1].keys()} for ln in self.args.layout_names}
                for ln, fd in failure_dicts:
                    for k in tot_failure_dict[ln]:
                        tot_failure_dict[ln][k] += fd[k]
                for ln in self.args.layout_names:
                    wandb.log({f'num_worker_failures_{ln}': sum(tot_failure_dict[ln].values()), 'timestep': self.learning_agent.num_timesteps})
                    print(f'Number of worker failures on {ln}: {tot_failure_dict[ln]}')

            # Evaluate
            mean_training_rew = np.mean([ep_info["r"] for ep_info in self.learning_agent.agent.ep_info_buffer])
            self.best_training_rew *= 0.98
            if (self.epoch + 1) % 5 == 0 or (mean_training_rew > self.best_training_rew and self.learning_agent.num_timesteps >= 1e6) or \
                (self.fcp_ck_rate and self.learning_agent.num_timesteps // self.fcp_ck_rate > (len(self.ck_list) - 1)):
                if mean_training_rew >= self.best_training_rew:
                    self.best_training_rew = mean_training_rew

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
                    if tot_failures <= self.fewest_failures:
                        best_path, best_tag = self.save_agents(tag='best')
                        self.fewest_failures = tot_failures
                        print(f'New fewest failures of {self.fewest_failures} reached, model saved to {best_path}/{best_tag}')
                    if all(env_success):
                        break
                else:
                    mean_reward, rew_per_layout = self.evaluate(self.learning_agent, timestep=self.learning_agent.num_timesteps)
                    # FCP pop checkpointing
                    if self.fcp_ck_rate:
                        if self.learning_agent.num_timesteps // self.fcp_ck_rate > (len(self.ck_list) - 1):
                            path, tag = self.save_agents(tag=f'ck_{len(self.ck_list)}')
                            self.ck_list.append((rew_per_layout, path, tag))
                    # Save best model
                    if mean_reward >= self.best_score:
                        best_path, best_tag = self.save_agents(tag='best')
                        print(f'New best score of {mean_reward} reached, model saved to {best_path}/{best_tag}')
                        self.best_score = mean_reward
            self.epoch += 1
        self.save_agents()
        self.agents = RLAgentTrainer.load_agents(self.args, self.name, best_path, best_tag)
        run.finish()

    def get_fcp_agents(self, layout_name):
        if len(self.ck_list) < 3:
            raise ValueError('Must have at least 3 checkpoints saved. Increase fcp_ck_rate or training length')
        agents = []
        # Best agent for this layout
        self.best_score = -1
        best_path, best_tag = None, None
        for scores, path, tag in self.ck_list:
            score = scores[layout_name]
            if score > self.best_score:
                self.best_score = score
                best_path, best_tag = path, tag
        best = RLAgentTrainer.load_agents(self.args, path=best_path, tag=best_tag)
        agents.extend(best)
        del best
        # Worst agent for this layout
        _, worst_path, worst_tag = self.ck_list[0]
        worst = RLAgentTrainer.load_agents(self.args, path=worst_path, tag=worst_tag)
        agents.extend(worst)
        del worst
        # Middle agent for this layout
        closest_to_mid_score = float('inf')
        mid_path, mid_tag = None, None
        for i, (scores, path, tag) in enumerate(self.ck_list):
            score = scores[layout_name]
            if abs((self.best_score / 2) - score) < closest_to_mid_score:
                closest_to_mid_score = score
                mid_path, mid_tag = path, tag
        mid = RLAgentTrainer.load_agents(self.args, path=mid_path, tag=mid_tag)
        agents.extend(mid)
        del mid
        return agents
