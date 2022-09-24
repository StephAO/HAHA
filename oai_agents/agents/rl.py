from oai_agents.agents.base_agent import SB3Wrapper, SB3LSTMWrapper, OAITrainer
from oai_agents.common.arguments import get_arguments
from oai_agents.common.networks import OAISinglePlayerFeatureExtractor
from oai_agents.common.state_encodings import ENCODING_SCHEMES
from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv
from oai_agents.gym_environments.worker_env import OvercookedSubtaskGymEnv

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO
import wandb

EPOCH_TIMESTEPS = 10000
VEC_ENV_CLS = DummyVecEnv#SubprocVecEnv


class SingleAgentTrainer(OAITrainer):
    ''' Train an RL agent to play with a provided agent '''
    def __init__(self, teammates, args, name=None, env=None, eval_env=None, use_lstm=False, hidden_dim=256, seed=None):
        name = name or 'two_single_agents'
        super(SingleAgentTrainer, self).__init__(name, args, seed=seed)
        self.args = args
        self.device = args.device
        self.use_lstm = use_lstm
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
        self.teammates = teammates
        self.n_tm = len(teammates)
        env_kwargs = {'shape_rewards': True, 'args': args}
        if env is None:
            self.env = make_vec_env(OvercookedGymEnv, n_envs=args.n_envs, env_kwargs=env_kwargs,
                                    vec_env_cls=VEC_ENV_CLS)
            self.eval_env = OvercookedGymEnv(shape_rewards=False, args=args)
        else:
            self.env = env
            self.eval_env = eval_env
        self.use_subtask_eval = (type(eval_env) == OvercookedSubtaskGymEnv)

        policy_kwargs = dict(
            features_extractor_class=OAISinglePlayerFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=hidden_dim),
            net_arch=[dict(pi=[hidden_dim, hidden_dim], vf=[hidden_dim, hidden_dim])]
        )
        if use_lstm:
            sb3_agent = RecurrentPPO('MultiInputLstmPolicy', self.env, policy_kwargs=policy_kwargs, verbose=1)
        else:
            sb3_agent = PPO('MultiInputPolicy', self.env, policy_kwargs=policy_kwargs, verbose=1)
        # sb3_agent.policy.to(self.device)
        self.learning_agent = self.wrap_agent(sb3_agent)
        self.agents = [self.learning_agent]
        # for agent in self.agents:
        #     agent.policy.to(self.device)

        # for i in range(self.args.n_envs):
        #     self.env.env_method('set_agent', self.teammates[np.random.randint(self.n_tm)], indices=i)

    def _get_constructor_parameters(self):
        return dict(args=self.args, name=self.name, use_lstm=self.use_lstm, hidden_dim=self.hidden_dim, seed=self.seed)

    def wrap_agent(self, sb3_agent):
        if self.use_lstm:
            agent = SB3LSTMWrapper(sb3_agent, f'rl_single_lstm_agent', self.args)
        else:
            agent = SB3Wrapper(sb3_agent, f'rl_single_agent', self.args)
        return agent

    def train_agents(self, total_timesteps=1e8, exp_name=None):
        exp_name = exp_name or self.args.exp_name
        run = wandb.init(project="overcooked_ai_test", entity=self.args.wandb_ent,
                         dir=str(self.args.base_dir / 'wandb'),
                         reinit=True, name=exp_name + '_' + self.learning_agent.name, mode=self.args.wandb_mode)
        if self.use_subtask_eval:
            self.num_success = 0
        while self.learning_agent.num_timesteps < total_timesteps:
            self.set_new_teammates()
            self.learning_agent.learn(total_timesteps=EPOCH_TIMESTEPS)
            eval_teammate = self.teammates[np.random.randint(len(self.teammates))]
            if self.use_subtask_eval:
                self.eval_env.set_teammate(eval_teammate)
                all_successes = self.eval_env.evaluate(self.learning_agent)
                self.num_success = (self.num_success + 1) if all_successes else 0
                if self.num_success >= 3:
                    break
            else:
                self.evaluate(self.learning_agent, eval_teammate)
        path, tag = self.save_agents()
        run.finish()


class MultipleAgentsTrainer(OAITrainer):
    ''' Train two independent RL agents to play with each other '''

    def __init__(self, args, name=None, num_agents=1, use_lstm=False, hidden_dim=256, fcp_ck_rate=None, seed=None):
        '''
        Train multiple agents with each other.
        :param num_agents: Number of agents to train. num_agents=1 mean self-play, num_agents > 1 is population play
        :param args: Experiment arguments. See arguments.py
        :param use_lstm: Whether agents should use an lstm policy or not
        :param hidden_dim: hidden dimensions for agents
        :param fcp_ck_rate: If not none, rate to save agents. Used primarily to get agents for Fictitous Co-Play
        :param seed: Random see
        '''
        name = name or 'two_single_agents'
        super(MultipleAgentsTrainer, self).__init__(name, args, seed=seed)
        self.device = args.device
        self.args = args
        self.fcp_ck_rate = fcp_ck_rate

        env_kwargs = {'shape_rewards': True, 'args': args}
        self.env = make_vec_env(OvercookedGymEnv, n_envs=args.n_envs, env_kwargs={**env_kwargs})
        self.eval_env = OvercookedGymEnv(shape_rewards=False, args=args)

        policy_kwargs = dict(
            features_extractor_class=OAISinglePlayerFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=hidden_dim),
            net_arch=[dict(pi=[hidden_dim, hidden_dim], vf=[hidden_dim, hidden_dim])]
        )

        self.agents = []
        if use_lstm:
            for i in range(num_agents):
                sb3_agent = RecurrentPPO('MultiInputLstmPolicy', self.env, policy_kwargs=policy_kwargs, verbose=1,
                                         n_steps=1024, batch_size=16)
                self.agents.append(SB3LSTMWrapper(sb3_agent, f'rl_multiagent_lstm_{i + 1}', args))
        else:
            for i in range(num_agents):
                sb3_agent = PPO("MultiInputPolicy", self.env, policy_kwargs=policy_kwargs, verbose=1)
                self.agents.append(SB3Wrapper(sb3_agent, f'rl_multiagent_{i + 1}', args))

        self.teammates = self.agents
        self.agents_in_training = np.ones(len(self.agents))
        self.agents_timesteps = np.zeros(len(self.agents))

    def set_agents(self, agents):
        self.agents = agents
        self.teammates = self.agents
        self.agents_in_training = np.ones(len(self.agents))
        self.agents_timesteps = np.zeros(len(self.agents))

    def train_agents(self, total_timesteps=1e8, exp_name=None):
        if self.fcp_ck_rate is not None:
            self.ck_list = []
            path, tag = self.save_agents(tag=f'ck_{len(self.ck_list)}')
            self.ck_list.append((0, path, tag))
        exp_name = exp_name or self.args.exp_name
        run = wandb.init(project="overcooked_ai_test", entity=self.args.wandb_ent,
                         dir=str(self.args.base_dir / 'wandb'),
                         reinit=True, name=exp_name + '_rl_two_single_agents', mode=self.args.wandb_mode)
        # Each agent should learn for `total_timesteps` steps. Keep training until all agents hit this threshold
        while any(self.agents_in_training):
            # Randomly select new teammates from population (can include learner)
            self.set_new_teammates()
            # Randomly choose agent that will learn this time
            learner_idx = np.random.choice(len(self.agents), p=self.agents_in_training)
            # Learn and update recoded timesteps for that agent
            self.agents[learner_idx].learn(total_timesteps=EPOCH_TIMESTEPS)
            self.agents_timesteps[learner_idx] = self.agents[learner_idx].num_timesteps
            if self.agents_timesteps[learner_idx] > total_timesteps:
                self.agents_in_training[learner_idx] = 0
            # Evaluate
            eval_tm = self.teammates[np.random.randint(len(self.teammates))]
            mean_reward = self.evaluate(self.agents[learner_idx], eval_tm, timestep=np.sum(self.agents_timesteps))

            if self.fcp_ck_rate and len(self.agents) == 1:
                if self.agents_timesteps[0] // self.fcp_ck_rate > (len(self.ck_list) - 1):
                    path, tag = self.save_agents(tag=f'ck_{len(self.ck_list)}')
                    self.ck_list.append((mean_reward, path, tag))
        run.finish()

    def get_fcp_agents(self):
        if len(self.ck_list) < 3:
            raise ValueError('Must have at least 3 checkpoints saved. Increase fcp_ck_rate or training length')
        agents = []
        best_score = 0
        best_path, best_tag = None, None
        for score, path, tag in self.ck_list:
            if score > best_score:
                best_score = score
                best_path, best_tag = path, tag
        best = self.load_agents(best_path, best_tag)
        agents.append(best.get_agents())
        del best
        _, worst_path, worst_tag = self.ck_list[0]
        worst = self.load_agents(worst_path, worst_tag)
        agents.append(worst.get_agents())
        del worst

        closest_to_mid_score = float('inf')
        mid_path, mid_tag = None, None
        for i, (score, path, tag) in enumerate(self.ck_list):
            if abs((best_score / 2) - score) < closest_to_mid_score:
                closest_to_mid_score = score
                mid_path, mid_tag = path, tag
        mid = self.load_agents(mid_path, mid_tag)
        agents.append(mid.get_agents())
        del mid
        return agents

    ### BASELINES ###
    @staticmethod
    def create_selfplay_agent(args, training_steps=1e8):
        self_play_trainer = MultipleAgentsTrainer(args, name='selfplay', num_agents=1, use_lstm=False)
        self_play_trainer.train_agents(total_timesteps=training_steps)
        return self_play_trainer.get_agents()

    @staticmethod
    def create_fcp_population(args, training_steps=1e8):
        agents = []
        for use_lstm in [True, False]:
            # hidden_dim = 16
            seed = 8
            for h_dim in [256, 16]:
                #     for seed in [1, 20]:#, 300, 4000]:
                ck_rate = training_steps / 10
                name = f'lstm_{h_dim}' if use_lstm else f'no_lstm_{h_dim}'
                mat = MultipleAgentsTrainer(args, num_agents=1, use_lstm=use_lstm, hidden_dim=h_dim,
                                            fcp_ck_rate=ck_rate, seed=seed)
                mat.train_agents(total_timesteps=training_steps)
                agents.extend(rl_sat.get_fcp_agents())
        pop = MultipleAgentsTrainer(args, num_agents=0)
        pop.set_agents(agents)
        pop.save_agents(str(self.args.base_dir / 'agent_models' / 'fcp' / self.args.layout_name / f'{len(agents)}_pop'))
        return pop.get_agents()


if __name__ == '__main__':
    args = get_arguments()
    sp = MultipleAgentsTrainer.create_selfplay_agent(args, training_steps=1e6)
    # pop = MultipleAgentsTrainer.create_fcp_population(args, training_steps=3e6)
    # fcp = SingleAgentTrainer(pop, args, 'fcp')
    # fcp.train_agents(1e6)


