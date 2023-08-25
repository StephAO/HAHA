from oai_agents.agents.base_agent import OAIAgent, OAITrainer
from oai_agents.common.arguments import get_arguments
from oai_agents.common.networks import GridEncoder, MLP, weights_init_, get_output_shape
from oai_agents.common.overcooked_dataset import OvercookedDataset
from oai_agents.common.state_encodings import ENCODING_SCHEMES
from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv
from oai_agents.agents.agent_utils import load_agent

from overcooked_ai_py.mdp.overcooked_mdp import Action

from gym import spaces
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.distributions.categorical import Categorical
from typing import Dict, Any, Union
import wandb


class BehaviouralCloningPolicy(nn.Module):
    def __init__(self, visual_obs_shape, agent_obs_shape, args, act=nn.ReLU, hidden_dim=64):
        """
        NN network for a behavioral cloning agent
        :param visual_obs_shape: Shape of any grid-like input to be passed into a CNN
        :param agent_obs_shape: Shape of any vector input to passed only into an MLP
        :param depth: Depth of CNN
        :param act: activation function
        :param hidden_dim: hidden dimension to use in NNs
        """
        super(BehaviouralCloningPolicy, self).__init__()
        self.device = args.device
        self.use_visual_obs = np.prod(visual_obs_shape) > 0
        self.use_agent_obs = np.prod(agent_obs_shape) > 0
        self.obs_dict = {}

        # Define CNN for grid-like observations
        if self.use_visual_obs:
            self.obs_dict['visual_obs'] = spaces.Box(0, 20, visual_obs_shape, dtype=int)
            self.cnn = GridEncoder(visual_obs_shape)
            self.cnn_output_shape = get_output_shape(self.cnn, [1, *visual_obs_shape])[0]
        else:
            self.obs_dict['visual_obs'] = spaces.Box(0, 1, (1,), dtype=int)
            self.cnn_output_shape = 0

        if self.use_agent_obs:
            self.obs_dict['agent_obs'] = spaces.Box(0, 400, agent_obs_shape, dtype=int)
        self.observation_space = spaces.Dict(self.obs_dict)

        # Define MLP for vector/feature based observations
        self.mlp = MLP(input_dim=int(self.cnn_output_shape + np.prod(agent_obs_shape)),
                       output_dim=hidden_dim, num_layers=2, hidden_dim=hidden_dim, act=act)
        self.action_predictor = nn.Linear(hidden_dim, Action.NUM_ACTIONS)

        self.apply(weights_init_)
        self.to(self.device)

    def get_latent_feats(self, obs):
        mlp_input = []
        # Concatenate all input features before passing them to MLP
        if self.use_visual_obs:
            # Add batch dim, avoids broadcasting errors down the line
            if len(obs['visual_obs'].shape) == 3:
                obs['visual_obs'] = obs['visual_obs'].unsqueeze(0)
            mlp_input.append(self.cnn.forward(obs['visual_obs']))
        if self.use_agent_obs:
            # Add batch dim, avoids broadcasting errors down the line
            if len(obs['agent_obs'].shape) == 3:
                obs['agent_obs'] = obs['agent_obs'].unsqueeze(0)
            mlp_input.append(obs['agent_obs'])
        return self.mlp.forward(th.cat(mlp_input, dim=-1).float())

    def forward(self, obs):
        return self.action_predictor(self.get_latent_feats(obs))

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        """Predict action. If sample is True, sample action from distribution, else pick best scoring action"""
        return Categorical(logits=self.forward(obs)).sample() if deterministic else th.argmax(self.forward(obs),
                                                                                              dim=-1), None

    def get_distribution(self, obs):
        return Categorical(logits=self.forward(obs))


class BehaviouralCloningAgent(OAIAgent):
    def __init__(self, visual_obs_shape, agent_obs_shape, args, hidden_dim=64, name=None):
        name = name or 'bc'
        super(BehaviouralCloningAgent, self).__init__(name, args)
        self.encoding_fn = ENCODING_SCHEMES['OAI_feats']
        self.visual_obs_shape, self.agent_obs_shape, self.args, self.hidden_dim = \
            visual_obs_shape, agent_obs_shape, args, hidden_dim
        self.device = args.device
        self.policy = BehaviouralCloningPolicy(visual_obs_shape, agent_obs_shape, args, hidden_dim=hidden_dim)
        self.observation_space = self.policy.observation_space
        self.to(self.device)
        self.num_timesteps = 0

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """
        Get data that need to be saved in order to re-create the model when loading it from disk.
        :return: The dictionary to pass to the as kwargs constructor when reconstruction this model.
        """
        return dict(
            visual_obs_shape=self.visual_obs_shape,
            agent_obs_shape=self.agent_obs_shape,
            hidden_dim=self.hidden_dim
        )

    def forward(self, obs):
        z = self.policy.get_latent_feats(obs)
        return self.policy.action_predictor(z)

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        deterministic = False # BC Agents are just much better is this is False
        obs = {k: th.tensor(v, device=self.device) for k, v in obs.items()}
        action_logits = self.forward(obs)
        action = th.argmax(action_logits, dim=-1) if deterministic else Categorical(logits=action_logits).sample()
        return action, None

    def get_distribution(self, obs: th.Tensor):
        obs = {k: th.tensor(v, device=self.device).unsqueeze(0) for k, v in obs.items()}
        return self.policy.get_distribution(obs)


class BehavioralCloningTrainer(OAITrainer):
    def __init__(self, dataset, args, name=None, layout_names=None, vis_eval=False):
        """
        Class to train BC agent
        :param env: Overcooked environment to use
        :param dataset: That dataset to train on - can be None if the only visualizing agetns
        :param args: arguments to use
        :param vis_eval: If true, the evaluate function will visualize the agents
        """
        name = name or 'bc'
        super(BehavioralCloningTrainer, self).__init__(name, args)
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.num_players = 2
        self.dataset = dataset
        self.datasets = None
        self.layout_names = layout_names or args.layout_names
        self.eval_envs = [OvercookedGymEnv(shape_rewards=False, is_eval_env=True, enc_fn='OAI_feats', horizon=400,
                                           layout_name=ln, args=args) for ln in self.layout_names]
        obs = self.eval_envs[0].get_obs(p_idx=0)
        visual_obs_shape = obs['visual_obs'].shape if 'visual_obs' in obs else 0
        agent_obs_shape = obs['agent_obs'].shape if 'agent_obs' in obs else 0
        self.agents = [BehaviouralCloningAgent(visual_obs_shape, agent_obs_shape, args, name=name+'1'),
                       BehaviouralCloningAgent(visual_obs_shape, agent_obs_shape, args, name=name+'2')]
        self.bc, self.human_proxy = None, None
        self.teammates = self.agents
        self.optimizers = [th.optim.Adam(self.agents[0].parameters(), lr=args.lr),
                           th.optim.Adam(self.agents[1].parameters(), lr=args.lr)]
        if vis_eval:
            self.eval_envs.setup_visualization()

    def setup_datasets(self):
        self.full_ds = OvercookedDataset(self.dataset, self.layout_names, self.args)
        train_size = len(self.full_ds) // 2
        train1, train2 = random_split(self.full_ds, [train_size, train_size])
        self.datasets = [train1, train2]
        action_weights = th.tensor(self.full_ds.get_action_weights(), dtype=th.float32, device=self.device)
        self.action_criterion = nn.CrossEntropyLoss(weight=action_weights)

    def run_batch(self, agent_idx, batch):
        """Train BC agent on a batch of data"""
        batch = {k: v.to(self.device) for k, v in batch.items()}
        # train agent on both players actions
        self.optimizers[agent_idx].zero_grad()
        preds = self.agents[agent_idx].forward({'agent_obs': batch['agent_obs']})


        loss = self.action_criterion(preds, batch['action'].long())
        loss.backward()
        self.optimizers[agent_idx].step()
        self.agents[agent_idx].num_timesteps += self.args.batch_size
        return loss.item()

    def run_epoch(self, agent_idx):
        self.agents[agent_idx].train()
        losses = []
        dl = DataLoader(self.datasets[agent_idx], batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        for batch in tqdm(dl):
            losses.append(self.run_batch(agent_idx, batch))
        self.agents[agent_idx].eval()
        return np.mean(losses)

    def train_agents(self, epochs=100, exp_name=None):
        """ Training routine """
        if self.datasets is None:
            self.setup_datasets()
        exp_name = exp_name or self.args.exp_name
        run = wandb.init(project="overcooked_ai", entity=self.args.wandb_ent,
                         dir=str(self.args.base_dir / 'wandb'),
                         reinit=True, name=exp_name + '_' + self.name, mode=self.args.wandb_mode)

        best_rew, best_path = [float('-inf'), float('-inf')], [None, None]
        best_loss = float('inf')
        for epoch in range(epochs):
            for i in range(2):
                train_loss = self.run_epoch(i)
                self.eval_teammates = [self.agents[i]]
                if best_loss < train_loss or epoch % 10 == 0:
                    best_loss = train_loss
                    mean_sp_reward, _ = self.evaluate(self.agents[i], num_eps_per_layout_per_tm=10, timestep=epoch, deterministic=False)
                    wandb.log({f'train_loss_{i}': train_loss, 'timestep': epoch})
                    if mean_sp_reward > best_rew[i]:
                        print(f'Saving new best BC model for agent {i} on epoch {epoch} with reward {mean_sp_reward}')
                        path = self.args.base_dir / 'agent_models' / self.name / self.args.exp_name / 'agents_dir'
                        Path(path).mkdir(parents=True, exist_ok=True)
                        self.agents[i].save(path / f'agent_{i}')
                        best_path[i] = path / f'agent_{i}'
                        best_rew[i] = mean_sp_reward

        # reload best ck
        for i in range(2):
            self.agents[i] = BehaviouralCloningAgent.load(best_path[i], self.args)
            self.agents[i].to(self.device)

        self.compare_agents()
        run.finish()

    def compare_agents(self):
        rewards = [-1, -1]
        for i in range(2):
            self.eval_teammates = [self.agents[i]]
            rewards[i], _ = self.evaluate(self.agents[i], num_eps_per_layout_per_tm=10, deterministic=False,
                                          log_wandb=False)
        self.bc, self.human_proxy = (self.agents[0], self.agents[1]) if rewards[1] > rewards[0] else\
                                    (self.agents[1], self.agents[0])
        self.save_bc_and_human_proxy()


    def save_bc_and_human_proxy(self, path: Union[Path, None] = None, tag: Union[str, None] = None):
        ''' Saves bc and human proxy that the trainer is training '''
        path = path or self.args.base_dir / 'agent_models' / self.name
        tag = tag or self.args.exp_name
        agent_path = path / tag / 'agents_dir'
        Path(agent_path).mkdir(parents=True, exist_ok=True)
        self.bc.save(agent_path / f'bc')
        self.human_proxy.save(agent_path / f'human_proxy')
        return path, tag


    @staticmethod
    def load_bc_and_human_proxy(args, name: str=None, path: Union[Path, None] = None, tag: Union[str, None] = None):
        ''' Loads each agent that the trainer is training '''
        path = path or args.base_dir / 'agent_models' / name
        tag = tag or args.exp_name
        agent_path = path / tag / 'agents_dir'
        # Load weights
        bc = load_agent(agent_path / 'bc', args).to(args.device)
        human_proxy = load_agent(agent_path / 'human_proxy', args).to(args.device)
        return bc, human_proxy

    def get_agents(self):
        if self.bc is None:
            self.compare_agents()
        return self.bc, self.human_proxy



if __name__ == '__main__':
    args = get_arguments()
    eval_only = False
    if eval_only:
        bct = BehavioralCloningTrainer('tf_test_5_5.2.pickle', args, vis_eval=True)
    else:
        args.batch_size = 4
        args.layout_names = ['tf_test_5_5']
        bct = BehavioralCloningTrainer('tf_test_5_5.2.pickle', args, vis_eval=True)
