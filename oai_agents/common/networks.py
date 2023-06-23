import gym
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def get_output_shape(model, image_dim):
    return model(th.rand(*(image_dim))).data.shape[1:]


def weights_init_(m):
    if hasattr(m, 'weight') and m.weight is not None and len(m.weight.shape) > 2:
        th.nn.init.xavier_uniform_(m.weight, gain=1)
    if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, th.Tensor):
        th.nn.init.constant_(m.bias, 0)


class GridEncoder(nn.Module):
    def __init__(self, grid_shape, act=nn.ReLU):
        super(GridEncoder, self).__init__()
        self.kernels = (5, 3, 3)
        self.strides = (1, 1, 1)
        self.channels = (64, 32, 32)
        self.padding = (2, 1, 1)

        layers = []
        current_channels = grid_shape[0]
        for i, (k, s, p, c) in enumerate(zip(self.kernels, self.strides, self.padding, self.channels)):
            layers.append(spectral_norm(nn.Conv2d(current_channels, c, k, stride=s, padding=p)))#spectral_norm( self.padding))
            layers.append(act())
            current_channels = c
            # depth *= 2

        layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*layers)

    def forward(self, obs):
        return self.encoder(obs.float())


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=2, act=nn.ReLU):
        super(MLP, self).__init__()
        if num_layers > 1:
            layers = [spectral_norm(nn.Linear(input_dim, hidden_dim)), act()]
        else:
            layers = [nn.Linear(input_dim, output_dim), act()]
        for _ in range(num_layers - 2):
            layers += [spectral_norm(nn.Linear(hidden_dim, hidden_dim)), act()]
        if num_layers > 1:
            layers += [spectral_norm(nn.Linear(hidden_dim, output_dim)), act()]
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs):
        return self.mlp(obs)

class OAISinglePlayerFeatureExtractor(BaseFeaturesExtractor):
    """
        :param observation_space: (gym.Space)
        :param features_dim: (int) Number of features extracted.
            This corresponds to the number of unit for the last layer.
        """

    def __init__(self, observation_space: gym.spaces.Dict, hidden_dim: int = 256):
        super(OAISinglePlayerFeatureExtractor, self).__init__(observation_space, hidden_dim)
        self.use_visual_obs = 'visual_obs' in observation_space.keys()
        self.use_vector_obs = 'agent_obs' in observation_space.keys()
        self.use_pl_comp_st = 'player_completed_subtasks' in observation_space.keys()
        self.use_tm_comp_st = 'teammate_completed_subtasks' in observation_space.keys()
        input_dim = 0
        if self.use_visual_obs:
            self.vis_encoder = GridEncoder(observation_space['visual_obs'].shape)
            test_shape = [1, *observation_space['visual_obs'].shape]
            input_dim += get_output_shape(self.vis_encoder, test_shape)[0]
        if self.use_vector_obs:
            input_dim += np.prod(observation_space['agent_obs'].shape)
        #if self.use_pl_comp_st:
        #    input_dim += np.prod(observation_space['player_completed_subtasks'].shape)
        if self.use_tm_comp_st:
            input_dim += np.prod(observation_space['teammate_completed_subtasks'].shape)

        # Define MLP for vector/feature based observations
        self.vector_encoder = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, num_layers=1)
        self.apply(weights_init_)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        latent_state = []
        # Concatenate all input features before passing them to MLP
        if self.use_visual_obs:
            # Convert all grid-like observations to features using CNN
            latent_state.append(self.vis_encoder.forward(observations['visual_obs']))
        if self.use_vector_obs:
            latent_state.append(th.flatten(observations['agent_obs'], start_dim=1))
        #if self.use_pl_comp_st:
        #    latent_state.append(th.flatten(observations['player_completed_subtasks'], start_dim=1))
        if self.use_tm_comp_st:
            latent_state.append(th.flatten(observations['teammate_completed_subtasks'], start_dim=1))

        return self.vector_encoder.forward(th.cat(latent_state, dim=-1))
