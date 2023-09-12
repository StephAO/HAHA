import torch
import torch.nn as nn

class ProficiencyPredictor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, output_dim, n_layers, dropout):
        super(ProficiencyPredictor, self).__init__()

        # Encoding layers
        self.obs_encoder = nn.Linear(obs_dim, hidden_dim)
        self.action_encoder = nn.Linear(action_dim, hidden_dim)

        # LSTM layer
        self.lstm = nn.LSTM(2*hidden_dim, hidden_dim, num_layers=n_layers,
                            dropout=dropout, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(2*hidden_dim, output_dim)

    def forward(self, agent_obs, action):
        # Encode observations and actions
        encoded_obs = self.obs_encoder(agent_obs)
        encoded_action = self.action_encoder(action)

        # Concatenate the encoded outputs
        lstm_input = torch.cat((encoded_obs, encoded_action), dim=-1)

        # Alternatively, to alternate between them:
        # lstm_input = torch.stack([encoded_obs, encoded_action], dim=1).view(agent_obs.size(0), agent_obs.size(1) * 2, -1)

        lstm_out, _ = self.lstm(lstm_input)
        outputs = self.fc(lstm_out)

        return outputs

def compute_bin_accuracies(conf_matrix):
    # Extracting diagonal (true positives for each class)
    true_positives = conf_matrix.diagonal()
    # Summing each column (total samples per class)
    total_per_class = conf_matrix.sum(axis=1)
    return true_positives / total_per_class