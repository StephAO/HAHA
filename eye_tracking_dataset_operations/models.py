import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import math


class LSTMFeatureBased(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, output_dim, n_layers, dropout):
        super(LSTMFeatureBased, self).__init__()

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


class LSTMLosslessEncoding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate):
        super(LSTMLosslessEncoding, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # New linear layer
        self.linear = nn.Linear(input_dim, hidden_dim)

        # LSTM layers
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate,
                            bidirectional=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Multiply by 2 for bidirectional

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Pass input through the linear layer first
        x = self.linear(x)

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)  # hidden state
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)  # cell state

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Take the output of the last time step
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)

        return out


class WarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, base_lr, max_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.max_lr = max_lr
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            lr = (self.max_lr - self.base_lr) / self.warmup_steps * self.last_epoch + self.base_lr
        else:
            lr = self.max_lr
        return [lr for _ in self.base_lrs]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.0):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model  # Store d_model as an instance attribute
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.positional_embeddings = nn.Embedding(max_len, d_model)
        nn.init.xavier_uniform_(self.positional_embeddings.weight)

    def forward(self, x):
        seq_len = x.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_embeddings = self.positional_embeddings(position_ids)

        x = x * math.sqrt(self.d_model) + position_embeddings
        x = self.layer_norm(x)
        return self.dropout(x)


class HeadTransform(nn.Module):
    # Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L664C1-L678C29
    def __init__(self, d_model):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.transform_act_fn = nn.ReLU()
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class SimpleTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, num_classes, input_dim, max_len, dropout=0.0):
        super(SimpleTransformer, self).__init__()
        self.d_model = d_model
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # nn.init.xavier_uniform_(self.cls_token)

        self.input_encoder = nn.Sequential(nn.Linear(input_dim, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout=dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.transform_head = HeadTransform(d_model)
        self.output_layer = nn.Linear(d_model, num_classes)

        self._init_parameters()

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
           Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).bool()

    def forward(self, src):
        batch_size, seq_len, _ = src.shape
        src = self.input_encoder(src)
        src = self.pos_encoder(src)

        device = src.device
        #att_mask = torch.full((seq_len, batch_size), False, device=device)
        causal_mask = self.generate_square_subsequent_mask(seq_len).to(device)

        output = self.transformer_encoder(src, mask=causal_mask, is_causal=True) #, src_key_padding_mask=att_mask
        output = self.transform_head(output)
        return self.output_layer(output)

    def _init_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                # Check if the parameter is not part of specific layers
                if 'pos_encoder.positional_embeddings.weight' not in name and 'cls_token' not in name:
                    nn.init.xavier_uniform_(p)
