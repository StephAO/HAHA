import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn


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
    def __init__(self, d_model, max_len=155, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Learnable positional encodings
        self.pe = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x):
        seq_len = x.size(1)  # Get the sequence length from the input
        pos_encoding = self.pe[:seq_len, :]  # Adjust the positional encoding size
        x = x + pos_encoding.unsqueeze(0)  # Add positional encoding to each sequence in the batch
        return self.dropout(x)


class SimpleTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, num_classes, input_dim, dropout=0.1):
        super(SimpleTransformer, self).__init__()
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # [CLS] token initialization
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, src):
        batch_size = src.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Repeat the CLS token for each sequence in the batch
        src = self.input_linear(src)
        src = torch.cat((cls_tokens, src), dim=1)  # Prepend CLS token
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        cls_output = output[:, 0, :]  # Extract the output corresponding to the CLS token
        return self.output_layer(cls_output)
