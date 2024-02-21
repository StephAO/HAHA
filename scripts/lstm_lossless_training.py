from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from eye_tracking_dataset_operations.memmap_create_and_retrieve import return_memmaps
from scripts import models
from eye_tracking_dataset_operations.eye_gaze_and_play_dataset import process_trial_data, process_data

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from oai_agents.common.arguments_transformer import LSTMConfig

# Memmap file paths
# TODO define these all in one place
participant_memmap_file = "/data/participant_memmap.dat"  # "/HAHA-eyetracking/data/participant_memmap.dat" "path/to/memmap/participant_memmap.dat"
obs_heatmap_memmap_file = "/data/obs_heatmap_memmap.dat"  # "/HAHA-eyetracking/data/obs_heatmap_memmap.dat" "path/to/memmap/obs_heatmap_memmap.dat"



num_timesteps_to_consider = LSTMConfig.num_timesteps_to_consider

# Model Initialization
d_model = LSTMConfig.d_model
nhead = LSTMConfig.nhead
num_layers = LSTMConfig.num_layers
dim_feedforward = LSTMConfig.dim_feedforward
num_classes = LSTMConfig.num_classes
input_dim = LSTMConfig.input_dim
hidden_dim = LSTMConfig.hidden_dim
output_dim = LSTMConfig.output_dim
dropout_rate = LSTMConfig.dropout_rate

# warmup

warmup_steps = LSTMConfig.warmup_steps
base_lr = LSTMConfig.base_lr
max_lr = LSTMConfig.max_lr
num_epochs = LSTMConfig.num_epochs

# only needed initially to make the memmaps, please comment out after the memmaps are created.
# setup_and_process_xdf_files("/path/t0/folder/with/xdfFiles", participant_memmap_file,obs_heatmap_memmap_file) # /path/t0/folder/with/xdfFiles


participant_memmap, obs_heatmap_memmap, subtask_memmap, gaze_obj_memmap = return_memmaps(participant_memmap_file, obs_heatmap_memmap_file, subtask_memmap_file, gaze_obj_memmap_file)

trial_data, trial_labels = process_data(participant_memmap, obs_heatmap_memmap, num_timesteps_to_consider)
processed_data = process_trial_data(trial_data, trial_labels)

participant_ids = list(processed_data.keys())

train_participant_ids, val_participant_ids = train_test_split(participant_ids, test_size=0.2, random_state=42)
print(f"train: {train_participant_ids}")
print(f"test: {val_participant_ids}")
X_train = [data for pid in train_participant_ids for data in processed_data[pid]['data']]
y_train = [labels for pid in train_participant_ids for labels in processed_data[pid]['labels']]
X_val = [data for pid in val_participant_ids for data in processed_data[pid]['data']]
y_val = [labels for pid in val_participant_ids for labels in processed_data[pid]['labels']]


# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Create dataset instances
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Initialize the LSTM model
lstm_model = models.LSTMLosslessEncoding(input_dim, hidden_dim, output_dim, num_layers, dropout_rate)

# Criterion and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=1e-4, weight_decay=1e-5)

# Assuming train_dataloader and val_dataloader are defined

train_losses = []
train_f1_scores = []
val_losses = []
val_f1_scores = []
train_bin_accuracies = [[] for _ in range(num_classes)]
val_bin_accuracies = [[] for _ in range(num_classes)]

train_correct_per_class = np.zeros(num_classes)
train_total_per_class = np.zeros(num_classes)
val_correct_per_class = np.zeros(num_classes)
val_total_per_class = np.zeros(num_classes)

# Training Loop

for epoch in range(num_epochs):
    lstm_model.train()
    total_loss, total_f1, total_samples = 0, 0, 0
    train_correct_per_class.fill(0)
    train_total_per_class.fill(0)

    for data, labels in train_dataloader:
        # Prepare data and labels
        if labels.dim() == 2 and labels.size(1) > 1:
            labels = labels[:, -1]

        optimizer.zero_grad()
        outputs = lstm_model(data)
        if labels.dim() > 1:
            labels = labels[:, -1]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total_loss += loss.item()
        f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
        total_f1 += f1
        total_samples += 1

        for label, pred in zip(labels, predicted):
            train_total_per_class[label] += 1
            if label == pred:
                train_correct_per_class[label] += 1

    train_loss = total_loss / len(train_dataloader)
    train_f1_score = total_f1 / total_samples
    train_losses.append(train_loss)
    train_f1_scores.append(train_f1_score)

    for i in range(num_classes):
        if train_total_per_class[i] > 0:
            train_bin_accuracies[i].append(train_correct_per_class[i] / train_total_per_class[i])

    # Validation phase
    lstm_model.eval()
    total_loss, total_f1, total_samples = 0, 0, 0
    val_correct_per_class.fill(0)
    val_total_per_class.fill(0)

    with torch.no_grad():
        for data, labels in val_dataloader:
            if labels.dim() == 2 and labels.size(1) > 1:
                labels = labels[:, -1]

            outputs = lstm_model(data)
            if labels.dim() > 1:
                labels = labels[:, -1]
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            total_loss += loss.item()
            f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
            total_f1 += f1
            total_samples += 1

            for label, pred in zip(labels, predicted):
                val_total_per_class[label] += 1
                if label == pred:
                    val_correct_per_class[label] += 1

    val_loss = total_loss / len(val_dataloader)
    val_f1_score = total_f1 / total_samples
    val_losses.append(val_loss)
    val_f1_scores.append(val_f1_score)

    for i in range(num_classes):
        if val_total_per_class[i] > 0:
            val_bin_accuracies[i].append(val_correct_per_class[i] / val_total_per_class[i])

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train F1 Score: {train_f1_score:.4f}, Val Loss: {val_loss:.4f}, Val F1 Score: {val_f1_score:.4f}")

# Plotting training and validation loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plotting training and validation F1 score
plt.subplot(1, 2, 2)
plt.plot(train_f1_scores, label='Train F1 Score')
plt.plot(val_f1_scores, label='Validation F1 Score')
plt.title('Training and Validation F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.show()

# Plotting bin accuracies for training
plt.figure(figsize=(10, 5))
for i in range(num_classes):
    plt.plot(range(1, len(train_bin_accuracies[i]) + 1), train_bin_accuracies[i], label=f'Train Bin {i} Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Per Bin Over Epochs')
plt.legend()
plt.show()

# Plotting bin accuracies for validation
plt.figure(figsize=(10, 5))
for i in range(num_classes):
    plt.plot(range(1, len(val_bin_accuracies[i]) + 1), val_bin_accuracies[i], label=f'Val Bin {i} Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy Per Bin Over Epochs')
plt.legend()
plt.show()

# Save the LSTM model
torch.save(lstm_model.state_dict(), 'lstm_model.pth')