### This file does not have the latest code and is deprecated as of now.


from oai_agents.common import overcooked_dataset_et_eyetracking
from scripts import models
from oai_agents.common import arguments
from oai_agents.common.state_encodings import ENCODING_SCHEMES
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report, confusion_matrix


def compute_bin_accuracies(conf_matrix):
    # Extracting diagonal (true positives for each class)
    true_positives = conf_matrix.diagonal()
    # Summing each column (total samples per class)
    total_per_class = conf_matrix.sum(axis=1)
    return true_positives / total_per_class


seq_len = 1  # This is the sequence length of the input to the model
obs_dim = 96  # This is the agent_obs size
action_dim = 6  # This is the one hot encoded action feature
hidden_dim = 64
output_dim = 4
num_layers = 2
num_epochs = 10
learning_rate = 0.0001
f1_results = []

args = arguments.get_arguments()
encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
OD = overcooked_dataset_et_eyetracking.OvercookedDataset(encoding_fn, args.layout_names, args, seq_len=seq_len,
                                                         num_classes=4)

distinct_trial_ids = list(set(OD.main_trials['trial_id']))
distinct_trial_ids.sort()  # Ensure they are in order

# Split trial IDs at the 85% mark
split_index = int(0.85 * len(distinct_trial_ids))
train_trial_ids = distinct_trial_ids[:split_index]
test_trial_ids = distinct_trial_ids[split_index:]

# Split the dataset based on these trial IDs
train_indices = [index for index, trial_id in enumerate(OD.main_trials['trial_id']) if trial_id in train_trial_ids]
test_indices = [index for index, trial_id in enumerate(OD.main_trials['trial_id']) if trial_id in test_trial_ids]

train_data = [OD[idx] for idx in train_indices]
test_data = [OD[idx] for idx in test_indices]

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

model = models.LSTMFeatureBased(obs_dim, action_dim, hidden_dim, output_dim, num_layers, 0.5)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training loop
train_losses = []
val_f1_scores = []
bin_accuracies_list = []
best_f1 = 0
patience = 3
no_improve_epochs = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        agent_obs = batch['agent_obs'].float()
        actions = batch['action'].float()
        score_bin = batch['score_bins'].long()

        optimizer.zero_grad()
        outputs = model(agent_obs, actions)  # Passing observations and actions separately

        # Compute the mean prediction across the sequence length for each batch
        outputs_mean = outputs.mean(dim=1)

        loss = criterion(outputs_mean, score_bin)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    all_preds = []
    all_true = []
    for batch in test_loader:
        agent_obs = batch['agent_obs'].float()
        actions = batch['action'].float()
        score_bin = batch['score_bins'].long()

        with torch.no_grad():
            outputs = model(agent_obs, actions)
            outputs_mean = outputs.mean(dim=1)
            _, preds = torch.max(outputs_mean, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(score_bin.cpu().numpy())

    f1 = f1_score(all_true, all_preds, average='macro')
    if f1 > best_f1:
        best_f1 = f1
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("Stopping training due to decreasing F1 score.")
        break
    val_f1_scores.append(f1)
    f1_results.append(best_f1)
    conf_matrix = confusion_matrix(all_true, all_preds)
    bin_accuracies = compute_bin_accuracies(conf_matrix)
    bin_accuracies_list.append(bin_accuracies)

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Val F1 Score: {f1}")
    print(f"Bin Accuracies: {bin_accuracies}")

print(f"RESULTS FOR SEQUENCE LENGTH {seq_len}")
# Plot training loss and validation F1 score
plt.figure(figsize=(15, 5))

# Training Loss
plt.subplot(1, 3, 1)
plt.plot(train_losses)
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# F1 Scores
plt.subplot(1, 3, 2)
plt.plot(val_f1_scores)
plt.title('Validation F1 Score Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')

# Bin Accuracies
for bin_idx in range(output_dim):
    plt.subplot(1, 3, 3)
    plt.plot([x[bin_idx] for x in bin_accuracies_list], label=f'Bin {bin_idx + 1}')
    plt.title('Bin Accuracies Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

plt.tight_layout()
plt.show()

print(classification_report(all_true, all_preds))
