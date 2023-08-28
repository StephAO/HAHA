from torch.utils.data import Dataset, DataLoader
import sys
from sklearn.metrics import f1_score
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

sys.path.append('/path_to_directory_containing_HAHA/')

from oai_agents.common.overcooked_dataset_Baseline.py import main as initialize_overcooked_dataset  # Import the main function

OD = initialize_overcooked_dataset()
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
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)
val_loader = DataLoader(test_data, batch_size=64, shuffle=False, drop_last=True)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

input_size = 97
hidden_size = 64
output_size = 4
num_layers = 2
num_epochs = 10
learning_rate = 0.00001


model = LSTMClassifier(input_size, hidden_size, output_size, num_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training loop
train_losses = []
val_f1_scores = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        agent_obs = batch['agent_obs'].float()
        actions = batch['action'].float()
        score_bin = batch['score_bins'].long()
        
        optimizer.zero_grad()
        inputs = torch.cat((agent_obs, actions.unsqueeze(-1)), dim=-1)
        #print(f"Input shape: {inputs.shape}")
        outputs = model(inputs)
        # print(outputs.shape)  # Should be [batch_size, number_of_classes]
        # print(score_bin.shape) 
        loss = criterion(outputs, score_bin)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    all_preds = []
    all_true = []
    for batch in val_loader:
        agent_obs = batch['agent_obs'].float()
        actions = batch['action'].float()
        score_bin = batch['score_bins'].long()

        with torch.no_grad():
            outputs = model(torch.cat((agent_obs, actions.unsqueeze(-1)), dim=-1))
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(score_bin.cpu().numpy())

    f1 = f1_score(all_true, all_preds, average='macro')
    val_f1_scores.append(f1)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val F1 Score: {f1}")

# Plot training loss and validation F1 score
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(val_f1_scores)
plt.title('Validation F1 Score Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')

plt.show()
