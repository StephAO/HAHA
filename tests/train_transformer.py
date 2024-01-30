import torch
from sklearn.model_selection import train_test_split
from torch import optim, nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import csv

from oai_agents.common.memmap_create_and_retrieve import return_memmaps
from tests import Models
from scripts.transformer_helper import process_trial_data, process_data

# Memmap file paths
participant_memmap_file = "/data/obs_heatmap_memmap.dat"  # "path/to/memmap/participant_memmap.dat"
obs_heatmap_memmap_file = "/data/participant_memmap.dat"  # "path/to/memmap/obs_heatmap_memmap.dat"

num_timesteps_to_consider = 20

# Model Initialization
d_model = 512
nhead = 8
num_layers = 8
dim_feedforward = 2048
num_classes = 4
input_dim = 1260  # Based on input dimension

# warmup
warmup_steps = 300
base_lr = 1e-6
max_lr = 1e-5

# Training Loop
num_epochs = 10

# only needed initially to make the memmaps, please comment out after the memmaps are created.
# setup_and_process_xdf_files("/path/t0/folder/with/xdfFiles", participant_memmap_file,obs_heatmap_memmap_file) #


participant_memmap, obs_heatmap_memmap = return_memmaps(participant_memmap_file, obs_heatmap_memmap_file)

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


train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

model = Models.SimpleTransformer(d_model, nhead, num_layers, dim_feedforward, num_classes, input_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-6, weight_decay=0)

scheduler = Models.WarmupScheduler(optimizer, warmup_steps, base_lr, max_lr)

decay_step_size = 50
decay_factor = 0.5
scheduler_decay = StepLR(optimizer, step_size=decay_step_size, gamma=decay_factor)



def calculate_accuracy(predicted, labels):
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total


learning_rates = []
train_losses, val_losses = [], []
train_accs, val_accs = [], []
average_gradient_norms = []
metrics = {
    'epoch': [],
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'learning_rate': []
}
for epoch in range(num_epochs):
    # Training phase
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for i, (data, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(data)  # outputs shape: [batch_size, seq_len, num_classes]
        if i == 0:
            print(f"Epoch {epoch + 1}")
            print(f"Input shape: {data.shape}")
            print(f"Output shape: {outputs.shape}")
            # print(f"Input example (first element): {data[0]}")
            # print(f"Output example (first element): {outputs[0]}")

        # Obtain batch size and sequence length from data
        batch_size, seq_len, _ = data.shape

        # Calculate loss (reshape as per your requirement)
        loss = criterion(torch.reshape(outputs, (batch_size * seq_len, num_classes)),
                         torch.reshape(labels, (batch_size * seq_len,)))
        loss.backward()  # Backward pass

        total_norm = 0
        num_params = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                total_norm += param.grad.norm().item()
                num_params += 1
        average_norm = total_norm / num_params
        average_gradient_norms.append(average_norm)
        optimizer.step()  # Update weights

        # Calculate accuracy
        _, predicted = torch.max(outputs, dim=2)
        predicted = predicted.view(-1)
        labels = labels.view(-1)
        acc = calculate_accuracy(predicted, labels)
        train_loss += loss.item()
        train_acc += acc

    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    train_accs.append(train_acc)
    train_losses.append(train_loss)

    # Learning rate scheduling
    if epoch < warmup_steps:
        scheduler.step()
    else:
        scheduler_decay.step()

    # Validation phase
    model.eval()
    val_loss, val_acc = 0.0, 0.0
    with torch.no_grad():
        for data, labels in val_dataloader:
            outputs = model(data)

            # Calculate loss
            batch_size, seq_len, _ = data.shape
            loss = criterion(torch.reshape(outputs, (batch_size * seq_len, num_classes)),
                             torch.reshape(labels, (batch_size * seq_len,)))

            # Calculate accuracy
            _, predicted = torch.max(outputs, dim=2)
            predicted = predicted.view(-1)
            labels = labels.view(-1)
            acc = calculate_accuracy(predicted, labels)
            val_loss += loss.item()
            val_acc += acc

        val_loss /= len(val_dataloader)
        val_acc /= len(val_dataloader)
        val_accs.append(val_acc)
        val_losses.append(val_loss)

    metrics['epoch'].append(epoch + 1)
    metrics['train_loss'].append(train_loss)
    metrics['train_acc'].append(train_acc)
    metrics['val_loss'].append(val_loss)
    metrics['val_acc'].append(val_acc)
    metrics['learning_rate'].append(current_lr)

    # Print epoch results
    print(
        f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

with open('../scripts/training_metrics.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ['Epoch', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy', 'Learning Rate'])
    for i in range(num_epochs):
        writer.writerow([metrics['epoch'][i], metrics['train_loss'][i], metrics['train_acc'][i],
                         metrics['val_loss'][i], metrics['val_acc'][i], metrics['learning_rate'][i]])

plt.figure()
plt.plot(average_gradient_norms)
plt.title('Average Gradient Norms Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Average Gradient Norm')
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot for learning rate
plt.figure(figsize=(6, 4))
plt.plot(learning_rates)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Over Epochs')
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save the model if needed
torch.save(model.state_dict(), '../scripts/model.pth')
