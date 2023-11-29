import torch
from sklearn.model_selection import train_test_split
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset

from scripts.memmap_create_and_retrieve import return_memmaps, setup_and_process_xdf_files
from tests.transformer import SimpleTransformer, WarmupScheduler
from tests.transformer_helper import process_trial_data, process_data


# Memmap file paths
participant_memmap_file = "path/to/memmap/participant_memmap.dat"  # "/HAHA-eyetracking/data/participant_memmap.dat"
obs_heatmap_memmap_file = "path/to/memmap/obs_heatmap_memmap.dat"  # "/HAHA-eyetracking/data/obs_heatmap_memmap.dat"

# num_participants = 18  # Total number of participants
# num_trials_per_participant = 18  # Trials per participant
# obs_channels = 27  # Number of binary masks in the observation data
# grid_shape = (9, 5)  # Padded grid size


num_timesteps_to_consider = 150

# Model Initialization
d_model = 256
nhead = 8
num_layers = 3
dim_feedforward = 512
num_classes = 3  # Assuming 3 classes for classification
input_dim = 1260  # Based on input dimension

# warmup

warmup_steps = 200  # Define the number of steps for warmup
base_lr = 1e-6  # Starting learning rate
max_lr = 1e-4  # Target learning rate (same as the optimizer's initial lr)

# Training Loop
num_epochs = 10

# only needed initially to make the memmaps, please comment out after the memmaps are created.

#setup_and_process_xdf_files("/path/t0/folder/with/xdfFiles", participant_memmap_file,obs_heatmap_memmap_file)

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


# Create dataset instances
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

model = SimpleTransformer(d_model, nhead, num_layers, dim_feedforward, num_classes, input_dim)

# Criterion and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


scheduler = WarmupScheduler(optimizer, warmup_steps, base_lr, max_lr)


# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total




for epoch in range(num_epochs):
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for data, labels in train_dataloader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(data)  # outputs shape: [batch_size, num_classes]

        if labels.dim() == 2 and labels.size(1) > 1:
            labels = labels[:, -1]  # Select the last label of each sequence

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Calculate accuracy
        acc = calculate_accuracy(outputs, labels)
        train_loss += loss.item()
        train_acc += acc

    # Compute average training loss and accuracy
    #     train_loss /= len(train_dataloader)
    #     train_acc /= len(train_dataloader)

    # Validation phase
    model.eval()
    val_loss, val_acc = 0.0, 0.0
    with torch.no_grad():
        for data, labels in val_dataloader:
            outputs = model(data)

            if labels.dim() == 2 and labels.size(1) > 1:
                labels = labels[:, -1]  # Select the last label of each sequence

            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)
            val_loss += loss.item()
            val_acc += acc

    # Compute average validation loss and accuracy
    #     val_loss /= len(val_dataloader)
    #     val_acc /= len(val_dataloader)

    print(
        f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Save the model if needed
torch.save(model.state_dict(), 'model.pth')
