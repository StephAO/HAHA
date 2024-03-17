import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import csv
import os
from eye_tracking_dataset_operations.memmap_create_and_retrieve import return_memmaps
from eye_tracking_dataset_operations.eye_gaze_and_play_dataset import EyeGazeAndPlayDataset
from collections import defaultdict
from sklearn.metrics import f1_score

def write_results_to_csv(results, headers, filepath):
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for row in results:
            writer.writerow(row)

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return self.fc3(x)



# def train_mlp(model, dataset, device, encoding_type, label_type):
#     model.train()
#     dataset.set_split('train')
#     dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    
#     training_results = []
#     for epoch in range(4000):  # Adjust epochs as needed
#         total_loss = 0
#         correct_predictions = 0
#         total_predictions = 0
        
#         for batch_idx, (data, labels) in enumerate(dataloader):
#             data, labels = data.to(device), labels.to(device).long()
        
#             # Data is now [batch_size, input_dim], Labels is now [batch_size]
#             print(f'Batch {batch_idx}: Data shape {data.shape}, Labels shape {labels.shape}')
            
#             optimizer.zero_grad()
#             outputs = model(data)

#             if torch.isnan(outputs).any():
#                 print(f'NaN detected in model outputs during training at epoch {epoch+1}, batch {batch_idx}')
#                 break
            
#             # Outputs is [batch_size, num_classes], consistent with Labels [batch_size]
#             print(f'Train Batch {batch_idx}: Data shape {data.shape}, Labels shape {labels.shape}, Outputs shape {outputs.shape}')
            
#             loss = criterion(outputs, labels.view(-1))
#             if torch.isnan(loss):
#                 print(f'NaN detected in loss during training at epoch {epoch+1}, batch {batch_idx}')
#                 break
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
            
#             correct_predictions += (predicted == labels.view(-1)).sum().item()
#             total_predictions += labels.numel()
        
        
#         avg_loss = total_loss / len(dataloader)
#         accuracy = correct_predictions / total_predictions
#         training_results.append((epoch + 1, avg_loss, accuracy))
    
#     training_filepath = f'mlp_training_results_{encoding_type}_{label_type}.csv'
#     write_results_to_csv(training_results, ['Epoch', 'Training Loss', 'Training Accuracy'], training_filepath)

def train_mlp(model, dataset, device, encoding_type, label_type, layout, agent):
    model.train()
    dataset.set_split('train')
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    training_results = []
    for epoch in range(4000): 
        # print(f"Starting training epoch {epoch+1}")
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        all_labels = []
        all_predictions = []
        
        for batch_idx, (data, labels) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device).long()
        
            optimizer.zero_grad()
            outputs = model(data)
            
            loss = criterion(outputs, labels.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            correct_predictions += (predicted == labels.view(-1)).sum().item()
            total_predictions += labels.numel()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        avg_f1 = f1_score(all_labels, all_predictions, average='weighted')
        #print(f"\tEpoch {epoch+1} completed: Avg Loss = {avg_loss}, Avg Accuracy = {accuracy}, Avg F1 Score = {avg_f1}")
        training_results.append((agent, layout, epoch + 1, avg_loss, accuracy, avg_f1))


        # Validation
        all_predictions = []
        all_labels = []
        print(f"Starting testing for layout {layout} and encoding type {encoding_type}")
        with torch.no_grad():
            for data_idx, (data, labels) in enumerate(dataloader):
                data, labels = data.to(device), labels.to(device).long()

                outputs = model(data)

                loss = criterion(outputs, labels.view(-1))
                batch_losses.append(loss.item())

                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                correct = (predicted == labels.view(-1)).sum().item()
                accuracy = correct / labels.numel()
                batch_accuracies.append(accuracy)

        avg_loss = np.mean(batch_losses)
        avg_accuracy = np.mean(batch_accuracies)
        avg_f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    training_filepath = f'mlp_training_results_{encoding_type}_{label_type}.csv'
    with open(training_filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        if os.stat(training_filepath).st_size == 0:
            writer.writerow(['Agent', 'Layout', 'Epoch', 'Training Loss', 'Training Accuracy', 'Training F1 Score'])
        for row in training_results:
            writer.writerow(row)


# def test_mlp(model, dataset, device, encoding_type, label_type):
#     model.eval()
#     dataset.set_split('test')
#     dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
#     criterion = nn.CrossEntropyLoss()
    
#     # Initialize lists to store per-batch results for detailed analysis.
#     batch_losses = []
#     batch_accuracies = []
#     all_predictions = []
#     all_labels = []
   

#     with torch.no_grad():
#         for data_idx, (data, labels) in enumerate(dataloader):
#             data, labels = data.to(device), labels.to(device).long()

#             outputs = model(data)
#             print(f'Test Batch {data_idx}: Data shape {data.shape}, Labels shape {labels.shape}, Outputs shape {outputs.shape}')

#             if torch.isnan(outputs).any():
#                 print(f'NaN detected in model outputs during testing at batch {data_idx}')
#                 continue  # Skip this batch if NaNs are detected

#             loss = criterion(outputs, labels.view(-1))
#             if torch.isnan(loss):
#                 print(f'NaN detected in loss during testing at batch {data_idx}')
#                 continue  # Skip this batch if NaNs are detected

#             batch_losses.append(loss.item())

#             _, predicted = torch.max(outputs, 1)
#             all_labels.extend(labels.view(-1).cpu().numpy())
#             all_predictions.extend(predicted.cpu().numpy())
#             correct = (predicted == labels.view(-1)).sum().item()
#             accuracy = correct / labels.numel()
#             batch_accuracies.append(accuracy)


#             # Append current batch predictions and labels to the list
#             all_predictions.extend(predicted.view(-1).cpu().numpy())
#             all_labels.extend(labels.view(-1).cpu().numpy())
#     print("Unique labels:", np.unique(all_labels))
#     print("Unique predictions:", np.unique(all_predictions))

#     # Calculate and print average loss and accuracy over all batches.
#     avg_loss = np.mean(batch_losses) if batch_losses else float('nan')
#     avg_accuracy = np.mean(batch_accuracies) if batch_accuracies else float('nan')
#     print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_accuracy:.4f}")

#     # Write detailed batch results to CSV
#     testing_filepath = f'mlp_testing_detailed_results_{encoding_type}_{label_type}.csv'
#     with open(testing_filepath, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['Batch', 'Loss', 'Accuracy'])
#         for i, (loss, accuracy) in enumerate(zip(batch_losses, batch_accuracies)):
#             writer.writerow([i, loss, accuracy])

#     # You could also return detailed results for further analysis if necessary
#     return avg_loss, avg_accuracy, all_predictions, all_labels





def test_mlp(model, dataset, device, encoding_type, label_type, layout, agent):
    model.eval()
    dataset.set_split('test')
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    
    batch_losses = []
    batch_accuracies = []
    all_predictions = []
    all_labels = []
    print(f"Starting testing for layout {layout} and encoding type {encoding_type}")
    with torch.no_grad():
        for data_idx, (data, labels) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device).long()

            outputs = model(data)

            loss = criterion(outputs, labels.view(-1))
            batch_losses.append(loss.item())

            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct = (predicted == labels.view(-1)).sum().item()
            accuracy = correct / labels.numel()
            batch_accuracies.append(accuracy)

    avg_loss = np.mean(batch_losses)
    avg_accuracy = np.mean(batch_accuracies)
    avg_f1 = f1_score(all_labels, all_predictions, average='weighted')

    testing_results = [(agent, layout, encoding_type, avg_loss, avg_accuracy, avg_f1)]
    testing_filepath = f'mlp_testing_detailed_results_{encoding_type}_{label_type}.csv'
    with open(testing_filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        if os.stat(testing_filepath).st_size == 0:
            writer.writerow(['Agent', 'Layout', 'Encoding Type', 'Test Loss', 'Test Accuracy', 'Test F1 Score'])
        for row in testing_results:
            writer.writerow(row)

    print(f'{layout} - {agent}: {avg_f1}')

    return avg_loss, avg_accuracy, all_predictions, all_labels, avg_f1

# Define your memmap file paths
participant_memmap_file = "/home/stephane/HAHA/eye_data/participant_memmap.dat"  # "path/to/memmap/participant_memmap.dat"
obs_heatmap_memmap_file = "/home/stephane/HAHA/eye_data/obs_heatmap_memmap.dat"  # "path/to/memmap/obs_heatmap_memmap.dat"
subtask_memmap_file = "/home/stephane/HAHA/eye_data/subtask_memmap.dat"  # "path/to/memmap/obs_heatmap_memmap.dat"
gaze_obj_memmap_file = "/home/stephane/HAHA/eye_data/gaze_obj_memmap.dat"  # "path/to/memmap/gaze_obj_file.dat"


participant_memmap, obs_heatmap_memmap, subtask_memmap, gaze_obj_memmap = return_memmaps(
    participant_memmap_file, obs_heatmap_memmap_file, subtask_memmap_file, gaze_obj_memmap_file)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


encoding_types = ['ceg']
layouts = ['asymmetric_advantages', 'coordination_ring', 'counter_circuit_o_1order']
label_type = 'score'
agents = ['haha', 'random_agent', 'selfplay']

for encoding_type in encoding_types:
    for layout in layouts:
        for agent in agents:
            print(f"Processing for encoding type: {encoding_type}, layout: {layout}, and agent: {agent}")
            dataset = EyeGazeAndPlayDataset(participant_memmap, obs_heatmap_memmap, subtask_memmap, gaze_obj_memmap,
                                            encoding_type, label_type, num_timesteps=10, layout_to_use=layout, agent_to_use=agent)
            
            # Ensure model initialization happens inside the loop to avoid reusing the same model weights across different runs
            model = SimpleMLP(input_dim=dataset.input_dim, hidden_dim=128, num_classes=dataset.num_classes).to(device)
            
            # Train and test the model; assume these functions handle setting the dataset splits internally
            train_mlp(model, dataset, device, encoding_type, label_type, layout, agent)
            test_mlp(model, dataset, device, encoding_type, label_type, layout, agent)



def check_label_distribution(dataset):
    label_counts = defaultdict(int)
    for _, label in dataset:
        label_counts[int(label.item())] += 1
    return dict(label_counts)

# Check distribution for each set
dataset.set_split('train')
train_label_distribution = check_label_distribution(dataset)
print("Training label distribution:", train_label_distribution)

dataset.set_split('test')
test_label_distribution = check_label_distribution(dataset)
print("Test label distribution:", test_label_distribution)
