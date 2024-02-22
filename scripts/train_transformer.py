import torch
from sklearn.model_selection import train_test_split
from torch import optim, nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import csv

from oai_agents.common.arguments_transformer import TransformerConfig, sweep_config
from eye_tracking_dataset_operations.memmap_create_and_retrieve import return_memmaps, setup_and_process_xdf_files
from eye_tracking_dataset_operations import models
from eye_tracking_dataset_operations.eye_gaze_and_play_dataset import EyeGazeAndPlayDataset
import wandb

sweep_id = wandb.sweep(sweep=sweep_config, project="HAHA_eyetracking")

# Memmap file paths
participant_memmap_file = "path/to/the/participant_memmap"  # "path/to/memmap/participant_memmap.dat"
obs_heatmap_memmap_file = "path/to/the/obs_heatmap_memmap"  # "path/to/memmap/obs_heatmap_memmap.dat"
subtask_memmap_file = "path/to/the/subtask_memmap"  # "path/to/memmap/obs_heatmap_memmap.dat"
gaze_obj_memmap_file = "path/to/the/gaze_obj_file"  # "path/to/memmap/gaze_obj_file.dat"

# only needed initially to make the memmaps, please comment out after the memmaps are created.
# setup_and_process_xdf_files("data/eye_tracking_data/", participant_memmap_file, obs_heatmap_memmap_file, subtask_memmap_file, gaze_obj_memmap_file)
# exit(0)

# fill_participant_questions_from_csv(participant_memmap_file, 'path/to/the/GameData_EachTrial.csv')


def sweep_train():
    participant_memmap, obs_heatmap_memmap, subtask_memmap, gaze_obj_memmap = return_memmaps(participant_memmap_file, obs_heatmap_memmap_file, subtask_memmap_file, gaze_obj_memmap_file)

    # Encoding options are Game Data 'gd', Eye Gaze 'eg', both 'gd+eg', Gaze Object 'go', or Collapsed Eye Gaze 'ceg'
    # Note that 'go and 'ceg' are baselines that aggregate over data over the time period, so should probably be
    # inputted to a Linear classifier not a transformer
    encoding_type = 'gd+eg'
    # Label options are 'score', 'subtask', 'q1', 'q2', 'q3', 'q4', or 'q5
    label_type = 'score'
    dataset = EyeGazeAndPlayDataset(participant_memmap, obs_heatmap_memmap, subtask_memmap, gaze_obj_memmap,
                                            encoding_type, label_type)

    # TODO ASAP save encoding type / label type with results / csv and in wandb logging to know what the results were for
    exp_name = f'{encoding_type}_{label_type}'
    # Configuration for hyperparameters
    wandb.init(mode='online')


    model = models.SimpleTransformer(
        d_model=TransformerConfig.d_model,
        nhead=TransformerConfig.n_head,
        num_layers=TransformerConfig.num_layers,
        dim_feedforward=TransformerConfig.dim_feedforward,
        num_classes=dataset.num_classes,
        max_len=dataset.num_timesteps,
        input_dim=dataset.input_dim
    )

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate, weight_decay=0)
    scheduler = models.WarmupScheduler(optimizer, TransformerConfig.warmup_steps, TransformerConfig.base_lr,
                                       TransformerConfig.max_lr)
    scheduler_decay = StepLR(optimizer, step_size=wandb.config.decay_step_size, gamma=wandb.config.decay_factor)

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
    for epoch in range(400):
        # Training phase
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        model.train()
        dataset.set_split('train')
        train_dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
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
            loss = criterion(torch.reshape(outputs, (batch_size * seq_len, dataset.num_classes)),
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
            if label_type != 'score':
                # Only calculate accuracy on last step
                predicted = predicted[:, -1]
                labels = labels[:, -1]

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
        if epoch < TransformerConfig.warmup_steps:
            scheduler.step()
        else:
            scheduler_decay.step()

        # Validation phase
        model.eval()
        dataset.set_split('val')
        val_dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for data, labels in val_dataloader:
                outputs = model(data)

                # Calculate loss
                batch_size, seq_len, _ = data.shape
                loss = criterion(torch.reshape(outputs, (batch_size * seq_len, TransformerConfig.num_classes)),
                                 torch.reshape(labels, (batch_size * seq_len,)))

                # Calculate accuracy
                _, predicted = torch.max(outputs, dim=2)
                if label_type != 'score':
                    # Only calculate accuracy on last step
                    predicted = predicted[:, -1]
                    labels = labels[:, -1]
                predicted = predicted.view(-1)
                labels = labels.view(-1)
                acc = calculate_accuracy(predicted, labels)
                val_loss += loss.item()
                val_acc += acc
                # TODO ASAP save best model based on val_acc

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
            f"Epoch {epoch + 1}/{TransformerConfig.num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        wandb.log({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss, 'train_acc': train_acc,'val_acc': val_acc})

    # TODO ASAP load best model and evaluate on test set and include in csv
    # Make sure to index last timestep for accuracy calculation if label_type does not equal 'score'

    with open('training_metrics.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['Epoch', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy', 'Learning Rate'])
        for i in range(TransformerConfig.num_epochs):
            writer.writerow([metrics['epoch'][i], metrics['train_loss'][i], metrics['train_acc'][i],
                             metrics['val_loss'][i], metrics['val_acc'][i], metrics['learning_rate'][i]])

    # plt.figure()
    # plt.plot(average_gradient_norms)
    # plt.title('Average Gradient Norms Over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Average Gradient Norm')
    # plt.show()
    #
    # plt.figure(figsize=(12, 5))
    # plt.plot(train_losses, label='Train Loss')
    # plt.plot(val_losses, label='Val Loss')
    # plt.title('Loss over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()
    #
    # # Plot for learning rate
    # plt.figure(figsize=(6, 4))
    # plt.plot(learning_rates)
    # plt.xlabel('Epoch')
    # plt.ylabel('Learning Rate')
    # plt.title('Learning Rate Over Epochs')
    # plt.show()
    #
    # plt.figure(figsize=(12, 5))
    # plt.plot(train_accs, label='Train Accuracy')
    # plt.plot(val_accs, label='Validation Accuracy')
    # plt.title('Accuracy over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()
    #
    # Save the model if needed
    # TODO ASAP give this model a better name based on encoding type and label type
    torch.save(model.state_dict(), 'model.pth')
    # wandb.save('model.pth')
    # wandb.log_artifact('model.pth', type='model')

# sweep_train()
wandb.agent(sweep_id, function=sweep_train, count=5)

