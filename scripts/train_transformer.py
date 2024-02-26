import torch
from sklearn.model_selection import train_test_split
from torch import optim, nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import csv
from sklearn.metrics import f1_score
import numpy as np

from eye_tracking_dataset_operations.preprocess_eyetracking import fill_participant_questions_from_csv
from oai_agents.common.arguments_transformer import TransformerConfig, sweep_config
from eye_tracking_dataset_operations.memmap_create_and_retrieve import return_memmaps, setup_and_process_xdf_files
from eye_tracking_dataset_operations import models
from eye_tracking_dataset_operations.eye_gaze_and_play_dataset import EyeGazeAndPlayDataset
import wandb

sweep_id = wandb.sweep(sweep=sweep_config, project="HAHA_eyetracking")

# Memmap file paths
participant_memmap_file = "/home/stephane/HAHA/eye_data/participant_memmap.dat"  # "path/to/memmap/participant_memmap.dat"
obs_heatmap_memmap_file = "/home/stephane/HAHA/eye_data/obs_heatmap_memmap.dat"  # "path/to/memmap/obs_heatmap_memmap.dat"
subtask_memmap_file = "/home/stephane/HAHA/eye_data/subtask_memmap.dat"  # "path/to/memmap/obs_heatmap_memmap.dat"
gaze_obj_memmap_file = "/home/stephane/HAHA/eye_data/gaze_obj_memmap.dat"  # "path/to/memmap/gaze_obj_file.dat"

# only needed initially to make the memmaps, please comment out after the memmaps are created.
# setup_and_process_xdf_files("/home/stephane/HAHA/eye_data/Data/xdf_files", participant_memmap_file, obs_heatmap_memmap_file, subtask_memmap_file, gaze_obj_memmap_file)
# exit(0)

#  fill_participant_questions_from_csv(participant_memmap_file, '/home/stephane/HAHA/eye_data/GameData_EachTrial.csv')


def sweep_train():
    participant_memmap, obs_heatmap_memmap, subtask_memmap, gaze_obj_memmap = return_memmaps(participant_memmap_file, obs_heatmap_memmap_file, subtask_memmap_file, gaze_obj_memmap_file)

    # Encoding options are Game Data 'gd', Eye Gaze 'eg', both 'gd+eg', Gaze Object 'go', or Collapsed Eye Gaze 'ceg'
    # Note that 'go and 'ceg' are baselines that aggregate over data over the time period, so should probably be
    # inputted to a Linear classifier not a transformer
    encoding_type = 'gd'
    # Label options are 'score', 'subtask', 'q1', 'q2', 'q3', 'q4', or 'q5
    label_type = 'score'
    

    # TODO ASAP save encoding type / label type with results / csv and in wandb logging to know what the results were for
    exp_name = f'{encoding_type}_{label_type}'
    # Configuration for hyperparameters
    with wandb.init(mode='online', group='EYE+GD') as run:

        dataset = EyeGazeAndPlayDataset(participant_memmap, obs_heatmap_memmap, subtask_memmap, gaze_obj_memmap,
                                            encoding_type, label_type, num_timesteps = wandb.config.num_timesteps_to_consider)

        run.name = f'{exp_name}_{wandb.config.learning_rate}_{wandb.config.batch_size}'

        wandb.config.update({
        "encoding_type": encoding_type,
        "label_type": label_type,
        })


        model = models.SimpleTransformer(
            d_model=TransformerConfig.d_model,
            nhead=TransformerConfig.n_head,
            num_layers=TransformerConfig.num_layers,
            dim_feedforward=TransformerConfig.dim_feedforward,
            num_classes=dataset.num_classes,
            max_len=dataset.num_timesteps,
            input_dim=dataset.input_dim
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate, weight_decay=0)
        scheduler = models.WarmupScheduler(optimizer, TransformerConfig.warmup_steps, TransformerConfig.base_lr,
                                        TransformerConfig.max_lr)
        scheduler_decay = StepLR(optimizer, step_size=wandb.config.decay_step_size, gamma=wandb.config.decay_factor)

        def calculate_accuracy(predicted, labels):
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            return correct / total

        def calculate_f1(predicted, labels, average='weighted'):
            true = labels.view(-1).cpu().numpy()
            preds = predicted.view(-1).cpu().numpy()
            return f1_score(true, preds, average=average)
        
        def check_labels_structure(labels):
            print("Type of labels:", type(labels))
            if isinstance(labels, (list, np.ndarray, torch.Tensor)):
                print("Length of labels:", len(labels))
                if len(labels) > 0:
                    print("Type of first element:", type(labels[0]))
                    if isinstance(labels[0], (list, np.ndarray, torch.Tensor)):
                        print("Length of first element:", len(labels[0]))
                        if len(labels[0]) > 0:
                            print("Type of first element of first element:", type(labels[0][0]))
            else:
                print("Labels is neither a list, numpy array, nor a PyTorch tensor.")

        learning_rates = []
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        average_gradient_norms = []
        metrics = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'train_f1':[],
            'val_loss': [],
            'val_acc': [],
            'val_f1':[],
            'learning_rate': [],
        }
        best_val_acc = 0.0
        
        for epoch in range(wandb.config.epochs):
            train_labels_list, train_preds_list = [], []
            val_labels_list, val_preds_list = [], []
            # Training phase
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            model.train()
            dataset.set_split('train')
            train_dataloader = DataLoader(dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers = 4)

            train_loss, train_acc = 0.0, 0.0
            for i, (data, labels) in enumerate(train_dataloader):
                # comment the next 2 lines if label_type is not subtask    
                # labels = labels[:, :, 0]
                # labels = labels.long()
                # tensors = [label[0] for label in labels]  # This will create a list of tensors

                # Stack the tensors to create a single tensor
                # labels = torch.stack(tensors)
                #data = torch.tensor(data, dtype=torch.float)
                #labels = torch.tensor(labels, dtype=torch.long)    
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()

                # Forward pass
                outputs = model(data)  # outputs shape: [batch_size, seq_len, num_classes]
                # print(f"Device of input data: {data.device}")
                # print(f"Device of model parameters: {next(model.parameters()).device}")
               

                if i == 0:
                    print(f"Epoch {epoch + 1}")
                    print(f"Input shape: {data.shape}")
                    print(f"Output shape: {outputs.shape}")
                    # print(f"Input example (first element): {data[0]}")
                    # print(f"Output example (first element): {outputs[0]}")

                # Obtain batch size and sequence length from data
                batch_size, seq_len, _ = data.shape
                # print(f"Min label: {labels.min().item()}, Max label: {labels.max().item()}")
                # print(f"num_classes:{dataset.num_classes}")


                # Calculate loss
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
                if label_type == 'score':
                    # Only calculate accuracy on last step
                    predicted = predicted[:, -1]
                    labels = labels[:, -1]

                predicted = predicted.view(-1)
                labels = labels.view(-1)
                acc = calculate_accuracy(predicted, labels)
                train_labels_list.append(labels)
                train_preds_list.append(predicted)
                train_loss += loss.item()
                train_acc += acc

            train_loss /= len(train_dataloader)
            train_acc /= len(train_dataloader)
            train_accs.append(train_acc)
            train_losses.append(train_loss)
            train_f1 = calculate_f1(torch.cat(train_preds_list), torch.cat(train_labels_list))
            

            # Learning rate scheduling
            if epoch < TransformerConfig.warmup_steps:
                scheduler.step()
            else:
                scheduler_decay.step()

            # Validation phase
            model.eval()
            dataset.set_split('val')
            val_dataloader = DataLoader(dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers = 4)
            val_loss, val_acc = 0.0, 0.0
            with torch.no_grad():
                for data, labels in val_dataloader:
                    # comment the next 2 lines if label_type is not subtask 
                    # labels = labels[:, :, 0] 
                    # labels = labels.long()
                    data, labels = data.to(device), labels.to(device)
                    outputs = model(data)

                    # Calculate loss
                    batch_size, seq_len, _ = data.shape
                    loss = criterion(torch.reshape(outputs, (batch_size * seq_len, dataset.num_classes)),
                                    torch.reshape(labels, (batch_size * seq_len,)))

                    # Calculate accuracy
                    _, predicted = torch.max(outputs, dim=2)
                    if label_type == 'score':
                        # Only calculate accuracy on last step
                        predicted = predicted[:, -1]
                        labels = labels[:, -1]
                    predicted = predicted.view(-1)
                    labels = labels.view(-1)
                    val_labels_list.append(labels)
                    val_preds_list.append(predicted)
                    acc = calculate_accuracy(predicted, labels)
                    val_loss += loss.item()
                    val_acc += acc
                    # TODO ASAP save best model based on val_acc
                    best_model_path = f'best_model_{exp_name}_{wandb.config.learning_rate}_{wandb.config.batch_size}.pth'
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        torch.save(model.state_dict(), best_model_path)
                        print(f"New best model saved with val_acc: {val_acc}")

                val_loss /= len(val_dataloader)
                val_acc /= len(val_dataloader)
                val_accs.append(val_acc)
                val_losses.append(val_loss)
                val_f1 = calculate_f1(torch.cat(val_preds_list), torch.cat(val_labels_list))
                

            metrics['epoch'].append(epoch + 1)
            metrics['train_loss'].append(train_loss)
            metrics['train_acc'].append(train_acc)
            metrics['train_f1'].append(train_f1)
            metrics['val_loss'].append(val_loss)
            metrics['val_acc'].append(val_acc)
            metrics['learning_rate'].append(current_lr)
            metrics['val_f1'].append(val_f1)

            # Print epoch results
            print(
                f"Epoch {epoch + 1}/{TransformerConfig.num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            wandb.log({'epoch': epoch + 1, 'train_loss': train_loss, 'train_f1': train_f1, 'val_loss': val_loss, 'train_acc': train_acc,'val_acc': val_acc, 'val_f1': val_f1, 'lr': current_lr})

        # TODO ASAP load best model and evaluate on test set and include in csv
        # Make sure to index last timestep for accuracy calculation if label_type does not equal 'score'
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        dataset.set_split('test')
        test_dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers = 4)
        test_acc = 0.0
        test_labels_list, test_preds_list = [], []
        with torch.no_grad():
            for data, labels in test_dataloader:
                # comment the next 2 lines if label_type is not subtask 
                # labels = labels[:, :, 0] 
                # labels = labels.long()
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                # Calculate accuracy
                # Adjust the accuracy calculation based on your needs
                _, predicted = torch.max(outputs, dim=2)
                if label_type == 'score':
                    # Only calculate accuracy on last step
                    predicted = predicted[:, -1]
                    labels = labels[:, -1]
                predicted = predicted.view(-1)
                labels = labels.view(-1)
                test_labels_list.append(labels)
                test_preds_list.append(predicted)
                acc = calculate_accuracy(predicted, labels)
                test_acc += acc
        test_acc /= len(test_dataloader)
        print(f"Test Accuracy: {test_acc}")

        test_f1 = calculate_f1(torch.cat(test_preds_list), torch.cat(test_labels_list))
        print(f"Test F1 Score: {test_f1}")
        

        with open('test_metrics.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Encoding Type', 'Label Type', 'Test Accuracy', 'F1_score'])
            writer.writerow([encoding_type, label_type, test_acc, test_f1])

        wandb.log({'test_accuracy': test_acc, 'test_f1': test_f1 })

        with open('training_metrics.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy','Train F1', 'Validation Loss', 'Validation Accuracy','Validation F1', 'Learning Rate'])
            for i in range(400):
                writer.writerow([metrics['epoch'][i], metrics['train_loss'][i], metrics['train_acc'][i], metrics['train_f1'][i],
                                metrics['val_loss'][i], metrics['val_acc'][i],metrics['val_f1'][i], metrics['learning_rate'][i]])

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
        descriptive_model_path = f'model_{encoding_type}_{label_type}.pth'
        torch.save(model.state_dict(), descriptive_model_path)
        print(f"Model saved as {descriptive_model_path}")
        wandb.save(descriptive_model_path)
        # wandb.save('model.pth')
        # wandb.log_artifact('model.pth', type='model')

# sweep_train()
wandb.agent(sweep_id, function=sweep_train, count=1)

