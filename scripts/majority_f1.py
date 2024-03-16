import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from eye_tracking_dataset_operations.eye_gaze_and_play_dataset import EyeGazeAndPlayDataset
from eye_tracking_dataset_operations.memmap_create_and_retrieve import return_memmaps

# Define the paths to your memmap files
participant_memmap_file = "/home/stephane/HAHA/eye_data/participant_memmap.dat"  # "path/to/memmap/participant_memmap.dat"
obs_heatmap_memmap_file = "/home/stephane/HAHA/eye_data/obs_heatmap_memmap.dat"  # "path/to/memmap/obs_heatmap_memmap.dat"
subtask_memmap_file = "/home/stephane/HAHA/eye_data/subtask_memmap.dat"  # "path/to/memmap/obs_heatmap_memmap.dat"
gaze_obj_memmap_file = "/home/stephane/HAHA/eye_data/gaze_obj_memmap.dat"  # "path/to/memmap/gaze_obj_file.dat"


# Specify all encoding, label types, layouts, and agents to iterate over
encoding_type = 'gd+eg'  # Example: 'eg' for Eye Gaze
label_types = ['score', 'subtask', 'q3']
layouts = ['asymmetric_advantages', 'coordination_ring', 'counter_circuit_o_1order']
agents = ['haha', 'random_agent', 'selfplay'] 

participant_memmap, obs_heatmap_memmap, subtask_memmap, gaze_obj_memmap = return_memmaps(
    participant_memmap_file, obs_heatmap_memmap_file, subtask_memmap_file, gaze_obj_memmap_file)

# Iterate over label types, layouts, and agents
for label_type in label_types:
    # Define CSV file path for the current label type
    csv_file = f'{label_type}_majority_class_f1_per_timestep.csv'
    # Initialize the dataframe
    df = pd.DataFrame()

    for layout in layouts:
        for agent in agents:
            # Create dataset instance for the test set with the specific layout and agent
            dataset = EyeGazeAndPlayDataset(
                participant_memmap, obs_heatmap_memmap, subtask_memmap, gaze_obj_memmap,
                encoding_type, label_type, num_timesteps=20,  # Adjust num_timesteps as needed
                layout_to_use=layout, agent_to_use=agent
            )
            dataset.set_split('test')

            # Prepare DataLoader
            test_dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

            # Collect labels for all timesteps and calculate the majority class F1 score per timestep
            all_labels = [np.array([]) for _ in range(dataset.num_timesteps)]

            for _, labels in test_dataloader:
                labels = labels.numpy()  # labels shape: [batch_size, num_timesteps]
                for timestep in range(dataset.num_timesteps):
                    # Flatten labels for each timestep before concatenation
                    timestep_labels = labels[:, timestep].flatten()
                    all_labels[timestep] = np.concatenate((all_labels[timestep], timestep_labels))

            # Calculate and store the F1 score for each timestep
            for timestep, labels in enumerate(all_labels):
                majority_class = np.argmax(np.bincount(labels.astype(int)))
                predicted_labels = np.full_like(labels, majority_class)
                f1 = f1_score(labels, predicted_labels, average='weighted')
                new_row = pd.DataFrame({
                    'encoding_type': encoding_type,
                    'label_type': label_type,
                    'layout': layout,
                    'agent': agent,
                    'timestep': timestep,
                    'majority_class_f1': f1,
                    'majority_class': majority_class
                }, index=[0])
                df = pd.concat([df, new_row], ignore_index=True)

    # Save the results to the CSV file after processing all layouts and agents for a label type
    df.to_csv(csv_file, index=False)