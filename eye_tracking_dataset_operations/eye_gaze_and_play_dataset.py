from collections import defaultdict
import numpy as np
import torch
# from scripts.memmap_creation import participant_memmap, obs_heatmap_memmap
from eye_tracking_dataset_operations.preprocess_eyetracking import combine_and_standardize
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

class EyeGazeAndPlayDataset(Dataset):
    def __init__(self, participant_memmap, obs_heatmap_memmap, subtask_memmap, gaze_obj_memmap, encoding_type, label_type):
        self.encoding_type = encoding_type
        self.label_type = label_type

        participant_ids = list(processed_data.keys())

        train_participant_ids, val_participant_ids = train_test_split(participant_ids, test_size=0.2, random_state=42)
        print(f"train: {train_participant_ids}")
        print(f"test: {val_participant_ids}")
        X_train = [data for pid in train_participant_ids for data in processed_data[pid]['data']]
        y_train = [labels for pid in train_participant_ids for labels in processed_data[pid]['labels']]
        X_val = [data for pid in val_participant_ids for data in processed_data[pid]['data']]
        y_val = [labels for pid in val_participant_ids for labels in processed_data[pid]['labels']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def process_data():
    """
    Process the data from memory-mapped arrays.

    Parameters:
    - participant_memmap: Memory-mapped array with participant records.
    - obs_heatmap_memmap: Memory-mapped array with observation and heatmap data.
    - num_timesteps_to_consider: Number of timesteps to consider for processing.

    Returns:
    - trial_data: Dictionary with processed data, keyed by (participant_id, trial_id).
    - trial_labels: Dictionary with labels, keyed by (participant_id, trial_id).
    """

    input_data = {}
    labels = {}
    horizon = 400

    for record in participant_memmap:
        participant_id, trial_id, score, start_idx, end_idx, question_1, question_2, question_3, question_4, question_5 = record
        questions = [question_1, question_2, question_3, question_4, question_5]

        if start_idx - end_idx != horizon:
            print(start_idx, end_idx, '!!!!!!!!!!!1111')
        if encoding_type == 'gd':
            in_data = obs_heatmap_memmap[start_idx:end_idx, :-1]
        elif encoding_type == 'eg':
            in_data = obs_heatmap_memmap[start_idx:end_idx, -1:]
        elif encoding_type == 'gd+eg':
            in_data = obs_heatmap_memmap[start_idx:end_idx]
        elif encoding_type == 'ceg':
            # Collapse in data loader
            in_data = obs_heatmap_memmap[start_idx:end_idx, -1:]
        elif encoding_type == 'go':
            # Collapse in data loader
            in_data = gaze_obj_memmap[start_idx:end_idx]
        else:
            raise ValueError(f'{encoding_type} is not a valid encoding type')

        assert in_data.shape[0] == horizon, f'Expected {horizon} timesteps, got {in_data.shape[0]}'
        input_data[(participant_id, trial_id)] = in_data.reshape((in_data.shape[0], -1))

        if label_type == 'score':
            labels[(participant_id, trial_id)] = np.full((horizon, 1), score).tolist()
        elif label_type == 'subtask':
            labels[(participant_id, trial_id)] = subtask_memmap[start_idx:end_idx]
        elif label_type in ['q1', 'q2', 'q3', 'q4', 'q5']:
            q_idx = int(label_type[1]) - 1
            answer = questions[q_idx]
            labels[(participant_id, trial_id)] = np.full((horizon, 1), answer).tolist()
        else:
            raise ValueError(f'{label_type} is not a valid label type')

    return input_data, labels


# Example usage:
# trial_data, trial_labels = process_data(participant_memmap, obs_heatmap_memmap, num_timesteps_to_consider)

def process_trial_data(trial_data, trial_labels, num_bins=3):
    """
    Process trial data by binning the labels and organizing the data.

    Parameters:
    - trial_data: Dictionary of trial data with keys as (participant_id, trial_id).
    - trial_labels: Dictionary of trial labels with keys as (participant_id, trial_id).
    - num_bins: Number of bins to use for binning the labels.

    Returns:
    - processed_datas: Dictionary with processed data and labels, organized by participant_id.
    """

    # Calculate quantiles as bin edges
    all_scores = [score for scores in trial_labels.values() for score in scores]
    quantiles = np.percentile(all_scores, np.linspace(0, 100, num_bins + 1))
    quantiles[-1] = quantiles[-1] + 1  # To ensure the maximum score falls within the last bin

    # Initialize containers
    processed_datas = defaultdict(lambda: {'data': [], 'labels': []})
    bin_counts = defaultdict(int)

    # Process each trial
    for (participant_id, trial_id), trial_data_list in trial_data.items():
        trial_data_tensor = torch.tensor(trial_data_list, dtype=torch.float32)
        trial_labels_list = trial_labels[(participant_id, trial_id)]

        # Bin the scores using quantiles
        binned_labels = np.digitize(trial_labels_list, quantiles, right=False) - 1
        binned_labels = np.clip(binned_labels, 0, num_bins - 1)
        trial_labels_tensor = torch.tensor(binned_labels, dtype=torch.long)

        processed_datas[participant_id]['data'].append(trial_data_tensor)
        processed_datas[participant_id]['labels'].append(trial_labels_tensor)

        for label in binned_labels:
            bin_counts[label] += 1

    # Optionally, you can return bin_counts if needed
    return processed_datas  # , bin_counts
