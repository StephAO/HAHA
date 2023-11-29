from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
# from scripts.memmap_creation import participant_memmap, obs_heatmap_memmap
from scripts.preprocess_eyetracking import combine_and_standardize
from pathlib import Path


def create_memmaps(participant_memmap_file, obs_heatmap_memmap_file, num_participants = 18, num_trials_per_participant = 18, obs_channels = 27, grid_shape = (9, 5)):
    """
    Creates and returns memory-mapped arrays for participants and observation heatmaps.

    Parameters:
    - participant_memmap_file: File path for the participant memory-mapped file.
    - obs_heatmap_memmap_file: File path for the observation heatmap memory-mapped file.
    - num_participants: Total number of participants.
    - num_trials_per_participant: Number of trials per participant.
    - obs_channels: Number of binary masks in the observation data.
    - grid_shape: Shape of the padded grid.

    Returns:
    - participant_memmap: Memory-mapped array for participants.
    - obs_heatmap_memmap: Memory-mapped array for observation heatmaps.
    """

    # Ensure the directory exists
    Path(participant_memmap_file).parent.mkdir(parents=True, exist_ok=True)
    Path(obs_heatmap_memmap_file).parent.mkdir(parents=True, exist_ok=True)


    participant_memmap = np.memmap(
        participant_memmap_file,
        dtype=[('participant_id', 'S6'), ('trial_id', 'i4'), ('score', 'i4'), ('start_index', 'i4'), ('end_index', 'i4')],
        mode='r+',
        shape=(num_participants * num_trials_per_participant, )
    )


    obs_heatmap_memmap = np.memmap(
        obs_heatmap_memmap_file,
        dtype='float32',
        mode='r+',
        shape=(num_participants * num_trials_per_participant * 400, obs_channels + 1, *grid_shape)
    )

    return participant_memmap, obs_heatmap_memmap

# Example usage:
# participant_memmap, obs_heatmap_memmap = create_memmaps(participant_memmap_file, obs_heatmap_memmap_file, num_participants, num_trials_per_participant, obs_channels, grid_shape)

def process_data(participant_memmap, obs_heatmap_memmap, num_timesteps_to_consider):
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

    trial_data = defaultdict(list)
    trial_labels = defaultdict(list)

    for record in participant_memmap:
        participant_id, trial_id, score, start_idx, end_idx = record
        obs_heatmap_data = obs_heatmap_memmap[start_idx:end_idx]

        # Process visual observation and heatmap for the first X timesteps
        for timestep_data in obs_heatmap_data[:num_timesteps_to_consider]:
            visual_obs = timestep_data[:-1, :, :]  # All but last channel
            heatmap = timestep_data[-1, :, :]     # Last channel
            flattened_output_with_score = combine_and_standardize(visual_obs, heatmap, score)
            trial_data[(participant_id, trial_id)].append(flattened_output_with_score)
            trial_labels[(participant_id, trial_id)].append(score)

    return trial_data, trial_labels

# Example usage:
# trial_data, trial_labels = process_data(participant_memmap, obs_heatmap_memmap, num_timesteps_to_consider)

def process_trial_data(trial_data, trial_labels, num_bins = 3):
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

# Example usage:
# processed_data = process_trial_data(trial_data, trial_labels, num_bins)


