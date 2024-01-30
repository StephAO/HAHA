import numpy as np
from pathlib import Path

from scripts.analyse_memmap import print_all_participant_data
from scripts.preprocess_eyetracking import process_folder_with_xdf_files

# Parameters
num_participants = 26  # Total number of participants
num_trials_per_participant = 18  # Trials per participant
obs_channels = 27  # Number of binary masks in the observation data
grid_shape = (9, 5)  # Padded grid size


def return_memmaps(participant_memmap_file, obs_heatmap_memmap_file, num_participants= num_participants, num_trials_per_participant=18,
                   obs_channels=27, grid_shape=(9, 5)):
    """
    Returns memory-mapped arrays for participants and observation heatmaps.

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
    # Path(participant_memmap_file).parent.mkdir(parents=True, exist_ok=True)
    # Path(obs_heatmap_memmap_file).parent.mkdir(parents=True, exist_ok=True)

    participant_memmap = np.memmap(
        participant_memmap_file,
        dtype=[('participant_id', 'S6'), ('trial_id', 'i4'), ('score', 'i4'), ('start_index', 'i4'),
               ('end_index', 'i4')],
        mode='r+',
        shape=(num_participants * num_trials_per_participant,)
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


def setup_and_process_xdf_files(data_folder, participant_memmap_file, obs_heatmap_memmap_file, num_participants=18,
                                num_trials_per_participant=num_participants,
                                obs_channels=27, grid_shape=(9, 5)):
    """
    Sets up memory-mapped files and processes a folder with XDF files.

    Parameters:
    - data_folder: Path to the folder containing XDF files.
    - participant_memmap_file: Path for the participant memory-mapped file.
    - obs_heatmap_memmap_file: Path for the observation heatmap memory-mapped file.
    - num_participants: Total number of participants.
    - num_trials_per_participant: Number of trials per participant.
    - obs_channels: Number of binary masks in the observation data.
    - grid_shape: Shape of the padded grid.
    """

    # Ensure the directory exists for memmap files
    Path(participant_memmap_file).parent.mkdir(parents=True, exist_ok=True)
    Path(obs_heatmap_memmap_file).parent.mkdir(parents=True, exist_ok=True)

    # Create participant memmap
    participant_memmap = np.memmap(
        participant_memmap_file,
        dtype=[('participant_id', 'S6'), ('trial_id', 'i4'), ('score', 'i4'), ('start_index', 'i4'), ('end_index', 'i4')],
        mode='w+',
        shape=(num_participants * num_trials_per_participant, )
    )

# Observation and Heatmap Memmap
    obs_heatmap_memmap = np.memmap(
        obs_heatmap_memmap_file,
        dtype='float32',
        mode='w+',
        shape=(num_participants * num_trials_per_participant * 400, obs_channels + 1, *grid_shape)
        # +1 in obs_channels to account for the heatmap
    )


# Process the XDF files in the data folder
    process_folder_with_xdf_files(data_folder, obs_heatmap_memmap, participant_memmap)


# function to analyse data
# print_all_participant_data(participant_memmap, obs_heatmap_memmap)