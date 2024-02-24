from collections import defaultdict
import numpy as np
import torch
# from scripts.memmap_creation import participant_memmap, obs_heatmap_memmap
from torch.utils.data import Dataset

class EyeGazeAndPlayDataset(Dataset):
    def __init__(self, participant_memmap, obs_heatmap_memmap, subtask_memmap, gaze_obj_memmap, encoding_type,
                 label_type, num_bins=4, num_timesteps=50):
        self.encoding_type = encoding_type
        self.label_type = label_type
        # TODO add linear classifier for go and ceg
        if self.encoding_type in ['go', 'ceg']:
            assert self.label_type != 'score', f'Encoding type {self.encoding_type} does not support score labels'
        self.num_bins = num_bins
        self.num_timesteps = num_timesteps
        self.num_trials_per_participant = 18
        self.horizon = 400

        self.inputs, self.labels = self.process_data(participant_memmap, obs_heatmap_memmap, subtask_memmap, gaze_obj_memmap)
        self.participant_ids = list(self.participant_ids)

        print(self.participant_ids)
        self.input_dim = self.inputs[(self.participant_ids[0], 1)].shape[-1]
        self.num_classes = {'score': self.num_bins, 'subtask': 12, 'q1': 7, 'q2': 7, 'q3': 8, 'q4': 7, 'q5': 7}[self.label_type]

        train_size, test_size = int(np.ceil(0.8 * len(self.participant_ids))), int(0.1 * np.ceil(len(self.participant_ids)))
        np.random.shuffle(self.participant_ids)
        # self.splits = {'train': self.participant_ids[:train_size],
        #                'test': self.participant_ids[train_size:train_size + test_size],
        #                'val': self.participant_ids[train_size + test_size:]}

        self.splits = {'train': [b'AF1021', b'CU2050', b'AF1057', b'AF1024', b'CU2047', b'AF1034', b'CU2017', b'AF1007', b'CU2014', b'CU1040', b'AF1046', b'AF1037', b'AF1048', b'CU2025', b'CU2046', b'CU2038', b'AF1056', b'CU2016', b'CU2032', b'AF1005', b'AF1039', b'CU2042', b'CU2008', b'AF1045', b'CU2018', b'AF1019', b'AF1026', b'AF1023', b'AF1052', b'CU2026', b'CU2019', b'AF1016', b'CU2048', b'CU2040', b'CU2052', b'AF1042', b'AF1008', b'AF1036', b'CU2015', b'AF1013', b'CU2028', b'CU2045', b'AF1012', b'AF1058', b'AF1032', b'AF1003', b'AF1053', b'AF1015', b'CU2044', b'CU2024', b'AF1051', b'AF1062', b'AF1028', b'CU2003', b'CU2002', b'AF1050', b'CU2030', b'CU2049', b'AF1041', b'AF1031'],
                       'test': [b'AF1028', b'CU2002', b'CU2035', b'CU2046', b'CU2050', b'AF1042', b'AF1005', b'AF1057', b'AF1013', b'AF1062'],
                       'val': [b'AF1003', b'CU2019', b'AF1050', b'AF1023', b'AF1056']}
        
        
        self.curr_split = 'train'

    def set_split(self, split):
        assert split in ['train', 'test', 'val']
        self.curr_split = split

    def __len__(self):
        return len(self.splits[self.curr_split]) * self.num_trials_per_participant

    def __getitem__(self, idx):
        participant_idx = idx // self.num_trials_per_participant
        participant_id = self.splits[self.curr_split][participant_idx]
        
        trial_id = (idx % self.num_trials_per_participant) + 1

        traj_start_idx = np.random.randint(0, self.horizon - self.num_timesteps)

        if(self.curr_split == 'test'):
            traj_start_idx = self.horizon // 2

        input = self.inputs[(participant_id, trial_id)][traj_start_idx:traj_start_idx + self.num_timesteps]
        if self.encoding_type in ['go', 'ceg']:
            # Sum over timesteps
            input = np.sum(input, dim=0)
            # Normalize
            input = input / np.sum(input, dim=-1, keepdims=True)

        label = self.labels[(participant_id, trial_id)][traj_start_idx:traj_start_idx + self.num_timesteps]
        
        return input, label

    def process_data(self, participant_memmap, obs_heatmap_memmap, subtask_memmap, gaze_obj_memmap):
        """
        Process the data from memory-mapped arrays.

        Parameters:
        - participant_memmap: Memory-mapped array with participant records.
        - obs_heatmap_memmap: Memory-mapped array with observation and heatmap data.
        - subtask_memmap: Memory-mapped array with subtask data.
        - gaze_obj_memmap: Memory-mapped array with gaze object data (i.e. counts of human looking at self/teammate/env

        Returns:
        - input_data: Dictionary with data to feed into network, keyed by (participant_id, trial_id). Data shape dependent on encoding type
        - labels: Dictionary with labels, keyed by (participant_id, trial_id). Data shape dependent on label type
        """
        null_participants = [b'AF1006', b'AF1035', b'AF1038', b'AF1049', b'AF1059', b'CU2034']
        input_data = {}
        labels = {}
        self.participant_ids = set()

        if self.label_type == 'score':
            self.all_scores = []

        for record in participant_memmap:
            participant_id, trial_id, score, start_idx, end_idx, question_1, question_2, question_3, question_4, question_5 = record
            if participant_id == b'' or participant_id in null_participants:
                continue
            self.participant_ids.add(participant_id)
            questions = [question_1, question_2, question_3, question_4, question_5]

            if self.label_type == 'score':
                self.all_scores.append(score)

            if start_idx == 0 and end_idx == 0:
                # Memmap is empty from this point on, stop processing. Should not happen if the number of participants
                # is set correctly
                break

            assert end_idx - start_idx == self.horizon, f'Expected {horizon} timesteps, got {end_idx - start_idx}'

            if self.encoding_type == 'gd':
                in_data = obs_heatmap_memmap[start_idx:end_idx, :-1]
            elif self.encoding_type == 'eg':
                in_data = obs_heatmap_memmap[start_idx:end_idx, -1:]
            elif self.encoding_type == 'gd+eg':
                in_data = obs_heatmap_memmap[start_idx:end_idx]
            elif self.encoding_type == 'ceg':
                # Collapse in data loader
                in_data = obs_heatmap_memmap[start_idx:end_idx, -1:]
            elif self.encoding_type == 'go':
                # Collapse in data loader
                in_data = gaze_obj_memmap[start_idx:end_idx]
            else:
                raise ValueError(f'{self.encoding_type} is not a valid encoding type')

            assert in_data.shape[0] == self.horizon, f'Expected {self.horizon} timesteps, got {in_data.shape[0]}'
            input_data[(participant_id, trial_id)] = in_data.reshape((in_data.shape[0], -1))

            if self.label_type == 'score':
                labels[(participant_id, trial_id)] = np.full((self.horizon, 1), score).tolist()
            elif self.label_type == 'subtask':
                labels[(participant_id, trial_id)] = subtask_memmap[start_idx:end_idx]
            elif self.label_type in ['q1', 'q2', 'q3', 'q4', 'q5']:
                q_idx = int(self.label_type[1]) - 1
                answer = questions[q_idx]
                labels[(participant_id, trial_id)] = np.full((self.horizon, 1), answer).tolist()
            else:
                raise ValueError(f'{self.label_type} is not a valid label type')

        # Process scores into proficienty bins
        if self.label_type == 'score':
            # Calculate quantiles as bin edges
            self.quantiles = np.percentile(self.all_scores, np.linspace(0, 100, self.num_bins + 1))
            self.quantiles[-1] = self.quantiles[-1] + 1  # To ensure the maximum score falls within the last bin
            print(f'Score min: {np.min(self.all_scores)}, max: {np.max(self.all_scores)}, mean: {np.mean(self.all_scores)}, Quantiles: {self.quantiles}')

            for k, v in labels.items():
                labels[k] = np.digitize(v, self.quantiles, right=False) - 1
                assert np.all(labels[k] >= 0) and np.all(labels[k] < self.num_bins), f'Invalid bin: {labels[k]}'

        return input_data, labels
