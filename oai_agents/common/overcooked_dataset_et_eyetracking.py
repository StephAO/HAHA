from oai_agents.common.state_encodings import ENCODING_SCHEMES
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, OvercookedGridworld, Action
from sklearn.preprocessing import OneHotEncoder



import json
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

import warnings
# Filter out UserWarnings
warnings.simplefilter(action='ignore', category=UserWarning)


class OvercookedDataset(Dataset):
    def __init__(self, dataset, layouts, args, seq_len, num_classes, add_subtask_info=True):

        self.seq_len = seq_len
        self.num_classes = num_classes
        self.step = 1
        self.add_subtask_info = add_subtask_info
        self.encoding_fn = ENCODING_SCHEMES['OAI_feats']
        self.data_path = '/Users/nikhilhulle/Desktop/HAHA-eyetracking/data/2019_hh_trials_all.pickle' #- For now the base classifier is based on 2019_hh_trials_all.pickle
        self.main_trials = pd.read_pickle(self.data_path)

        if dataset == '2019_hh_trials_all.pickle':
            self.main_trials.loc[self.main_trials.layout_name == 'random0', 'layout_name'] = 'forced_coordination'
            self.main_trials.loc[self.main_trials.layout_name == 'random3', 'layout_name'] = 'counter_circuit_o_1order'
        print(f'Number of all trials: {len(self.main_trials)}')
        self.layouts = layouts
        if layouts != 'all':
            self.main_trials = self.main_trials[self.main_trials['layout_name'].isin(layouts)]

        self.layout_to_mdp = {}
        self.grid_shape = [0, 0]
        for layout in self.main_trials.layout_name.unique():
            mdp = OvercookedGridworld.from_layout_name(layout)
            self.layout_to_mdp[layout] = mdp
            self.grid_shape[0] = max(mdp.shape[0], self.grid_shape[0])
            self.grid_shape[1] = max(mdp.shape[1], self.grid_shape[1])

        print(f'Number of {str(layouts)} trials: {len(self.main_trials)}, max grid size: {self.grid_shape}')

        def str_to_actions(joint_action):
            """
            Convert df cell format of a joint action to a joint action as a tuple of indices.
            Used to convert pickle files which are stored as strings into np.arrays
            """
            try:
                joint_action = json.loads(joint_action)
            except json.decoder.JSONDecodeError:
                # Hacky fix taken from https://github.com/HumanCompatibleAI/human_aware_rl/blob/master/human_aware_rl/human/data_processing_utils.py#L29
                joint_action = eval(joint_action)
            for i in range(2):
                if type(joint_action[i]) is list:
                    joint_action[i] = tuple(joint_action[i])
                if type(joint_action[i]) is str:
                    joint_action[i] = joint_action[i].lower()
            return [Action.ACTION_TO_INDEX[a] for a in joint_action]

        def str_to_obss(df):
            """
            Convert from a df cell format of a state to an Overcooked State
            Used to convert pickle files which are stored as strings into overcooked states
            """
            state = df['state']
            if type(state) is str:
                state = json.loads(state)
            state = OvercookedState.from_dict(state)
            mdp = self.layout_to_mdp[df['layout_name']]
            obs = self.encoding_fn(mdp, state, self.grid_shape, args.horizon)
            df['state'] = state
            df['agent_obs'] = obs['agent_obs']
            return df

        self.main_trials['joint_action'] = self.main_trials['joint_action'].apply(str_to_actions)
        self.main_trials = self.main_trials.apply(str_to_obss, axis=1)
        self.main_trials = self.main_trials[['agent_obs', 'joint_action', 'score', 'trial_id', 'score_total']]
        self.ind_trials = [self.main_trials.copy(deep=True), self.main_trials.copy(deep=True)]
        for i in range(2):
            self.ind_trials[i]['agent_obs'] = self.ind_trials[i].apply(lambda row: row['agent_obs'][i], axis=1)
            self.ind_trials[i]['action'] = self.ind_trials[i].apply(lambda row: row['joint_action'][i], axis=1)
            self.ind_trials[i]['trial_id'] = self.ind_trials[i]['trial_id']
            self.ind_trials[i]['score_total'] = self.ind_trials[i]['score_total']

        self.main_trials = pd.concat([self.ind_trials[i][['agent_obs', 'action', 'score', 'trial_id', 'score_total']] for i in range(2)], axis=0, ignore_index=True)
        self.encoder = OneHotEncoder(sparse=False)
        self._fit_encoder()
        self.update_score_for_each_trial_id()
        self.bin_scores()



    def get_action_weights(self):
        return self.action_weights


    def __len__(self):
        return (len(self.main_trials) - self.seq_len + 1) // self.step


    def update_score_for_each_trial_id(self):
        unique_trial_ids = self.main_trials['trial_id'].unique()

        for trial_id in unique_trial_ids:
            # Get the last timestep for the specific trial_id
            last_timestep_data = self.main_trials[self.main_trials['trial_id'] == trial_id].iloc[-1]
            max_score_for_trial = last_timestep_data['score_total']

            self.main_trials.loc[self.main_trials['trial_id'] == trial_id, 'score'] = max_score_for_trial
            print(f"Updated scores for trial ID {trial_id} to {max_score_for_trial}")



    """
    pd.qcut function bins the continuous data into equal sized buckets based on sample quantiles.
    In this case, with q=4, it would create quartiles, attempting to ensure that each quartile contains
    approximately the same number of data points.
    """
    def bin_scores(self):
        # Divide the scores into 4 bins based on quantiles

        score_bins = pd.qcut(self.main_trials['score'], q=self.num_classes, labels=False)
        self.main_trials['score_bins'] = score_bins


    def _fit_encoder(self):
        # Fit the encoder with unique action values
        unique_actions = np.unique(self.main_trials['action'].values).reshape(-1, 1)
        print(f"unique_Actions: {unique_actions}")
        self.encoder.fit(unique_actions)


    def __getitem__(self, idx):
        start_idx = idx * self.step
        end_idx = start_idx + self.seq_len

        data_sequence = self.main_trials.iloc[start_idx:end_idx]
        agent_obs_sequence = np.stack(data_sequence['agent_obs'].values)
        action_sequence = self.encoder.transform(data_sequence[['action']])

        score_total_sequence = np.stack(data_sequence['score_total'].values)
        score_bin = data_sequence['score_bins'].iloc[-1]

        #If sequence is shorter than seq_len, pad it
        padding_length = self.seq_len - len(agent_obs_sequence)
        if padding_length > 0:
            agent_obs_pad = np.zeros((padding_length, agent_obs_sequence.shape[1]))
            agent_obs_sequence = np.vstack((agent_obs_sequence, agent_obs_pad))

            action_pad = np.zeros((padding_length, action_sequence.shape[1]))
            action_sequence = np.vstack((action_sequence, action_pad))

            score_total_pad = np.zeros(padding_length)
            score_total_sequence = np.hstack((score_total_sequence, score_total_pad))

        return {'agent_obs': agent_obs_sequence, 'action': action_sequence, 'score_bins': score_bin, 'score': score_total_sequence}