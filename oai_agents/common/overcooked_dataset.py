from oai_agents.common.arguments import get_arguments
from oai_agents.common.state_encodings import ENCODING_SCHEMES
from oai_agents.common.subtasks import Subtasks, calculate_completed_subtask
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, OvercookedGridworld, Action

import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class OvercookedDataset(Dataset):
    def __init__(self, dataset, layouts, args, add_subtask_info=True):
        self.add_subtask_info = add_subtask_info
        self.encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
        self.data_path = args.base_dir / args.data_path / dataset
        self.main_trials = pd.read_pickle(self.data_path)
        if dataset == '2019_hh_trials_all.pickle':
            self.main_trials.loc[self.main_trials.layout_name == 'random0', 'layout_name'] = 'forced_coordination'
            self.main_trials.loc[self.main_trials.layout_name == 'random3', 'layout_name'] = 'counter_circuit_o_1order'
        print(f'Number of all trials: {len(self.main_trials)}')
        self.layouts = layouts
        if layouts != 'all':
            self.main_trials = self.main_trials[self.main_trials['layout_name'].isin(layouts)]
        # print(self.main_trials['joint_action'])
        # print(self.main_trials['joint_action'].values[0])
        # print(type(self.main_trials['joint_action'].values[0]))


        self.layout_to_env = {}
        self.grid_shape = [0, 0]
        for layout in self.main_trials.layout_name.unique():
            env = OvercookedEnv.from_mdp(OvercookedGridworld.from_layout_name(layout), horizon=args.horizon)
            self.layout_to_env[layout] = env
            self.grid_shape[0] = max(env.mdp.shape[0], self.grid_shape[0])
            self.grid_shape[1] = max(env.mdp.shape[1], self.grid_shape[1])

        print(f'Number of {str(layouts)} trials: {len(self.main_trials)}, max grid size: {self.grid_shape}')
        # Remove all transitions where both agents noop-ed
        self.main_trials = self.main_trials[self.main_trials['joint_action'] != '[[0, 0], [0, 0]]']
        print(f'Number of {str(layouts)} trials without double noops: {len(self.main_trials)}')

        self.action_ratios = {k: 0 for k in Action.ALL_ACTIONS}

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
                assert joint_action[i] in Action.ALL_ACTIONS
                self.action_ratios[joint_action[i]] += 1
            return np.array([Action.ACTION_TO_INDEX[a] for a in joint_action])

        def str_to_obss(df):
            """
            Convert from a df cell format of a state to an Overcooked State
            Used to convert pickle files which are stored as strings into overcooked states
            """
            state = df['state']
            if type(state) is str:
                state = json.loads(state)
            state = OvercookedState.from_dict(state)
            env = self.layout_to_env[df['layout_name']]
            obs = self.encoding_fn(env.mdp, state, self.grid_shape, args.horizon)
            df['state'] = state
            if 'visual_obs' in obs:
                df['visual_obs'] = obs['visual_obs']
            if 'agent_obs' in obs:
                df['agent_obs'] = obs['agent_obs']
            return df

        self.main_trials['joint_action'] = self.main_trials['joint_action'].apply(str_to_actions)
        self.main_trials = self.main_trials.apply(str_to_obss, axis=1)

        self.add_subtasks()

        # Calculate class weights for cross entropy
        self.action_weights = np.ones(6)
        for action in Action.ALL_ACTIONS:
            self.action_weights[Action.ACTION_TO_INDEX[action]] = self.action_ratios[action] + 1 # Avoids nans if there are no subtasks of that type
        self.action_weights = 1.0 / self.action_weights
        self.action_weights = len(Action.ALL_ACTIONS) * self.action_weights / self.action_weights.sum()

    def get_action_weights(self):
        return self.action_weights

    def get_subtask_weights(self):
        return self.subtask_weights

    def __len__(self):
        return len(self.main_trials)

    def __getitem__(self, idx):
        data_point = self.main_trials.iloc[idx]
        item_dict = {'joint_action': data_point['joint_action'],
                     'subtasks': np.array( [[data_point['p1_curr_subtask'], data_point['p2_curr_subtask']],
                                           [data_point['p1_next_subtask'], data_point['p2_next_subtask']]])}
        if 'visual_obs' in data_point:
            item_dict['visual_obs'] = data_point['visual_obs'].squeeze()
        if 'agent_obs' in data_point:
            item_dict['agent_obs'] = data_point['agent_obs'].squeeze()
        return item_dict

    def add_subtasks(self):
        curr_trial = None
        subtask_start_idx = [0, 0]
        interact_id = Action.ACTION_TO_INDEX[Action.INTERACT]

        # Add columns in dataframe
        self.main_trials['p1_curr_subtask'] = None
        self.main_trials['p2_curr_subtask'] = None
        self.main_trials['p1_next_subtask'] = None
        self.main_trials['p2_next_subtask'] = None
        self.main_trials = self.main_trials.reset_index()

        # Iterate over all rows
        for index, row in tqdm(self.main_trials.iterrows()):
            if row['trial_id'] != curr_trial:
                # Start of a new trial, label final actions of previous trial with unknown subtask
                subtask = Subtasks.SUBTASKS_TO_IDS['unknown']
                for i in range(len(row['state'].players)):
                    self.main_trials.loc[subtask_start_idx[i]:index-1, f'p{i + 1}_curr_subtask'] = subtask
                    self.main_trials.loc[subtask_start_idx[i]-1:index-1, f'p{i + 1}_next_subtask'] = subtask
                curr_trial = row['trial_id']
                # Store starting index of next subtask
                subtask_start_idx = [index, index]

            # For each agent
            for i in range(len(row['state'].players)):
                try: # Get next row
                    next_row = self.main_trials.loc[index + 1]
                except KeyError: # End of file, label last actions with 'unknown'
                    subtask = Subtasks.SUBTASKS_TO_IDS['unknown']
                    self.main_trials.loc[subtask_start_idx[i]:index, f'p{i + 1}_curr_subtask'] = subtask
                    self.main_trials.loc[subtask_start_idx[i]-1:index, f'p{i + 1}_next_subtask'] = subtask
                    subtask_start_idx[i] = index + 1
                    continue

                # All subtasks will start and end with an INTERACT action
                if row['joint_action'][i] == interact_id:
                    # Make sure the next row is part of the current trial
                    if next_row['trial_id'] == curr_trial:
                        # Find out which subtask has been completed
                        subtask = calculate_completed_subtask(row['layout'], row['state'], next_row['state'], i)
                        if subtask is None:
                            continue # No completed subtask, continue to next player
                    else:
                        continue

                    # Label previous actions with subtask
                    self.main_trials.loc[subtask_start_idx[i]:index, f'p{i + 1}_curr_subtask'] = subtask
                    self.main_trials.loc[subtask_start_idx[i]-1:index-1, f'p{i + 1}_next_subtask'] = subtask
                    subtask_start_idx[i] = index + 1

        # Make sure that this data is defined everywhere
        assert not (self.main_trials['p1_curr_subtask'].isna().any())
        assert not (self.main_trials['p2_curr_subtask'].isna().any())
        assert not (self.main_trials['p1_next_subtask'].isna().any())
        assert not (self.main_trials['p2_next_subtask'].isna().any())

        # Calculate subtask ratios to be used as weights in a cross entropy loss
        self.subtask_weights = np.ones(Subtasks.NUM_SUBTASKS)
        for i in range(2):
            counts = self.main_trials[f'p{i+1}_next_subtask'].value_counts().to_dict()
            for k, v in counts.items():
                self.subtask_weights[k] += v
        self.subtask_weights = 1.0 / self.subtask_weights
        self.subtask_weights = Subtasks.NUM_SUBTASKS * self.subtask_weights / self.subtask_weights.sum()


def main():
    args = get_arguments()
    env = OvercookedEnv.from_mdp(OvercookedGridworld.from_layout_name(args.layout_names[0]), horizon=400)
    encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
    OD = OvercookedDataset(encoding_fn, args.layout_names, args) #OvercookedDataset(env, encoding_fn, args)

    dataloader = DataLoader(OD, batch_size=1, shuffle=True, num_workers=0)
    for batch in dataloader:
        print(batch)
        exit(0)


if __name__ == '__main__':
    main()
