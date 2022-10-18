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
        self.encoding_fn = ENCODING_SCHEMES['OAI_feats']
        self.data_path = args.base_dir / args.data_path / dataset
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
            mdp = self.layout_to_mdp[df['layout_name']]
            obs = self.encoding_fn(mdp, state, self.grid_shape, args.horizon)
            df['state'] = state
            df['agent_obs'] = obs['agent_obs']
            return df

        self.main_trials['joint_action'] = self.main_trials['joint_action'].apply(str_to_actions)
        self.main_trials = self.main_trials.apply(str_to_obss, axis=1)
        self.main_trials = self.main_trials[['agent_obs', 'joint_action']]

        self.ind_trials = [self.main_trials.copy(deep=True), self.main_trials.copy(deep=True)]
        for i in range(2):
            self.ind_trials[i]['agent_obs'] = self.ind_trials[i].apply(lambda row: row['agent_obs'][i], axis=1)
            self.ind_trials[i]['action'] = self.ind_trials[i].apply(lambda row: row['joint_action'][i], axis=1)

        self.main_trials = pd.concat(self.ind_trials, axis=0, ignore_index=True)
        # Calculate class weights for cross entropy
        self.action_weights = np.ones(6)
        for action in Action.ALL_ACTIONS:
            self.action_weights[Action.ACTION_TO_INDEX[action]] = self.action_ratios[action] + 1 # Avoids nans if there are no subtasks of that type
        self.action_weights = 1.0 / self.action_weights
        self.action_weights = len(Action.ALL_ACTIONS) * self.action_weights / self.action_weights.sum()

    def get_action_weights(self):
        return self.action_weights

    def __len__(self):
        return len(self.main_trials)

    def __getitem__(self, idx):
        data_point = self.main_trials.iloc[idx]
        return {'agent_obs': data_point['agent_obs'].squeeze(), 'action': data_point['action']}


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
