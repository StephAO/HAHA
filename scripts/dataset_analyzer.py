import json
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, OvercookedGridworld, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
import numpy as np
from pathlib import Path
import pandas as pd
from oai_agents.common.subtasks import Subtasks, calculate_completed_subtask


layouts = 'all' #['asymmetric_advantages'] # 'all'
data_path = '../data/generated_data'  # args.base_dir / args.data_path / args.dataset
filename = 'tf_test_5_5.2.pickle' #'2019_hh_trials_all.pickle' 'all_trials.pickle' 'tf_test_5_5.1.pickle'
main_trials = pd.read_pickle(Path(data_path) / filename)
if filename == '2019_hh_trials_all.pickle':
    main_trials.loc[main_trials.layout_name == 'random0', 'layout_name'] = 'forced_coordination'
    main_trials.loc[main_trials.layout_name == 'random3', 'layout_name'] = 'counter_circuit'
print(f'Number of all trials: {len(main_trials)}')
if layouts != 'all':
    main_trials = main_trials[main_trials['layout_name'].isin(layouts)]

# Remove all transitions where both agents noop-ed
noop_trials = main_trials[main_trials['joint_action'] != '[[0, 0], [0, 0]]']
print(f'Number of {str(layouts)} trials without double noops: {len(noop_trials)}')
# print(main_trials['layout_name'])

action_ratios = {k: 0 for k in Action.ALL_ACTIONS}
all_noops, p1_noops, p2_noops, double_noops = 0, 0, 0, 0

#TODO a lot of this code is copy and pasted from overcooked dataset. Consider just making resuable functions
def str_to_actions(joint_action):
    """
    Convert df cell format of a joint action to a joint action as a tuple of indices.
    Used to convert pickle files which are stored as strings into np.arrays
    """
    global all_noops, p1_noops, p2_noops, double_noops
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
        action_ratios[joint_action[i]] += 1
        if joint_action[i] == Action.STAY:
            all_noops += 1
            if i == 0:
                p1_noops += 1
            elif i == 1:
                p2_noops += 1
    if joint_action[0] == Action.STAY and joint_action[1] == Action.STAY:
        double_noops += 1
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
    df['state'] = state
    return df

def add_subtasks(df):
    curr_trial = None
    subtask_start_idx = [0, 0]
    interact_id = Action.ACTION_TO_INDEX[Action.INTERACT]

    # Add columns in dataframe
    df['p1_curr_subtask'] = None
    df['p2_curr_subtask'] = None
    df['p1_next_subtask'] = None
    df['p2_next_subtask'] = None
    df = df.reset_index()

    # Iterate over all rows
    for index, row in df.iterrows():
        if row['trial_id'] != curr_trial:
            # Start of a new trial, label final actions of previous trial with unknown subtask
            subtask = Subtasks.SUBTASKS_TO_IDS['unknown']
            for i in range(2):
                df.loc[subtask_start_idx[i]:index-1, f'p{i + 1}_curr_subtask'] = subtask
                df.loc[subtask_start_idx[i]-1:index-1, f'p{i + 1}_next_subtask'] = subtask
            curr_trial = row['trial_id']
            # Store starting index of next subtask
            subtask_start_idx = [index, index]

        # For each agent
        for i in range(2):
            try: # Get next row
                next_row = df.loc[index + 1]
            except KeyError: # End of file, label last actions with 'unknown'
                subtask = Subtasks.SUBTASKS_TO_IDS['unknown']
                df.loc[subtask_start_idx[i]:index, f'p{i + 1}_curr_subtask'] = subtask
                df.loc[subtask_start_idx[i]-1:index, f'p{i + 1}_next_subtask'] = subtask
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
                df.loc[subtask_start_idx[i]:index, f'p{i + 1}_curr_subtask'] = subtask
                df.loc[subtask_start_idx[i]-1:index-1, f'p{i + 1}_next_subtask'] = subtask
                subtask_start_idx[i] = index + 1

    # Make sure that this data is defined everywhere
    assert not (df['p1_curr_subtask'].isna().any())
    assert not (df['p2_curr_subtask'].isna().any())
    assert not (df['p1_next_subtask'].isna().any())
    assert not (df['p2_next_subtask'].isna().any())

    # Calculate subtask ratios to be used as weights in a cross entropy loss
    subtask_counts = np.zeros(Subtasks.NUM_SUBTASKS)
    for i in range(2):
        counts = df[f'p{i+1}_curr_subtask'].value_counts().to_dict()
        print(f'Player {i+1} subtask splits')
        for k, v in counts.items():
            subtask_counts[k] += v
            print(f'{Subtasks.IDS_TO_SUBTASKS[k]}: {v}')

layout_to_env = {}
for layout in main_trials.layout_name.unique():
    layout_to_env[layout] = OvercookedEnv.from_mdp(OvercookedGridworld.from_layout_name(layout), horizon=400)
main_trials['joint_action'] = main_trials['joint_action'].apply(str_to_actions)

main_trials = main_trials.apply(str_to_obss, axis=1)

print(f'p1 noops: {p1_noops} / {len(main_trials)} = {p1_noops * 100 / len(main_trials):.3f}%')
print(f'p2 noops: {p2_noops} / {len(main_trials)} = {p2_noops * 100 / len(main_trials):.3f}%')
print(f'all noops: {all_noops} / {2*len(main_trials)} = {all_noops * 100 / (2*len(main_trials)):.3f}%')
print(f'double noops: {double_noops} / {len(main_trials)} = {double_noops * 100 / len(main_trials):.3f}%')
print(f'Total non-noop transitions: {2*len(main_trials) - all_noops}')

add_subtasks(main_trials)