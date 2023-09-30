import json
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, OvercookedGridworld, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
import numpy as np
from pathlib import Path
import pandas as pd
from oai_agents.common.subtasks import Subtasks, calculate_completed_subtask
from scipy.stats import entropy


layouts = 'all' #['asymmetric_advantages'] # 'all'
data_path = 'data/'  # args.base_dir / args.data_path / args.dataset
filename = '2019_hh_trials_all.pickle'# 'all_trials.pickle' 'tf_test_5_5.1.pickle'
main_trials = pd.read_pickle(Path(data_path) / filename)
if filename == '2019_hh_trials_all.pickle':
    main_trials.loc[main_trials.layout_name == 'random0', 'layout_name'] = 'forced_coordination'
    main_trials.loc[main_trials.layout_name == 'random3', 'layout_name'] = 'counter_circuit'
print(f'Number of all trials: {len(main_trials)}')
if layouts != 'all':
    main_trials = main_trials[main_trials['layout_name'].isin(layouts)]

# Remove all transitions where both agents noop-ed
main_trials = main_trials[main_trials['joint_action'] != '[[0, 0], [0, 0]]']
print(f'Number of {str(layouts)} trials without double noops: {len(main_trials)}')
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

layout_to_env = {}
for layout in main_trials.layout_name.unique():
    layout_to_env[layout] = OvercookedEnv.from_mdp(OvercookedGridworld.from_layout_name(layout), horizon=400)
main_trials['joint_action'] = main_trials['joint_action'].apply(str_to_actions)

main_trials = main_trials.apply(str_to_obss, axis=1)



actions_per_state = {}
for index, row in main_trials.iterrows():
    for i in range(2):
        state_hash = row["state"].specific_hash(i)
        if state_hash not in actions_per_state:
            actions_per_state[state_hash] = {k: 0 for k in range(len(Action.ALL_ACTIONS))}
            # print('!')
        # print(actions_per_state[state_hash])
        actions_per_state[state_hash][row['joint_action'][i]] += 1

action_dist_per_state = []
for k, v in actions_per_state.items():
    action_dist_per_state.append((np.sum(list(actions_per_state[k].values())), actions_per_state[k]))


# import copy
# adps = copy.deepcopy(action_dist_per_state)[:50]
# for i, (total_actions, dist) in enumerate(adps):
#     if total_actions > 10:
#         print(f'pre : state {i} has {total_actions} actions, with distribution {dist}')

entropy_per_state = []
action_dist_per_state.sort(key=lambda x: x[0], reverse=True)
for i, (total_actions, dist) in enumerate(action_dist_per_state):
    if total_actions > 10:
        print(f'post: state {i} has {total_actions} actions, with distribution {dist}')
        probs = np.zeros(len(dist))
        for k, v in dist.items():
            probs[k] = v
        probs = probs / np.sum(probs)
        entropy_per_state.append(entropy(probs))


print(f'AVERAGE ENTRPOPY: {np.mean(entropy_per_state)}')
# print(f'p1 noops: {p1_noops} / {len(main_trials)} = {p1_noops * 100 / len(main_trials):.3f}%')
# print(f'p2 noops: {p2_noops} / {len(main_trials)} = {p2_noops * 100 / len(main_trials):.3f}%')
# print(f'all noops: {all_noops} / {2*len(main_trials)} = {all_noops * 100 / (2*len(main_trials)):.3f}%')
# print(f'double noops: {double_noops} / {len(main_trials)} = {double_noops * 100 / len(main_trials):.3f}%')
# print(f'Total non-noop transitions: {2*len(main_trials) - all_noops}')
#
# add_subtasks(main_trials)