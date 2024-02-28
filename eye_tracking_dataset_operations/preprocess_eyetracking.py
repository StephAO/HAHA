import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import pygame
import glob
import pyxdf
import csv

from collections import defaultdict
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, OvercookedGridworld
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from oai_agents.common.state_encodings import OAI_encode_state
from oai_agents.common.subtasks import Subtasks, calculate_completed_subtask


desired_N = 9
desired_M = 5

GAZE_OBJ_TO_IDX = {'self': 0, 'teammate': 1, 'environment': 2}
LAYOUT_TO_IDX = {'coordination_ring': 1, 'asymmetric_advantages': 2, 'counter_circuit_o_1order': 3}
AGENT_TO_IDX = {'haha': 0, 'selfplay': 1, 'random_agent': 2}

def map_eye_tracking(eye_data, top_left_x, top_left_y, surface_size, tile_size, grid_shape, hud_size, state, p_idx):
    grid_top_left = (top_left_x, top_left_y + hud_size)
    heat_map = np.zeros(grid_shape)
    gaze_obj_count = np.zeros(len(GAZE_OBJ_TO_IDX), dtype=int)
    # +1 to make it inclusive to the total size
    x_bins = list(range(tile_size, surface_size[0] + 1, tile_size))
    y_bins = list(range(tile_size, surface_size[1] + 1, tile_size))
    for x, y in eye_data:
        x -= grid_top_left[0]
        y -= grid_top_left[1]
        if not (0 < x < x_bins[-1] and 0 < y < y_bins[-1]):
            # Out of bounds
            continue

        x_bin = np.digitize(x, x_bins)
        y_bin = np.digitize(y, y_bins)
        heat_map[x_bin][y_bin] += 1
        gaze_obj_count[eye_pos_to_gaze_obj((x_bin, y_bin), state, p_idx)] += 1

    if np.max(heat_map) == 0:
        heat_map = np.full_like(heat_map, 1e-8)
    else:
        heat_map /= np.max(heat_map)

    return heat_map, gaze_obj_count

def eye_pos_to_gaze_obj(pos, state, p_idx):
    if pos == state.players[0].position:
        gaze_obj = 'self' if p_idx == 0 else 'teammate'
    elif pos == state.players[1].position:
        gaze_obj = 'self' if p_idx == 1 else 'teammate'
    else:
        gaze_obj = 'environment'
    return GAZE_OBJ_TO_IDX[gaze_obj]


def pad_to_shape(tensor, pad_shape):
    """
    Pads the given tensor to the specified shape.
    """
    padded = np.zeros(tensor.shape[:-2] + pad_shape)
    n, m = tensor.shape[-2], tensor.shape[-1]
    padded[..., :n, :m] = tensor
    return padded


def get_max_scores_per_trial(game_data_df):
    trial_scores = game_data_df['GameEvents'].apply(lambda x: (json.loads(x)['trial_id'], json.loads(x)['score']))
    trial_scores_df = pd.DataFrame(trial_scores.tolist(), columns=['trial_id', 'score'])
    max_scores = trial_scores_df.groupby('trial_id')['score'].max()
    return max_scores.to_dict()


def process_xdf_file(xdf_file_path):
    print(xdf_file_path)
    data, header = pyxdf.load_xdf(xdf_file_path)

    game_data_df = pd.DataFrame()
    eye_data_df = pd.DataFrame()

    for i in data:
        if i['info']['name'] == ['GameData']:
            game_data_df['Time'] = i['time_stamps']
            game_data_df['GameEvents'] = [ge[0] for ge in i['time_series']]
        elif i['info']['name'] == ['tobiiPro']:
            eye_data_df['Time'] = i['time_stamps']
            eye_data_df['LRavgXposClip'] = (i['time_series'][:, 0] + i['time_series'][:, 15]) * 1920 / 2
            eye_data_df['LRavgYposClip'] = (i['time_series'][:, 1] + i['time_series'][:, 16]) * 1080 / 2

    # Filter out rows where X position is not null
    eye_data_df = eye_data_df[eye_data_df['LRavgXposClip'].notnull()]
    eye_data_gen = eye_data_df.iterrows()
    _, prev_eye_row = next(eye_data_gen)

    game_data_df = game_data_df[game_data_df['GameEvents'].apply(lambda x: json.loads(x)['trial_id']) != 0].reset_index(
        drop=True)

    prev_time, time = None, None
    prev_trial_id, trial_id = None, None
    prev_state, state = None, None
    prev_participant_id, participant_id = None, None
    game_data = None
    curr_step = 0
    subtask_start_idx = [0, 0]
    max_scores = get_max_scores_per_trial(game_data_df)
    processed_data = defaultdict(lambda: defaultdict(list))
    subtask_data = defaultdict(lambda: defaultdict(lambda: np.zeros((400, 2))))
    gaze_obj_data = defaultdict(lambda: defaultdict(lambda: np.zeros((400, 3))))


    game_data_0 = json.loads(game_data_df.iloc[0]['GameEvents'])
    mdp = OvercookedGridworld.from_layout_name(game_data_0['layout_name'])

    for index, row in game_data_df.iterrows():
        prev_participant_id, prev_trial_id, prev_time, prev_state, prev_game_data = participant_id, trial_id, time, state, game_data
        time, game_str = row['Time'], row['GameEvents']
        game_data = json.loads(game_str)
        curr_step = curr_step + 1 if prev_trial_id == game_data['trial_id'] else 1
        if game_data['cur_gameloop'] != curr_step:
            if curr_step > 400:
                continue
        trial_id = game_data['trial_id']
        participant_id = str(game_data.get('user_id'))
        state = OvercookedState.from_dict(json.loads(game_data['state']))

        layout = LAYOUT_TO_IDX[game_data['layout_name']]
        agent = AGENT_TO_IDX[game_data['agent']]

        # Calculate subtask data
        # New trial, ignore eye_date between trials
        if prev_trial_id != trial_id:
            prev_time = time - 0.2
            for i in range(2):
                if prev_participant_id is not None:
                    subtask_data[prev_participant_id][prev_trial_id][subtask_start_idx[i]:, i] = Subtasks.SUBTASKS_TO_IDS['unknown']
                assert game_data['cur_gameloop'] == 1
                subtask_start_idx[i] = game_data['cur_gameloop']
        elif prev_game_data is not None:
            # Add subtask data
            try:
                joint_action = json.loads(prev_game_data['joint_action'])
            except json.decoder.JSONDecodeError:
                # Hacky fix taken from https://github.com/HumanCompatibleAI/human_aware_rl/blob/master/human_aware_rl/human/data_processing_utils.py#L29
                joint_action = eval(prev_game_data['joint_action'])

            for i in range(2):
                # All subtasks will start and end with an INTERACT action
                if joint_action[i] == 'interact':
                    # Find out which subtask has been completed
                    subtask = calculate_completed_subtask(prev_game_data['layout'], prev_state, state, i)
                    if subtask is not None:
                        # Label previous actions with subtask
                        assert game_data['cur_gameloop'] > subtask_start_idx[i]
                        # cur_gameloops is 1-indexed
                        subtask_data[participant_id][trial_id][subtask_start_idx[i]:game_data['cur_gameloop'] - 1, i] = subtask
                        subtask_start_idx[i] = game_data['cur_gameloop'] - 1

        # Calculate eye data
        eye_data = []
        while prev_eye_row['Time'] < prev_time:
            _, prev_eye_row = next(eye_data_gen)
            continue
        while prev_eye_row['Time'] <= time:
            eye_data.append((prev_eye_row['LRavgXposClip'], prev_eye_row['LRavgYposClip']))
            _, prev_eye_row = next(eye_data_gen)

        x, y, surface_size, tile_size, grid_shape, hud_size = game_data['dimension']
        hud_size = 50

        heatmap, gaze_obj_count = map_eye_tracking(eye_data, x, y, surface_size, tile_size, grid_shape, hud_size, state, game_data['p_idx'])
        gaze_obj_data[participant_id][trial_id][game_data['cur_gameloop'] - 1] = gaze_obj_count
        mdp = OvercookedGridworld.from_layout_name(game_data['layout_name'])
        visual_observation = OAI_encode_state(mdp, state, grid_shape, p_idx=game_data['p_idx'], horizon=400)['visual_obs']

        visual_obs_padded = pad_to_shape(visual_observation, (desired_N, desired_M))
        heatmap_padded = pad_to_shape(heatmap, (desired_N, desired_M))
        highest_score = max_scores[trial_id]
        processed_data[participant_id][trial_id].append((visual_obs_padded, heatmap_padded, highest_score, layout, agent))

    # Label last subtask as unknown
    for i in range(2):
        subtask_data[participant_id][trial_id][subtask_start_idx[i]:, i] = Subtasks.SUBTASKS_TO_IDS['unknown']
        subtask_start_idx[i] = game_data['cur_gameloop']

    return processed_data, subtask_data, gaze_obj_data


def combine_state_and_heatmap(state, heatmap, mdp, tile_size, hud_size, timestep, trial_id):
    surface = StateVisualizer(tile_size=tile_size).render_state(state, grid=mdp.terrain_mtx,
                                                                hud_data={"timestep": timestep})
    pil_string_image = pygame.image.tostring(surface, "RGBA", False)
    state_img = Image.frombytes("RGBA", surface.get_size(), pil_string_image)

    heatmap_img = np.zeros((*state_img.size, 4), dtype=int)
    heatmap_img[:, :, 0] = 255
    for x in range(heatmap.shape[0]):
        for y in range(heatmap.shape[1]):
            heatmap_img[x * tile_size: (x + 1) * tile_size, y * tile_size + hud_size: (y + 1) * tile_size + hud_size,
            3] = heatmap[x][y] * 255
    heatmap_img = Image.fromarray(np.uint8(np.transpose(heatmap_img, (1, 0, 2))), "RGBA")

    state_img = Image.alpha_composite(state_img, heatmap_img)
    Path(f'screenshots/').mkdir(parents=True, exist_ok=True)
    state_img.save(f'screenshots/{trial_id}_{timestep}.png')


def process_folder_with_xdf_files(folder_path, obs_heatmap_memmap, participant_memmap, subtask_memmap, gaze_obj_memmap):
    next_index_for_obs_heatmap = 0
    xdf_files = glob.glob(str(Path(folder_path) / '*.xdf'))

    participant_index = 0
    for xdf_file in xdf_files:
        game_and_eye_data, subtask_data, gaze_obj_data = process_xdf_file(xdf_file)

        for userid, trials in game_and_eye_data.items():
            for trial_id, data_list in trials.items():
                # Determine the starting index for this user-trial combination in obs_heatmap_memmap
                start_index = next_index_for_obs_heatmap

                assert len(data_list) == 400
                for i, data in enumerate(data_list):
                    observation, heatmap, score, layout, agent = data

                    # Store observation and heatmap data in obs_heatmap_memmap
                    obs_heatmap_memmap[next_index_for_obs_heatmap, :-1, :, :] = observation
                    obs_heatmap_memmap[next_index_for_obs_heatmap, -1, :, :] = heatmap

                    subtask_memmap[next_index_for_obs_heatmap] = subtask_data[userid][trial_id][i]
                    gaze_obj_memmap[next_index_for_obs_heatmap] = gaze_obj_data[userid][trial_id][i]

                    # Increment the index for the next data point in obs_heatmap_memmap
                    next_index_for_obs_heatmap += 1

                # Update the participant_memmap with a single record for the trial
                # TODO ASAP 0, 0, 0, 0, 0 should probably updated to include question answers
                participant_memmap[participant_index] = (
                    userid.encode(), trial_id, layout, agent, score,
                    start_index, next_index_for_obs_heatmap, 0, 0, 0, 0, 0
                )
                participant_index += 1

# process_folder_with_xdf_files("path/to/xdf/files")

def fill_participant_questions_from_csv(participant_memmap_file, csv_file_path):
    """
    Fill the Question_1 to Question_5 fields in participant_memmap based on a CSV file.
    """

    num_participants = 83  # Total number of participants
    num_trials_per_participant = 18  # Trials per participant

    # Load participant data from CSV into a dictionary
    participant_memmap = np.memmap(
        participant_memmap_file,
        dtype=[('participant_id', 'S6'), ('trial_id', 'i4'), ('layout', 'i4'), ('score', 'i4'), ('agent', 'i4'), ('start_index', 'i4'),
               ('end_index', 'i4'), ('Question_1', 'i4'), ('Question_2', 'i4'), ('Question_3', 'i4'), ('Question_4', 'i4'), ('Question_5', 'i4')],
        mode='r+',
        shape=(num_participants * num_trials_per_participant,)
    )
    participant_answers = {}
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Assuming 'participant_id' matches the CSV column name and is a string
            participant_id = row['Participant_ID']
            participant_answers[participant_id] = {
                'Question_1': int(row['Question1']),
                'Question_2': int(row['Question2']),
                'Question_3': int(row['Question3']),
                'Question_4': int(row['Question4']),
                'Question_5': int(row['Question5']),
            }

    # Iterate through participant_memmap and update questions for each participant
    for record in participant_memmap:
        # Decode participant_id from bytes to string to match keys in participant_answers

        participant_id = record['participant_id'].decode()
        if participant_id in participant_answers:
            # Retrieve the answers for the current participant
            answers = participant_answers[participant_id]
            # Update the record with the answers from the CSV
            record['Question_1'] = answers['Question_1']
            record['Question_2'] = answers['Question_2']
            record['Question_3'] = answers['Question_3']
            record['Question_4'] = answers['Question_4']
            record['Question_5'] = answers['Question_5']

# Example usage:
# fill_participant_questions_from_csv(participant_memmap, 'path/to/your/csv_file.csv')