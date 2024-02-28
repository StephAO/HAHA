import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
from PIL import Image
import pygame

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, OvercookedGridworld, Action
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
import pyxdf

from oai_agents.common.subtasks import Subtasks, calculate_completed_subtask


TERRAIN_CHAR_TO_NAME = {
    ' ': 'Floor',
    'X': 'Counter',
    'P': 'Pot',
    'D': 'Dish Dispenser',
    'O': 'Onion Dispenser',
    'S': 'Serving Area'
}


def xdf_to_panda_df(xdf_file):
    data, header = pyxdf.load_xdf(xdf_file)

    game_data_df = {}
    eye_data_df = {}

    for i in data:
        if i['info']['name'] == ['GameData']:
            game_data_df['Time'] = i['time_stamps']
            game_data_df['GameEvents'] = i['time_series']
            game_data_df['GameEvents'] = [ge[0] for ge in game_data_df['GameEvents']]
        if i['info']['name'] == ['tobiiPro']:
            # for j, channel in enumerate(i['info']['desc'][0]['channels'][0]['channel']):
            #     print(j, channel['label'], '-->', i['time_series'][0, j])
            eye_data_df['Time'] = i['time_stamps']
            eye_data_df['RX'] = i['time_series'][:, 0] * 1920
            eye_data_df['RY'] = i['time_series'][:, 1] * 1080
            eye_data_df['LX'] = i['time_series'][:, 15] * 1920
            eye_data_df['LY'] = i['time_series'][:, 16] * 1080
            eye_data_df['AvgX'] = (i['time_series'][:, 0] + i['time_series'][:, 15]) * 1920 / 2
            eye_data_df['AvgY'] = (i['time_series'][:, 1] + i['time_series'][:, 16]) * 1080 / 2
    game_data_df = pd.DataFrame(game_data_df)
    eye_data_df = pd.DataFrame(eye_data_df)
    eye_data_df.index += 1
    return game_data_df, eye_data_df


def create_full_dataframe(xdf_file):
    print(f'Parsing {xdf_file}')
    game_data_df, eye_data_df = xdf_to_panda_df(xdf_file)

    last_game_timestep = json.loads(game_data_df['GameEvents'].iloc[-1])['cur_gameloop']
    pd.options.display.float_format = '{:.2f}'.format

    while last_game_timestep != 400:
        game_data_df = game_data_df.iloc[:-1]
        last_game_timestep = json.loads(game_data_df['GameEvents'].iloc[-1])['cur_gameloop']

    game_data_gen = game_data_df.iterrows()
    _, game_data_row = next(game_data_gen)

    min_game_time, max_game_time = game_data_df['Time'].min(), game_data_df['Time'].max()

    eye_data_df = eye_data_df[eye_data_df['Time'] > min_game_time]
    eye_data_df = eye_data_df[eye_data_df['Time'] < max_game_time + 0.2]

    mdps = {}
    grid_centers = {}

    # timestep 0
    first_game_time, game_str = game_data_row['Time'], game_data_row['GameEvents']
    curr_game_data = json.loads(game_str)
    OvercookedGridworld.from_layout_name(curr_game_data['layout_name'])
    curr_state = OvercookedState.from_dict(json.loads(curr_game_data['state']))
    curr_game_timestep, curr_trial_id = curr_game_data['cur_gameloop'], curr_game_data['trial_id']
    # timestep 1
    _, game_data_row = next(game_data_gen)
    next_game_time, next_game_str = game_data_row['Time'], game_data_row['GameEvents']
    next_game_data = json.loads(next_game_str)
    next_state = OvercookedState.from_dict(json.loads(next_game_data['state']))

    user_id = curr_game_data['user_id']
    player_subtask_count = {}
    player_subtask_count[(curr_game_data['trial_id'], curr_game_data['agent'], curr_game_data['layout_name'])] = [0, 0]

    object_columns = {k: [] for k in ['Game_Timestep', 'Trial_ID', 'AvgGridX', 'AvgGridY', 'AvgObj', 'LObj', 'RObj']}
    eye_data_df['p1_curr_subtask'] = None
    eye_data_df['p2_curr_subtask'] = None
    subtask_start_idx = [1, 1]
    between_trial_rows = []

    on_last_row = False

    for index, row in eye_data_df.iterrows():
        # If time indicates a new game step, update curr_game_data
        if row['Time'] >= next_game_time:
            if on_last_row:
                break
            curr_state, curr_game_data = next_state, next_game_data
            curr_game_timestep, curr_trial_id = curr_game_data['cur_gameloop'], curr_game_data['trial_id']
            # Find time of next game step. If no next game step, set next_game_time to 0.2s after the last game step
            try:
                _, game_data_row = next(game_data_gen)
                next_game_time, next_game_str = game_data_row['Time'], game_data_row['GameEvents']
                next_game_data = json.loads(next_game_str)
                next_state = OvercookedState.from_dict(json.loads(next_game_data['state']))
            except StopIteration:
                on_last_row = True
                next_game_time += 0.2
                for i in range(2):
                    subtask = Subtasks.SUBTASKS_TO_IDS['unknown']
                    eye_data_df.loc[subtask_start_idx[i]:index + 1, f'p{i + 1}_curr_subtask'] = subtask
                    subtask_start_idx[i] = index + 1


            curr_next_same_game = True
            if curr_trial_id != next_game_data['trial_id']:
                curr_next_same_game = False
                player_subtask_count[(next_game_data['trial_id'], next_game_data['agent'], next_game_data['layout_name'])] = [0, 0]
                for i in range(2):
                    subtask = Subtasks.SUBTASKS_TO_IDS['unknown']
                    # NOTE because this is based on eye data index, the "next" subtask starts one eye dataframe index
                    # after the interact action occurs, NOT on the start of the next game step
                    eye_data_df.loc[subtask_start_idx[i]:index + 1, f'p{i + 1}_curr_subtask'] = subtask
                    subtask_start_idx[i] = index + 1

            # Check for subtask completion
            # For each agent
            try:
                joint_action = json.loads(curr_game_data['joint_action'])
            except json.decoder.JSONDecodeError:
                # Hacky fix taken from https://github.com/HumanCompatibleAI/human_aware_rl/blob/master/human_aware_rl/human/data_processing_utils.py#L29
                joint_action = eval(joint_action)
            for i in range(2):
                # All subtasks will start and end with an INTERACT action
                if joint_action[i] == 'interact' and curr_next_same_game:
                    # Find out which subtask has been completed
                    subtask = calculate_completed_subtask(curr_game_data['layout'], curr_state, next_state, i)
                    if subtask is not None:
                        # Label previous actions with subtask
                        eye_data_df.loc[subtask_start_idx[i]:index + 1, f'p{i + 1}_curr_subtask'] = subtask
                        subtask_start_idx[i] = index + 1
                        player_subtask_count[(curr_game_data['trial_id'], curr_game_data['agent'], curr_game_data['layout_name'])][i] += 1

        # New trial, remove data between trials
        if curr_trial_id != next_game_data['trial_id']:
            between_trial_rows.append(index)
            continue

        object_columns['Game_Timestep'].append(curr_game_timestep)
        object_columns['Trial_ID'].append(curr_trial_id)

        top_left_x, top_left_y, surface_size, tile_size, grid_shape, hud_size = curr_game_data['dimension']
        hud_size = 50

        grid_top_left = (top_left_x, top_left_y + hud_size)

        if curr_game_data['layout_name'] not in mdps:
            ln = curr_game_data['layout_name']
            mdps[ln] = OvercookedGridworld.from_layout_name(ln)
            grid_centers[ln] = []
            for x in range(mdps[ln].width):
                grid_row = []
                for y in range(mdps[ln].height):
                    grid_center = ((x + 0.5) * tile_size + top_left_x, (y + 0.5) * tile_size + top_left_y + hud_size)
                    terrain_type = mdps[ln].get_terrain_type_at_pos((x, y))
                    grid_row.append((grid_center, terrain_type))
                grid_centers[ln].append(grid_row)

        # +1 to make it inclusive to the total size
        x_bins = list(range(tile_size, surface_size[0] + 1, tile_size))
        y_bins = list(range(tile_size, surface_size[1] + 1, tile_size))

        for eye_type in ['Avg', 'L', 'R']:
            x, y = row[eye_type + 'X'], row[eye_type + 'Y']
            x -= grid_top_left[0]
            y -= grid_top_left[1]
            if np.isnan(x) or np.isnan(y):
                obj = 'null'
                grid_x, grid_y = -1, -1
            elif not (0 < x < x_bins[-1] and 0 < y < y_bins[-1]):
                # Out of bounds
                obj = 'OOB'
                grid_x, grid_y = -1, -1
            else:
                grid_x = np.digitize(x, x_bins)
                grid_y = np.digitize(y, y_bins)
                pos = (grid_x, grid_y)
                obj = eye_pos_to_gaze_obj(pos, curr_state, mdps[curr_game_data['layout_name']], curr_game_data['p_idx'])
            object_columns[eye_type + 'Obj'].append(obj)
            if eye_type == 'Avg':
                object_columns['AvgGridX'].append(grid_x)
                object_columns['AvgGridY'].append(grid_y)

    subtask = Subtasks.SUBTASKS_TO_IDS['unknown']
    for i in range(2):
        eye_data_df.loc[subtask_start_idx[i]:index, f'p{i + 1}_curr_subtask'] = subtask

    eye_data_df.drop(between_trial_rows, axis=0, inplace=True)

    for k, v in object_columns.items():
        eye_data_df[k] = v

    # pd.set_option('display.min_rows', 1000)
    # pd.set_option('display.max_rows', 1000)
    # print(eye_data_df.tail(1000))

    # Make sure that this data is defined everywhere
    assert not (eye_data_df['p1_curr_subtask'].isna().any())
    assert not (eye_data_df['p2_curr_subtask'].isna().any())

    # print(grid_centers)

    csv_filename = str(xdf_file).replace('.xdf', '_GazeLabels.csv')
    eye_data_df.to_csv(csv_filename)
    print(f'Created {csv_filename} of length {len(eye_data_df)} from {xdf_file}')
    return user_id, player_subtask_count


def eye_pos_to_gaze_obj(pos, state, mdp, p_idx):
    gaze_obj = TERRAIN_CHAR_TO_NAME[mdp.get_terrain_type_at_pos(pos)]
    if pos in state.objects:
        gaze_obj = state.objects[pos].name
    elif pos == state.players[0].position:
        if p_idx == 0:
            gaze_obj = 'human_player'
        else:
            gaze_obj = 'teammate'
    elif pos == state.players[1].position:
        if p_idx == 1:
            gaze_obj = 'human_player'
        else:
            gaze_obj = 'teammate'
    return gaze_obj


def map_eye_tracking_to_grid(eye_data, top_left_x, top_left_y, surface_size, tile_size, grid_shape, hud_size):
    grid_top_left = (top_left_x, top_left_y + hud_size)
    heat_map = np.zeros(grid_shape)
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

    if np.max(heat_map) == 0:
        heat_map = np.full_like(heat_map, 1e-8)
    else:
        heat_map /= np.max(heat_map)

    return heat_map


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


def create_heatmap(xdf_file):
    # game_data_df = pd.read_csv('data/eye_tracking_data/P99_9_GameData.csv')
    # eye_data_df = pd.read_csv('data/eye_tracking_data/P99_9_GameEyeData.csv')
    game_data_df, eye_data_df = xdf_to_panda_df(xdf_file)

    # eye_data_df = eye_data_df[eye_data_df['avgX'].notnull()]
    eye_data_gen = eye_data_df.iterrows()
    _, prev_eye_row = next(eye_data_gen)

    game_data_0 = json.loads(game_data_df.iloc[0]['GameEvents'])
    mdp = OvercookedGridworld.from_layout_name(game_data_0['layout_name'])

    for index, row in game_data_df.iterrows():
        time, game_str = row['Time'], row['GameEvents']
        game_data = json.loads(game_str)
        state = OvercookedState.from_dict(json.loads(game_data['state']))
        eye_data = []
        # print(time)

        # NOTE, I should be looking until the start of the next state, rather than until the start of this state
        # so this is all off by 1
        while prev_eye_row['Time'] <= time:
            eye_data.append((prev_eye_row['AvgX'], prev_eye_row['AvgY']))
            _, prev_eye_row = next(eye_data_gen)

        x, y, surface_size, tile_size, grid_shape, hud_size = game_data['dimension']
        hud_size = 50
        # tile_size, hud_size = game_data['dimension'][3], 50#game_data['dimension'][5]
        heatmap = map_eye_tracking_to_grid(eye_data, x, y, surface_size, tile_size, grid_shape, hud_size)
        combine_state_and_heatmap(state, heatmap, mdp, tile_size, hud_size, game_data['cur_gameloop'],
                                  game_data['trial_id'])
        # if index > 5:
        #     exit(0)


def contains_gaze_labels(files_in_folder):
    for filename in files_in_folder:
        if 'GazeLabels.csv' in filename:
            return True
    return False


if __name__ == '__main__':
    root_directory = 'C:/Users/anthony.ries/OneDrive - US Army/Documents/MDrive/Experiments/OAI_eyetracking/Data/'
    player_metrics = {}
    for folder in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, folder)
        print(folder_path)
        if os.path.isdir(folder_path) and (folder.startswith("AF") or folder.startswith("CU")):
            files_in_folder = os.listdir(folder_path)

            # Check if any file in the folder contains the text "GazeLabels"
            if contains_gaze_labels(files_in_folder):
                continue  # Skip to the next folder

            for filename in files_in_folder:
                file_path = os.path.join(folder_path, filename)

                # Check if it is a file and has a .xdf extension
                if os.path.isfile(file_path) and filename.endswith('.xdf'):
                    user_id, subtask_counts = create_full_dataframe(file_path)
                    player_metrics[user_id] = subtask_counts


    player_metrics_df = pd.DataFrame()
    for user_id, subtask_counts in player_metrics.items():
        for (trial_id, agent_name, layout_name), counts in subtask_counts.items():
            humanCRC = (counts[0] / (counts[0] + counts[1])) - (counts[1] / (counts[0] + counts[1]))
            player_metrics_df = player_metrics_df.append({'user_id': user_id, 'trial_id': trial_id,
                                                          'agent_name': agent_name, 'layout_name': layout_name,
                                                          'Human_PCA': counts[0], 'Agent_PCA': counts[1],
                                                          'Human_PCA_duration': counts[0] / 80, 'Agent_PCA_duration': counts[1] / 80,
                                                          'Human_CRC': humanCRC, 'Agent_CRC': -humanCRC}, ignore_index=True)
    print(player_metrics_df)
    player_metrics_df.to_csv(os.path.join(root_directory, 'player_metrics.csv'))

                    
    # create_heatmap('data/eye_tracking_data/oaiET_stephane_test2.xdf')
    # xdf_to_panda_df('data/eye_tracking_data/oaiET_stephane_test3.xdf')

    # create_full_dataframe('data/eye_tracking_data/oaiET_CU2025.xdf')
    # create_full_dataframe('data/eye_tracking_data/oaiET_stephane_test2.xdf')