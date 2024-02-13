import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import pygame
import glob
import pyxdf

from collections import defaultdict
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, OvercookedGridworld
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from oai_agents.common.state_encodings import OAI_encode_state

desired_N = 9
desired_M = 5
next_index_for_obs_heatmap = 0


def map_eye_tracking(eye_data, top_left_x, top_left_y, surface_size, tile_size, grid_shape, hud_size):
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

    # Convert to DataFrame
    game_data_df = pd.DataFrame(game_data_df)
    eye_data_df = pd.DataFrame(eye_data_df)

    # Filter out rows where X position is not null
    eye_data_df = eye_data_df[eye_data_df['LRavgXposClip'].notnull()]
    eye_data_gen = eye_data_df.iterrows()
    _, prev_eye_row = next(eye_data_gen)

    game_data_df = game_data_df[game_data_df['GameEvents'].apply(lambda x: json.loads(x)['trial_id']) != 0].reset_index(
        drop=True)

    processed_data = []
    max_scores = get_max_scores_per_trial(game_data_df)
    processed_data = defaultdict(lambda: defaultdict(list))

    game_data_0 = json.loads(game_data_df.iloc[0]['GameEvents'])
    mdp = OvercookedGridworld.from_layout_name(game_data_0['layout_name'])

    for index, row in game_data_df.iterrows():
        time, game_str = row['Time'], row['GameEvents']
        game_data = json.loads(game_str)
        state = OvercookedState.from_dict(json.loads(game_data['state']))

        eye_data = []
        while prev_eye_row['Time'] <= time:
            eye_data.append((prev_eye_row['LRavgXposClip'], prev_eye_row['LRavgYposClip']))
            _, prev_eye_row = next(eye_data_gen)

        x, y, surface_size, tile_size, grid_shape, hud_size = game_data['dimension']
        hud_size = 50

        heatmap = map_eye_tracking(eye_data, x, y, surface_size, tile_size, grid_shape, hud_size)
        mdp = OvercookedGridworld.from_layout_name(game_data['layout_name'])
        visual_observation = OAI_encode_state(mdp, state, grid_shape, horizon=50)['visual_obs']
        visual_observation_for_one_player = visual_observation[0:1, :, :, :]

        trial_id = game_data['trial_id']
        visual_obs_padded = pad_to_shape(visual_observation_for_one_player, (desired_N, desired_M))
        heatmap_padded = pad_to_shape(heatmap, (desired_N, desired_M))
        participant_id = str(game_data.get('user_id'))
        highest_score = max_scores[trial_id]
        processed_data[participant_id][trial_id].append((visual_obs_padded, heatmap_padded, highest_score))

    return processed_data


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


def process_folder_with_xdf_files(folder_path, obs_heatmap_memmap, participant_memmap):
    global next_index_for_obs_heatmap
    xdf_files = glob.glob(str(Path(folder_path) / '*.xdf'))

    participant_index = 0
    for xdf_file in xdf_files:
        processed_data = process_xdf_file(xdf_file)

        for userid, trials in processed_data.items():
            for trial_id, data_list in trials.items():
                # Determine the starting index for this user-trial combination in obs_heatmap_memmap
                start_index = next_index_for_obs_heatmap

                for data in data_list:
                    observation, heatmap, score = data

                    # Store observation and heatmap data in obs_heatmap_memmap
                    obs_heatmap_memmap[next_index_for_obs_heatmap, :-1, :, :] = observation
                    obs_heatmap_memmap[next_index_for_obs_heatmap, -1, :, :] = heatmap

                    # Increment the index for the next data point in obs_heatmap_memmap
                    next_index_for_obs_heatmap += 1

                # Update the participant_memmap with a single record for the trial
                participant_memmap[participant_index] = (
                    userid.encode(), trial_id, score,
                    start_index, next_index_for_obs_heatmap
                )
                participant_index += 1


# process_folder_with_xdf_files("path/to/xdf/files")


def combine_and_standardize(visual_obs, heatmap, score, desired_N=9, desired_M=5):
    """
    Combines visual observation and heatmap into a single flattened array with the score appended at the end.
    """
    visual_obs_padded = pad_to_shape(visual_obs, (desired_N, desired_M))
    heatmap_padded = pad_to_shape(heatmap, (desired_N, desired_M))

    heatmap_expanded = np.expand_dims(heatmap_padded, axis=0)
    visual_obs_with_heatmap = np.concatenate((visual_obs_padded, heatmap_expanded), axis=0)

    flattened_output = visual_obs_with_heatmap.flatten()
    return flattened_output
