import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import pygame

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, OvercookedGridworld
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer


def map_eye_tracking_to_grid(eye_data, top_left_x, top_left_y, surface_size, tile_size, grid_shape, hud_size):
    grid_top_left = (top_left_x, top_left_y + hud_size)
    heat_map = np.zeros(grid_shape)
    x_bins = list(range(tile_size, surface_size[0], tile_size))
    y_bins = list(range(tile_size, surface_size[1] - hud_size, tile_size))
    for x, y in eye_data:
        x -= grid_top_left[0]
        y -= grid_top_left[1]
        if not (0 < x < surface_size[0] and 0 < y < surface_size[1] - hud_size):
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


def combine_state_and_heatmap(state, heatmap, mdp, tile_size, hud_size, timestep):
    surface = StateVisualizer(tile_size=tile_size).render_state(state, grid=mdp.terrain_mtx, hud_data={"timestep": timestep})
    pil_string_image = pygame.image.tostring(surface, "RGBA", False)
    state_img = Image.frombytes("RGBA", surface.get_size(), pil_string_image)

    heatmap_img = np.zeros((*state_img.size, 4), dtype=int)
    heatmap_img[:, :, 0] = 255
    for x in range(heatmap.shape[0]):
        for y in range(heatmap.shape[1]):
            heatmap_img[x * tile_size: (x + 1) * tile_size, y * tile_size + hud_size: (y + 1) * tile_size + hud_size, 3] = heatmap[x][y] * 255
    heatmap_img = Image.fromarray(np.uint8(np.transpose(heatmap_img, (1, 0, 2))), "RGBA")

    state_img = Image.alpha_composite(state_img, heatmap_img)
    state_img.save(f'screenshots/{timestep}.png')



game_data_df = pd.read_csv('data/eye_tracking_data/P99_9_GameData.csv')
eye_data_df = pd.read_csv('data/eye_tracking_data/P99_9_GameEyeData.csv')
eye_data_df = eye_data_df[eye_data_df['LRavgXposClip'].notnull()]
eye_data_gen = eye_data_df.iterrows()
_, prev_eye_row = next(eye_data_gen)

mdp = OvercookedGridworld.from_layout_name('asymmetric_advantages')

for index, row in game_data_df.iterrows():
    time, game_str = row['GameTime'], row['GameEvents']
    game_data = json.loads(game_str)
    state = OvercookedState.from_dict(json.loads(game_data['state']))
    eye_data = []
    while prev_eye_row['Time'] <= time:
        eye_data.append((prev_eye_row['LRavgXposClip'], prev_eye_row['LRavgYposClip']))
        _, prev_eye_row = next(eye_data_gen)

    x, y, surface_size, tile_size, grid_shape, hud_size = game_data['dimension']
    tile_size, hud_size = 175, 50
    heatmap = map_eye_tracking_to_grid(eye_data, x, y, surface_size, tile_size, grid_shape, hud_size)
    combine_state_and_heatmap(state, heatmap, mdp, tile_size, hud_size, game_data['cur_gameloop'])
    # if index > 5:
    #     exit(0)