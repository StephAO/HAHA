from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, OvercookedGridworld, Direction, Action, PlayerState
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from typing import Dict, Tuple
import numpy as np

def OAI_feats_closure():
    mlams = {}
    def OAI_get_feats(mdp: OvercookedGridworld, state: OvercookedState, grid_shape: tuple, horizon: int,
                      num_pots: int = 2, p_idx=None, goal_objects=None):
        """
        Uses Overcooked-ai's BC 96 dim BC featurization. Only returns agent_obs
        """
        nonlocal mlams
        if mdp.layout_name not in mlams:
            all_counters = mdp.get_counter_locations()
            COUNTERS_PARAMS = {
                'start_orientations': False,
                'wait_allowed': False,
                'counter_goals': all_counters,
                'counter_drop': all_counters,
                'counter_pickup': all_counters,
                'same_motion_goals': True
            }
            mlams[mdp.layout_name] = MediumLevelActionManager.from_pickle_or_compute(mdp, COUNTERS_PARAMS, force_compute=False)
        mlam = mlams[mdp.layout_name]
        agent_obs = mdp.featurize_state(state, mlam, num_pots=num_pots)
        if p_idx is not None:
            agent_obs = agent_obs[p_idx]
        else:
            agent_obs = np.stack(agent_obs, axis=0)
        return {'agent_obs': agent_obs}
    return OAI_get_feats

OAI_feats = OAI_feats_closure()


def OAI_encode_state(mdp: OvercookedGridworld, state: OvercookedState, grid_shape: tuple, horizon: int, p_idx=None,
                     goal_objects=None):
    """
    Uses Overcooked-ai's RL lossless encoding by stacking 27 binary masks (27xNxM). Only returns visual_obs.
    """
    visual_obs = mdp.lossless_state_encoding(state, horizon=horizon, goal_objects=goal_objects)
    visual_obs = np.stack(visual_obs, axis=0)
    # Reorder to channels first
    visual_obs = np.transpose(visual_obs, (0, 3, 1, 2))
    grid_shape = (2, visual_obs.shape[1], *grid_shape)
    assert len(visual_obs.shape) == len(grid_shape)
    assert all([visual_obs.shape[i] <= grid_shape[i] for i in range(len(visual_obs.shape))])
    padding_amount = [(0, grid_shape[i] - visual_obs.shape[i]) for i in range(len(grid_shape))]
    visual_obs = np.pad(visual_obs, padding_amount)
    if p_idx is not None:
        visual_obs = visual_obs[p_idx]
    return {'visual_obs': visual_obs}


def OAI_egocentric_encode_state(mdp: OvercookedGridworld, state: OvercookedState,
                                grid_shape: tuple, horizon: int, p_idx=None, goal_objects=None) -> Dict[str, np.array]:
    """
    Returns the egocentric encode state. Player will always be facing down (aka. SOUTH).
    grid_shape: The desired padded output shape from the egocentric view
    """
    if len(grid_shape) > 2 or grid_shape[0] % 2 == 0 or grid_shape[1] % 2 == 0:
        raise ValueError(f'Ego grid shape must be 2D and both dimensions must be odd! {grid_shape} is invalid.')

    # Get np.array representing current state
    visual_obs = mdp.lossless_state_encoding(state, horizon=horizon, goal_objects=goal_objects)  # This returns 2xNxMxF (F is # features)
    visual_obs = np.stack(visual_obs, axis=0)
    visual_obs = np.transpose(visual_obs, (0, 3, 1, 2))  # Reorder to features first --> 2xFxNxM
    num_players, num_features = visual_obs.shape[0], visual_obs.shape[1]

    # Remove orientation features since they are now irrelevant.
    # There are num_players * num_directions features.
    #num_layers_to_skip = num_players*len(Direction.ALL_DIRECTIONS)
    #idx_slice = list(range(num_players)) + list(range(num_players+num_layers_to_skip, num_features))
    #visual_obs = visual_obs[:, idx_slice, :, :]
    #assert visual_obs.shape[1] == num_features - num_layers_to_skip
    #num_features = num_features - num_layers_to_skip

    # Now we mask out the egocentric view
    assert len(state.players) == num_players
    if p_idx is not None:
        ego_visual_obs = get_egocentric_grid(visual_obs[p_idx], grid_shape, state.players[p_idx])
        assert ego_visual_obs.shape == (num_features, *grid_shape)
    else:
        ego_visual_obs = np.stack([get_egocentric_grid(visual_obs[idx], grid_shape, player)
                                   for idx, player in enumerate(state.players)])
        assert ego_visual_obs.shape == (num_players, num_features, *grid_shape)
    return {'visual_obs': ego_visual_obs}


def get_egocentric_grid(grid: np.array, ego_grid_shape: Tuple[int, int], player: PlayerState) -> np.array:
    assert len(grid.shape) == 3  # (Features, X, Y)
    # We pad so that we can mask the egocentric view without worrying about out-of-bounds errors
    x_pad_amount = ego_grid_shape[0] // 2
    y_pad_amount = ego_grid_shape[1] // 2
    padding_amount = ((0, 0), (x_pad_amount, x_pad_amount), (y_pad_amount, y_pad_amount))
    padded_grid = np.pad(grid, padding_amount)

    player_obs = padded_grid[:,
                             player.position[0]: player.position[0] + 2*x_pad_amount + 1,
                             player.position[1]: player.position[1] + 2*y_pad_amount + 1]

    if player.orientation == Direction.SOUTH:
        return player_obs
    elif player.orientation == Direction.NORTH:
        return np.rot90(player_obs, k=2, axes=(1, 2))
    elif player.orientation == Direction.EAST:
        return np.rot90(player_obs, k=-1, axes=(1, 2))
    elif player.orientation == Direction.WEST:
        return np.rot90(player_obs, k=1, axes=(1, 2))

    raise ValueError('Invalid direction! This should not be possible.')


ENCODING_SCHEMES = {
    'OAI_feats': OAI_feats,
    'OAI_lossless': OAI_encode_state,
    'OAI_egocentric': OAI_egocentric_encode_state,
}

if __name__ == '__main__':
    import timeit
    from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, OvercookedGridworld, Direction, Action
    env = OvercookedEnv.from_mdp(OvercookedGridworld.from_layout_name('asymmetric_advantages'), horizon=400)
    env.reset()
    for name, encoding_fn in ENCODING_SCHEMES.items():
        grid_shape = (7,7) if name == 'OAI_egocentric' else (10,10)
        obs = encoding_fn(env.mdp, env.state, grid_shape, 400)
        time_taken = timeit.timeit(lambda: encoding_fn(env.mdp, env.state, grid_shape, 400), number=10)
        d = {k: v.shape for k,v in obs.items()}
        print(f'{name} function returns dict: {d}) and takes {time_taken} to complete')
        # print(vis_obs.shape, agents_obs.shape)

