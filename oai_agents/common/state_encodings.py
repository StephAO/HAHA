from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, OvercookedGridworld, Direction, Action, PlayerState
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from typing import Dict, Tuple
import numpy as np


# Note: Don't use
def encode_state(mdp: OvercookedGridworld, state: OvercookedState, grid_shape: tuple, horizon: int,
                 inc_agent_obs: bool=True, inc_soup_time: bool=True, inc_urgency=True, p_idx=None):
    """
    CNN Approach. Differs from OAI's lossless state by using 4 (opt. 5) layers instead of 20 and IDs instead of binary masks.
    Optionally adds robot observations (6 1-byte features) that includes additional information in a regular vector form.
    Visual Observation -- 3xNxM
        Map size: NxM
        4 (opt.5) Channels:
          - Agent ID
          - Unmovable Terrain ID
          - Movable items ID
          - Status ID 1 (for agents, this is item held; for pots it is the number of onions / ready)
          - Status ID 2 (for agents, this is direction; for pots it is time remaining)

    Robot Observation -- 7 (8 opt.):
        - Player idx
        - Absolute position x
        - Absolute position y
        - Direction (one hot)
        - Item id (if carrying an item)
        # - Facing empty counter
        - Time remaining
        - (Optional) Urgency (if horizon - overcooked_state.timestep < 40)
        TODOs (Maybe)
        - relative location to the other agent
        - relative location to the closest onion
        - relative location to the closest dish
        - relative location to the closest soup
        - relative location to the closest onion dispenser
        - relative location to the closest dish dispenser
        - relative location to the closest serving location
        - relative location to the closest pot (one for each pot state: empty, 1 onion, 2 onions, cooking, and ready).
    """
    AGENTS = ['player_1', 'player_2']
    UNMOVABLE_TERRAIN = ['floor', 'counter', 'pot', 'onion_dispenser', 'tomato_dispenser', 'dish_dispenser', 'serving_location']
    MOVABLE_ITEMS = ['onion', 'tomato', 'dish', 'soup']
    STATUS = ['north', 'south', 'east', 'west', 'empty', '1 onion', '2 onions', 'cooking', 'ready']

    UNMOVABLE_TERRAIN_TO_IDX = {agent: idx for idx, agent in enumerate(UNMOVABLE_TERRAIN)}
    MOVABLE_ITEMS_TO_IDX = {agent: idx for idx, agent in enumerate(MOVABLE_ITEMS)}
    STATUS_TO_IDX = {agent: idx for idx, agent in enumerate(STATUS)}

    num_channels = 5 if inc_soup_time else 4
    visual_obs = np.zeros( (num_channels, *grid_shape), dtype=np.int)
    if inc_agent_obs:
        agent_obs = np.zeros((2, 7 if inc_urgency else 6), dtype=np.float32)
    else:
        agent_obs = np.array([[],[]], dtype=np.float32)

    # STATUS done throughout
    S1_IDX = 3
    S2_IDX = 4

    # AGENTS
    A_IDX = 0
    for i, player in enumerate(state.players):
        visual_obs[A_IDX][player.position] = i + 1
        # STATUS
        if player.held_object is not None:
            visual_obs[S1_IDX][player.position] = MOVABLE_ITEMS_TO_IDX[player.held_object.name]
        visual_obs[S2_IDX][player.position] = Direction.DIRECTION_TO_INDEX[player.orientation]

        if inc_agent_obs:
            agent_obs[i][0] = i # Agent index
            agent_obs[i][1] = player.position[0] # Agent x
            agent_obs[i][2] = player.position[1] # Agent y
            agent_obs[i][3] = Direction.DIRECTION_TO_INDEX[player.orientation] # Agent direction
            if player.held_object is not None:
                agent_obs[i][4] = MOVABLE_ITEMS_TO_IDX[player.held_object.name]  # Agent object held
            agent_obs[i][5] = horizon - state.timestep  # Time remaining
            if inc_urgency and horizon - state.timestep < 40:
                agent_obs[i][6] = 1

    # Shared by both agents
    # UNMOVABLE TERRAIN
    UT_IDX = 1
    for loc in mdp.get_counter_locations():
        visual_obs[UT_IDX][loc] = UNMOVABLE_TERRAIN_TO_IDX['counter']
    for loc in mdp.get_pot_locations():
        visual_obs[UT_IDX][loc] = UNMOVABLE_TERRAIN_TO_IDX['pot']
    for loc in mdp.get_onion_dispenser_locations():
        visual_obs[UT_IDX][loc] = UNMOVABLE_TERRAIN_TO_IDX['onion_dispenser']
    for loc in mdp.get_tomato_dispenser_locations():
        visual_obs[UT_IDX][loc] = UNMOVABLE_TERRAIN_TO_IDX['tomato_dispenser']
    for loc in mdp.get_dish_dispenser_locations():
        visual_obs[UT_IDX][loc] = UNMOVABLE_TERRAIN_TO_IDX['dish_dispenser']
    for loc in mdp.get_serving_locations():
        visual_obs[UT_IDX][loc] = UNMOVABLE_TERRAIN_TO_IDX['serving_location']

    # MOVABLE ITEMS
    MI_IDX = 2
    for item in state.all_objects_list:
        if item.name == 'onion':
            visual_obs[MI_IDX][item.position] = MOVABLE_ITEMS_TO_IDX['onion']
        elif item.name == 'tomato':
            visual_obs[MI_IDX][item.position] = MOVABLE_ITEMS_TO_IDX['tomato']
        elif item.name == 'dish':
            visual_obs[MI_IDX][item.position] = MOVABLE_ITEMS_TO_IDX['dish']
        elif item.name == 'soup':
            # In this encoding, soup only exists outside the pot
            # Inside the pot it is treated as a different states of the pot
            if item.position not in mdp.get_pot_locations():
                visual_obs[MI_IDX][item.position] = MOVABLE_ITEMS_TO_IDX['soup']
            else:
                num_ingredients = len(item.ingredients)
                if item.is_idle:
                    visual_obs[S1_IDX][item.position] = num_ingredients
                elif item.is_ready:
                    visual_obs[S1_IDX][item.position] = 4 # ready
                else: # item is cooking
                    visual_obs[S1_IDX][item.position] = 3 # cooking
                    visual_obs[S2_IDX][item.position] = item.cook_time_remaining
        else:
            raise ValueError(f"Unrecognized object: {item.name}")

    if p_idx is not None:
        agent_obs = agent_obs[p_idx]
    else:
        visual_obs = np.tile(visual_obs, (2,1,1,1))
    return {'visual_obs': visual_obs, 'agent_obs': agent_obs}


def OAI_BC_featurize_state(mdp: OvercookedGridworld, state: OvercookedState, grid_shape: tuple, horizon: int, num_pots: int = 2, p_idx=None):
    """
    Uses Overcooked-ai's BC 64 dim BC featurization. Only returns agent_obs
    """
    mlam = MediumLevelActionManager.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=True)
    agent_obs = mdp.featurize_state(state, mlam, num_pots=num_pots)
    if p_idx is not None:
        agent_obs = agent_obs[p_idx]
    else:
        agent_obs = np.stack(agent_obs, axis=0)
    return {'agent_obs': agent_obs}


def OAI_RL_encode_state(mdp: OvercookedGridworld, state: OvercookedState, grid_shape: tuple, horizon: int, p_idx=None):
    """
    Uses Overcooked-ai's RL lossless encoding by stacking 20 binary masks (20xNxM). Only returns visual_obs.
    """
    visual_obs = mdp.lossless_state_encoding(state, horizon=horizon)
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
                                grid_shape: tuple, horizon: int, p_idx=None) -> Dict[str, np.array]:
    """
    Returns the egocentric encode state. Player will always be facing down (aka. SOUTH).
    grid_shape: The desired padded output shape from the egocentric view
    """
    if len(grid_shape) > 2 or grid_shape[0] % 2 == 0 or grid_shape[1] % 2 == 0:
        raise ValueError(f'Ego grid shape must be 2D and both dimensions must be odd! {grid_shape} is invalid.')

    # Get np.array representing current state
    visual_obs = mdp.lossless_state_encoding(state, horizon=horizon)  # This returns 2xNxMxF (F is # features)
    visual_obs = np.stack(visual_obs, axis=0)
    visual_obs = np.transpose(visual_obs, (0, 3, 1, 2))  # Reorder to features first --> 2xFxNxM
    num_players, num_features = visual_obs.shape[0], visual_obs.shape[1]

    # Remove orientation features since they are now irrelevant.
    # There are num_players * num_directions features.
    num_layers_to_skip = num_players*len(Direction.ALL_DIRECTIONS)
    idx_slice =list(range(num_players)) + list(range(num_players+num_layers_to_skip, num_features))
    visual_obs = visual_obs[:, idx_slice, :, :]
    assert visual_obs.shape[1] == num_features - num_layers_to_skip
    num_features = num_features - num_layers_to_skip

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
    'OAI_feats': OAI_BC_featurize_state,
    'OAI_lossless': OAI_RL_encode_state,
    'OAI_egocentric': OAI_egocentric_encode_state,
    'dense_lossless': encode_state
}

if __name__ == '__main__':
    import timeit
    from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, OvercookedGridworld, Direction, Action
    env = OvercookedEnv.from_mdp(OvercookedGridworld.from_layout_name('asymmetric_advantages'), horizon=400)
    env.reset()
    grid_shape = (15, 15)
    for name, encoding_fn in ENCODING_SCHEMES.items():
        vis_obs, agents_obs = encoding_fn(env.mdp, env.state, grid_shape, 400)
        time_taken = timeit.timeit(lambda: encoding_fn(env.mdp, env.state, grid_shape, 400), number=10)
        print(f'{name} function returns tuple with shapes({vis_obs.shape}, {agents_obs.shape}) and takes {time_taken} to complete')
        # print(vis_obs.shape, agents_obs.shape)

