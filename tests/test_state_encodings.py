import pytest
import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import PlayerState, Direction
from oai_agents.common.state_encodings import get_egocentric_grid, OAI_egocentric_encode_state, OAI_RL_encode_state
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv


@pytest.fixture
def grid():
    # 3,4 grid
    return np.array([
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 1]],  # player
        [[0, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 0]],  # onion
        [[0, 0, 0, 0],
         [1, 1, 1, 1],
         [0, 0, 0, 0]],  # counter
    ])


@pytest.fixture
def player():
    return PlayerState((2, 3), Direction.SOUTH)


def test_get_egocentric_grid_bottom_right(grid, player):
    expected_grid = np.array([
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],  # player
        [[1, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],  # onion
        [[1, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],  # counter
    ])
    assert np.array_equal(get_egocentric_grid(grid, (3, 3), player), expected_grid)


def test_get_egocentric_grid_top_right(grid, player):
    grid[0] = np.zeros((3, 4))
    grid[0, 0, 3] = 1
    player.position = (0, 3)
    expected_grid = np.array([
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],  # player
        [[0, 0, 0],
         [0, 0, 0],
         [1, 0, 0]],  # onion
        [[0, 0, 0],
         [0, 0, 0],
         [1, 1, 0]],  # counter
    ])
    assert np.array_equal(get_egocentric_grid(grid, (3, 3), player), expected_grid)


def test_get_egocentric_grid_top_left(grid, player):
    grid[0] = np.zeros((3, 4))
    grid[0, 0, 0] = 1
    player.position = (0, 0)
    expected_grid = np.array([
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],  # player
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],  # onion
        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 1]],  # counter
    ])
    assert np.array_equal(get_egocentric_grid(grid, (3, 3), player), expected_grid)


def test_get_egocentric_grid_bottom_left(grid, player):
    grid[0] = np.zeros((3, 4))
    grid[0, 2, 0] = 1
    player.position = (2, 0)
    expected_grid = np.array([
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],  # player
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],  # onion
        [[0, 1, 1],
         [0, 0, 0],
         [0, 0, 0]],  # counter
    ])
    assert np.array_equal(get_egocentric_grid(grid, (3, 3), player), expected_grid)


def test_get_egocentric_grid_big_grid(grid, player):
    expected_grid = np.array([
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],  # player
        [[0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],  # onion
        [[0, 0, 0, 0, 0],
         [1, 1, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],  # counter
    ])
    assert np.array_equal(get_egocentric_grid(grid, (5, 5), player), expected_grid)


# South is covered by the default case
def test_get_egocentric_grid_north(grid, player):
    player.orientation = Direction.NORTH
    expected_grid = np.array([
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],  # player
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 1]],  # onion
        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 1]],  # counter
    ])
    assert np.array_equal(get_egocentric_grid(grid, (3, 3), player), expected_grid)


def test_get_egocentric_grid_east(grid, player):
    player.orientation = Direction.EAST
    expected_grid = np.array([
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],  # player
        [[0, 0, 1],
         [0, 0, 0],
         [0, 0, 0]],  # onion
        [[0, 0, 1],
         [0, 0, 1],
         [0, 0, 0]],  # counter
    ])
    assert np.array_equal(get_egocentric_grid(grid, (3, 3), player), expected_grid)


def test_get_egocentric_grid_west(grid, player):
    player.orientation = Direction.WEST
    expected_grid = np.array([
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],  # player
        [[0, 0, 0],
         [0, 0, 0],
         [1, 0, 0]],  # onion
        [[0, 0, 0],
         [1, 0, 0],
         [1, 0, 0]],  # counter
    ])
    assert np.array_equal(get_egocentric_grid(grid, (3, 3), player), expected_grid)


def test_OAI_egocentric_encode_state():
    env = OvercookedEnv.from_mdp(OvercookedGridworld.from_layout_name('asymmetric_advantages'), horizon=400)

    rl_results = OAI_RL_encode_state(env.mdp, env.state, (20, 20), 400)
    num_feats = rl_results['visual_obs'].shape[1] - 8  # Removing player orientations

    # Check shape
    result = OAI_egocentric_encode_state(env.mdp, env.state, (5, 5), 400, p_idx=0)
    assert result['visual_obs'].shape == (num_feats, 5, 5)  # check size
    result = OAI_egocentric_encode_state(env.mdp, env.state, (5, 5), 400)
    assert result['visual_obs'].shape == (2, num_feats, 5, 5)

    # Check that both players are in an egocentric position
    expected_ego_pos = np.array([[0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0]])
    assert np.array_equal(result['visual_obs'][0][0], expected_ego_pos)
    assert np.array_equal(result['visual_obs'][1][0], expected_ego_pos)
