import numpy as np

class Subtasks:
    SUBTASKS = ['get_onion_from_dispenser', 'get_onion_from_counter', 'put_onion_in_pot', 'put_onion_closer',
                'get_plate_from_dish_rack', 'get_plate_from_counter', 'put_plate_closer', 'get_soup',
                'get_soup_from_counter', 'put_soup_closer', 'serve_soup', 'unknown']
    HUMAN_READABLE_ST = ['I am grabbing an onion from the dispenser', 'I am grabbing an onion from the counter',
                         'I am putting my onion in the pot', 'I am placing my onion closer to the pot',
                         'I am grabbing a dish from the dispenser', 'I am grabbing dish from the counter',
                         'I am placing my dish closer to the pot', 'I am getting the soup',
                         'I am grabbing the soup from the counter', 'I am placing the soup closer',
                         'I am serving the soup', 'I am unsure what to do']
    NUM_SUBTASKS = len(SUBTASKS)
    SUBTASKS_TO_IDS = {s: i for i, s in enumerate(SUBTASKS)}
    IDS_TO_SUBTASKS = {v: k for k, v in SUBTASKS_TO_IDS.items()}
    HR_SUBTASKS_TO_IDS = {s: i for i, s in enumerate(HUMAN_READABLE_ST)}
    IDS_TO_HR_SUBTASKS = {v: k for k, v in HR_SUBTASKS_TO_IDS.items()}
    BASE_STS = ['get_onion_from_dispenser', 'put_onion_in_pot', 'get_plate_from_dish_rack', 'get_soup', 'serve_soup']
    SUPP_STS = ['put_onion_closer', 'put_plate_closer', 'put_soup_closer']
    COMP_STS = ['get_onion_from_counter', 'get_plate_from_counter', 'get_soup_from_counter']


def facing(layout, player):
    '''Returns terrain type that the agent is facing'''
    x, y = np.array(player.position) + np.array(player.orientation)
    if type(layout) == str:
        layout = [[t for t in row.strip("[]'")] for row in layout.split("', '")]
    return layout[y][x]

def calculate_completed_subtask(layout, prev_state, curr_state, p_idx):
    '''
    Find out which subtask has been completed between prev_state and curr_state for player with index p_idx
    :param layout: layout of the env
    :param prev_state: previous state
    :param curr_state: current state
    :param p_idx: player index
    :return: Completed subtask ID, or None if no subtask was completed
    '''
    prev_obj = prev_state.players[p_idx].held_object.name if prev_state.players[p_idx].held_object else None
    curr_obj = curr_state.players[p_idx].held_object.name if curr_state.players[p_idx].held_object else None
    tile_in_front = facing(layout, prev_state.players[p_idx])
    # Object held didn't change -- This interaction didn't actually transition to a new subtask
    if prev_obj == curr_obj:
        subtask = None
        return subtask

    # Pick up an onion
    elif prev_obj is None and curr_obj == 'onion':
        # Facing an onion dispenser
        if tile_in_front == 'O':
            subtask = 'get_onion_from_dispenser'
        # Facing a counter
        elif tile_in_front == 'X':
            subtask = 'get_onion_from_counter'
        else:
            raise ValueError(f'Unexpected transition. {prev_obj} -> {curr_obj} while facing {tile_in_front}')
    # Place an onion
    elif prev_obj == 'onion' and curr_obj is None:
        # Facing a pot
        if tile_in_front == 'P':
            subtask = 'put_onion_in_pot'
        # Facing a counter
        elif tile_in_front == 'X':
            subtask = 'put_onion_closer'
        else:
            raise ValueError(f'Unexpected transition. {prev_obj} -> {curr_obj} while facing {tile_in_front}')
    # Pick up a dish
    elif prev_obj is None and curr_obj == 'dish':
        # Facing a dish dispenser
        if tile_in_front == 'D':
            subtask = 'get_plate_from_dish_rack'
        # Facing a counter
        elif tile_in_front == 'X':
            subtask = 'get_plate_from_counter'
        else:
            raise ValueError(f'Unexpected transition. {prev_obj} -> {curr_obj} while facing {tile_in_front}')
    # Place a dish
    elif prev_obj == 'dish' and curr_obj is None:
        # Facing a counter
        if tile_in_front == 'X':
            subtask = 'put_plate_closer'
        else:
            raise ValueError(f'Unexpected transition. {prev_obj} -> {curr_obj} while facing {tile_in_front}')
    # Pick up soup from pot using plate
    elif prev_obj == 'dish' and curr_obj == 'soup':
        # Facing a counter
        if tile_in_front == 'P':
            subtask = 'get_soup'
        else:
            raise ValueError(f'Unexpected transition. {prev_obj} -> {curr_obj} while facing {tile_in_front}')
    # Pick up soup from counter
    elif prev_obj is None and curr_obj == 'soup':
        # Facing a counter
        if tile_in_front == 'X':
            subtask = 'get_soup_from_counter'
        else:
            raise ValueError(f'Unexpected transition. {prev_obj} -> {curr_obj} while facing {tile_in_front}')
    # Place soup
    elif prev_obj == 'soup' and curr_obj is None:
        # Facing a service station
        if tile_in_front == 'S':
            subtask = 'serve_soup'
        # Facing a counter
        elif tile_in_front == 'X':
            subtask = 'put_soup_closer'
        else:
            raise ValueError(f'Unexpected transition. {prev_obj} -> {curr_obj} while facing {tile_in_front}')
    else:
        raise ValueError(f'Unexpected transition. {prev_obj} -> {curr_obj}.')

    return Subtasks.SUBTASKS_TO_IDS[subtask]


def non_full_pot_exists(state, terrain):
    """
    Returns true if there is currently an empty soup
    NOTE: Assumes there are 2 pots
    """
    n_full_soups = 0
    for obj in state.objects.values():
        x, y = obj.position
        if obj.name == 'soup' and terrain[y][x] == 'P' and obj.is_full:
            n_full_soups += 1
    return n_full_soups < 2

def get_doable_subtasks(state, prev_subtask, layout_name, terrain, p_idx, n_counters):
    '''
    Returns a mask subtasks that could be accomplished for a given state and player idx
    :param state: curr state
    :param terrain: layout
    :param p_idx: player idx
    :return: a np array of length NUM_SUBTASKS holding a 1 if the corresponding subtask is doable, otherwise a 0
    '''
    # TODO instead of making exceptions per layout, we could generalize this to any layout by seeing if agents are
    # physically capable of moving to the required feature
    subtask_mask = np.zeros(Subtasks.NUM_SUBTASKS)
    # The player is not holding any objects, so it can only accomplish tasks that require getting an object
    if state.players[p_idx].held_object is None:
        # These are always possible if the player is not holding an object
        if not (layout_name == 'forced_coordination' and p_idx == 0):
            subtask_mask[Subtasks.SUBTASKS_TO_IDS['get_onion_from_dispenser']] = 1
        if not (layout_name == 'forced_coordination' and p_idx == 0):
            subtask_mask[Subtasks.SUBTASKS_TO_IDS['get_plate_from_dish_rack']] = 1

        # The following subtasks are only possible on some configurations for some players (this filters useless tasks)
        if not ((layout_name == 'forced_coordination' and p_idx == 1) or layout_name == 'asymmetric_advantages'):
            # These are only possible if the respective objects exist on a counter somewhere
            for obj in state.objects.values():
                x, y = obj.position
                if obj.name == 'onion' and prev_subtask != 'put_onion_closer':
                        subtask_mask[Subtasks.SUBTASKS_TO_IDS['get_onion_from_counter']] = 1
                elif obj.name == 'dish' and prev_subtask != 'put_plate_closer':

                        subtask_mask[Subtasks.SUBTASKS_TO_IDS['get_plate_from_counter']] = 1
                elif obj.name == 'soup' and obj.is_ready and terrain[y][x] != 'P' and prev_subtask != 'put_soup_closer':
                    # Different than above check because this task isn't useful for either p_idx
                    if layout_name != 'forced_coordination':
                        subtask_mask[Subtasks.SUBTASKS_TO_IDS['get_soup_from_counter']] = 1
    # The player is holding an onion, so it can only accomplish tasks that involve putting the onion somewhere
    elif state.players[p_idx].held_object.name == 'onion':
        # There must be an empty counter to put something down
        if len(state.objects.values()) < n_counters and prev_subtask != 'get_onion_from_counter':
            if not ((layout_name == 'forced_coordination' and p_idx == 0) or layout_name == 'asymmetric_advantages'):
                subtask_mask[Subtasks.SUBTASKS_TO_IDS['put_onion_closer']] = 1
        # There must be an empty pot to put an onion into
        if not (layout_name == 'forced_coordination' and p_idx == 1):
            if non_full_pot_exists(state, terrain):
                subtask_mask[Subtasks.SUBTASKS_TO_IDS['put_onion_in_pot']] = 1
    # The player is holding a plate, so it can only accomplish tasks that involve putting the plate somewhere
    elif state.players[p_idx].held_object.name == 'dish':
        # There must be an empty counter to put something down
        if len(state.objects.values()) < n_counters and prev_subtask != 'get_plate_from_counter':
            if not ((layout_name == 'forced_coordination' and p_idx == 0) or layout_name == 'asymmetric_advantages'):
                subtask_mask[Subtasks.SUBTASKS_TO_IDS['put_plate_closer']] = 1
        if not (layout_name == 'forced_coordination' and p_idx == 1):
            # Can only grab the soup using the plate if a soup is currently cooking
            for obj in state.objects.values():
                x, y = obj.position
                if obj.name == 'soup' and terrain[y][x] == 'P' and not obj.is_idle:
                    subtask_mask[Subtasks.SUBTASKS_TO_IDS['get_soup']] = 1
                    break
    # The player is holding a soup, so it can only accomplish tasks that involve putting the soup somewhere
    elif state.players[p_idx].held_object.name == 'soup':
        if not (layout_name == 'forced_coordination' and p_idx == 1):
            subtask_mask[Subtasks.SUBTASKS_TO_IDS['serve_soup']] = 1
        # There must be an empty counter to put something down
        if len(state.objects.values()) < n_counters and prev_subtask != 'get_soup_from_counter':
            if not (layout_name in ['forced_coordination', 'asymmetric_advantages']):
                subtask_mask[Subtasks.SUBTASKS_TO_IDS['put_soup_closer']] = 1

    # If no other subtask is possible, then set subtask to unknown
    if np.sum(subtask_mask) == 0:
        subtask_mask[Subtasks.SUBTASKS_TO_IDS['unknown']] = 1

    return subtask_mask
