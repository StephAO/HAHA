def print_all_participant_data(participant_memmap, obs_heatmap_memmap):
    """
    Print data for each participant, including trial IDs, scores, and shapes of observations and heatmaps.
    """
    current_participant_id = None

    for record in participant_memmap:
        participant_id = record['participant_id'].decode()
        trial_id = record['trial_id']
        score = record['score']
        start_index = record['start_index']
        end_index = record['end_index']

        # Check if we're still processing the same participant
        if participant_id != current_participant_id:
            if current_participant_id is not None:
                print()  # Add a newline between participants for clarity
            current_participant_id = participant_id
            print(f"Participant ID: {participant_id}")

        # Print trial and score information
        print(f"  Trial ID: {trial_id}, Score: {score}")
        print(f"  Data Indices: {start_index} to {end_index}")

        # Access the corresponding observation and heatmap data
        obs_heatmap_data = obs_heatmap_memmap[start_index:end_index]
        print(f"  Observation Heatmap Data Shape: {obs_heatmap_data.shape}")


