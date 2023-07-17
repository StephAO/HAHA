if __name__ == "__main__":
    """
    Sample commands
    python scripts/run_overcooked_game.py --agent human --teammate agent_models/HAHA
    """
    additional_args = [
        ('--agent', {'type': str, 'default': 'human', 'help': '"human" to used keyboard inputs or a path to a saved agent'}),
        ('--teammate', {'type': str, 'default': 'agent_models/HAHA', 'help': 'Path to saved agent to use as teammate'}),
        ('--layout', {'type': str, 'default': 'counter_circuit_o_1order', 'help': 'Layout to play on'}),
        ('--p_idx', {'type': int, 'default': 0, 'help': 'Player idx of agent (teammate will have other player idx), Can be 0 or 1.'})
    ]

    args = get_arguments(additional_args)

    t_idx = 1 - args.p_idx
    tm = load_agent(Path(args.teammate), args)
    tm.set_idx(t_idx, args.layout, is_hrl=isinstance(tm, HierarchicalRL), tune_subtasks=False)
    if args.agent == 'human':
        agent = args.agent
    else:
        agent = load_agent(Path(args.agent), args)
        agent.set_idx(args.p_idx, args.layout, is_hrl=isinstance(agent, HierarchicalRL), tune_subtasks=False)

    dc = OvercookedGUI(args, agent=agent, teammate=tm, layout=args.layout, p_idx=args.p_idx)
    dc.on_execute()
