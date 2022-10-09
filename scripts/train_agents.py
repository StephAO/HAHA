from oai_agents.agents.rl import SingleAgentTrainer, MultipleAgentsTrainer
from oai_agents.common.arguments import get_arguments

from pathlib import Path
from stable_baselines3.common.evaluation import evaluate_policy


def calculate_agent_pairing_score_matrix(agents, args):
    eval_envs_kwargs = {'is_eval_env': True, 'args': args}
    eval_envs = [OvercookedGymEnv(**{'index': i, **eval_envs_kwargs}) for i in range(len(args.layout_names))]
    score_matrix = np.zeros((len(agents), len(agents)))
    for eval_env in eval_envs:
        for i, p1 in enumerate(agents):
            for j, p2 in enumerate(agents):
                env.set_teammate(p2.policy)
                mean_reward, std_reward = evaluate_policy(p1, eval_env, n_eval_episodes=1,
                                                          deterministic=True, warn=False, render=False)
                score_matrix[i][j] = mean_reward
    return score_matrix

### BASELINES ###

# SP
def create_selfplay_agent(args, training_steps=1e7):
    self_play_trainer = MultipleAgentsTrainer(args, name='selfplay', num_agents=1, use_lstm=False)
    self_play_trainer.train_agents(total_timesteps=training_steps)
    return self_play_trainer.get_agents()

# PP
def create_population_play_agent(args, pop_size=8, training_steps=1e7):
    pop_play_trainer = MultipleAgentsTrainer(args, name='pop_play', num_agents=pop_size, use_lstm=False, hidden_dim=64)
    pop_play_trainer.train_agents(total_timesteps=training_steps)
    all_agents = pop_play_trainer.get_agents()
    score_matrix = calculate_agent_pairing_score_matrix(all_agents, args)
    avg_score_per_agent = np.mean(score_matrix, axis=1)
    best_agent = all_agents[np.argmax(avg_score_per_agent)]
    return best_agent

# FCP
def create_fcp_population(args, training_steps=1e7):
    agents = []
    for h_dim in [16, 64, 256]:
        for use_lstm in [False, True]:
            seed = 88
            ck_rate = training_steps / 10
            name = f'lstm_{h_dim}' if use_lstm else f'no_lstm_{h_dim}'
            print(f'Starting training for: {name}')
            mat = MultipleAgentsTrainer(args, name=name, num_agents=1, use_lstm=use_lstm, hidden_dim=h_dim,
                                        fcp_ck_rate=ck_rate, seed=seed)
            mat.train_agents(total_timesteps=training_steps)
            mat.save_agents(path=(args.base_dir / 'agent_models' / 'sp'), tag=name)
            agents.extend(mat.get_fcp_agents())
    pop = MultipleAgentsTrainer(args, name='fcp_pop', num_agents=0)
    pop.set_agents(agents)
    pop.save_agents()
    return pop.get_agents()

def create_fcp_agent(teammates, args, training_steps=1e7):
    fcp_trainer = SingleAgentTrainer(teammates, args, name='fcp')
    fcp_trainer.train_agents(total_timesteps=training_steps)
    return fcp_trainer.get_agents()[0]

def create_all_agents(args, training_steps=1e7, agents_to_train='all'):
    agents = {}
    fcp_pop = None
    if agents_to_train == 'all' or 'sp' in agents_to_train:
        sp = MultipleAgentsTrainer.create_selfplay_agent(args, training_steps=5e6)
        agents['sp'] = sp
    if agents_to_train == 'all' or 'pp' in agents_to_train:
        pp = create_population_play_agent(args, training_steps=training_steps)
        agents['pp'] = pp
    if agents_to_train == 'all' or 'fcp' in agents_to_train:
        # Get teammates
        if fcp_pop is None:
            try:
                mat = MultipleAgentsTrainer(args, name='fcp_pop', num_agents=0)
                fcp_pop = mat.load_agents()
            except FileNotFoundError as e:
                print(f'Could not find saved FCP population, creating them from scratch...\nFull Error: {e}')
                fcp_pop = create_fcp_population(args, 4e6)
        # Create FCP agent
        fcp = create_fcp_agent(fcp_pop, args, training_steps)
        agents['fcp'] = fcp
    if agents_to_train == 'all' or 'pp' in agents_to_train:
        # Get teammates
        if fcp_pop is None:
            try:
                mat = MultipleAgentsTrainer(args, name='fcp_pop', num_agents=0)
                fcp_pop = mat.load_agents()
            except FileNotFoundError:
                print(f'Could not find saved FCP population, creating them from scratch...\nFull Error: {e}')
                fcp_pop = create_fcp_population(args, training_steps)

        # Create subtask worker
        try:
            name = 'multi_agent_subtask_worker'
            worker = MultiAgentSubtaskWorker.load(Path(args.base_dir / 'agent_models' / name / args.exp_name), args)
        except FileNotFoundError:
            print(f'Could not find saved subtask worker, creating them from scratch...\nFull Error: {e}')
            worker, _ = MultiAgentSubtaskWorker.create_model_from_scratch(args, teammates=fcp_pop)

        # Create manager
        rlmt = RLManagerTrainer(worker, fcp_pop, args)
        rlmt.train_agents(total_timesteps=training_steps)
        manager = rlmt.get_agents()[0]
        hrl = HierarchicalRL(worker, manager, args)
        hrl.save(Path(Path(args.base_dir / 'agent_models' / hrl.name / args.exp_name)))
    # TODO BCP


def create_test_population(args, training_steps=1e7):
    agents = []
    h_dim= 64
    seed = 888

    name = 'frame_stack'
    print(f'Starting training for: {name}')
    mat = MultipleAgentsTrainer(args, name=name, num_agents=1, use_frame_stack=True, hidden_dim=h_dim, seed=seed)
    mat.train_agents(total_timesteps=1e6)

    name = 'substasks'
    print(f'Starting training for: {name}')
    mat = MultipleAgentsTrainer(args, name=name, num_agents=1, use_subtask_counts=True, hidden_dim=h_dim, seed=seed)
    mat.train_agents(total_timesteps=1e6)

    name = 'frame_stack+substasks'
    print(f'Starting training for: {name}')
    mat = MultipleAgentsTrainer(args, name=name, num_agents=1, use_subtask_counts=True, use_frame_stack=True,
                                hidden_dim=h_dim, seed=seed)
    mat.train_agents(total_timesteps=1e6)

    name = 'nothing'
    print(f'Starting training for: {name}')
    mat = MultipleAgentsTrainer(args, name=name, num_agents=1, hidden_dim=h_dim, seed=seed)
    mat.train_agents(total_timesteps=1e6)

    return pop.get_agents()


if __name__ == '__main__':
    args = get_arguments()
    create_test_population(args)
    # create_all_agents(args, agents_to_train=['fcp'])
