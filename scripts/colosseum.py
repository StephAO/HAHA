from oai_agents.agents.agent_utils import load_agent, DummyAgent
from oai_agents.common.arguments import get_arguments
from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import tqdm
import train_agents

def eval_agents_with_various_teammates(agents_to_evaluate, teammates):
    eval_envs_kwargs = {'is_eval_env': True, 'args': args, 'horizon': 400}
    eval_envs = [OvercookedGymEnv(**{'env_index': i, **eval_envs_kwargs}) for i in range(len(args.layout_names))]
    score_matrices = {ln: np.zeros((len(agents), len(agents))) for ln in args.layout_names}

    tot_rounds = len(eval_envs) * len(agents_to_evaluate) * len(teammates)
    with tqdm(total=tot_rounds) as pbar:
        for eval_env in eval_envs:
            for i, p1 in enumerate(agents_to_evaluate):
                for j, p2 in enumerate(teammates):
                    p2 = p2[eval_env.layout_name] if type(p2) == dict else p2
                    env.set_teammate(p2)
                    mean_reward, std_reward = evaluate_policy(p1, eval_env, n_eval_episodes=10,
                                                              deterministic=False, warn=False, render=False)
                    score_matrices[eval_env.layout_name][i][j] = mean_reward
                    pbar.update(1)
                    pbar.close()
    return score_matrices

def load_agents_population(filepaths, args):
    agents = []
    for fn in filepaths:
        agent = load_agent(fn)
        print(f'loaded agent {agent.name}')
        agents.append(agent)
    return agents

if __name__ == "__main__":
    args = get_arguments()
    base_dir = args.base_dir / 'agent_models'
    main_agents_fns = ["fcp/last_hope/agents_dir/agent_0", "bcp/last_hope/agents_dir/agent_0", "selfplay/best/agents_dir/agent_0", "best_hrl", ]
    main_agents_fns = [base_dir / fn for fn in main_agents_fns]

    main_agents = load_agents_population(main_agents_fns, args)
    bc, human_proxy = train_agents.get_bc_and_human_proxy(args)

    tms = [human_proxy, DummyAgent('random'), *main_agents]

    score_matrices = eval_agents_with_various_teammates(main_agents, tms)
    for layout in args.layout_names:
        print(f"For env: {layout}")
        for i, agent in enumerate(main_agents):
            print(f"Agent {agent.name} had an average score of {np.mean(score_matrices[layout][i])}")

