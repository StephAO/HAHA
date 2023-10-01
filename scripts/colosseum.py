from oai_agents.agents.agent_utils import load_agent, DummyAgent
from oai_agents.common.arguments import get_arguments
from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv

import numpy as np
from pathlib import Path
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import tqdm
from scripts.train_agents import get_bc_and_human_proxy

def run_game(env, agent, deterministic=False):
    done = False
    cum_reward = 0
    obs = env.reset()
    while not done:
        action = agent.predict(obs, deterministic=deterministic)[0]
        obs, reward, done, info = env.step(action)
        cum_reward += reward
    return cum_reward

def eval_agents_with_various_teammates(agents_to_evaluate, teammates, deterministic=False):
    eval_envs_kwargs = {'is_eval_env': True, 'args': args, 'horizon': 400, 'ret_completed_subtasks': True}
    args.layout_names = ['counter_circuit_o_1order', 'asymmetric_advantages', 'cramped_room', 'forced_coordination', 'coordination_ring']
    # args.layout_names = [ln + '_mod' for ln in args.layout_names]
    score_matrices = {ln: np.zeros((len(agents_to_evaluate), len(teammates))) for ln in args.layout_names}

    tot_rounds = len(args.layout_names) * len(agents_to_evaluate) * len(teammates)
    with tqdm(total=tot_rounds) as pbar:
        for i, p1 in enumerate(agents_to_evaluate):
            eval_envs_kwargs['ret_completed_subtasks'] = ('haha' in p1.name)
            eval_envs = [OvercookedGymEnv(**{'env_index': i, **eval_envs_kwargs}) for i in range(len(args.layout_names))]
            for eval_env in eval_envs:
                eval_env.deterministic = deterministic
                for j, p2 in enumerate(teammates):
                    # if type(p2) != dict and p1.name == p2.name:
                    #     continue
                    p2 = p2[eval_env.layout_name][0] if type(p2) == dict else p2
                    print(f'Now evaluating {p1.name} with teammates {p2.name}')
                    for p_idx in [0, 1]:
                        p1.set_encoding_params(p_idx, eval_env.args.horizon, eval_env, is_haha=('haha' in p1.name), tune_subtasks="tuned" in p1.name)
                        p2.set_encoding_params(1 - p_idx, eval_env.args.horizon, eval_env, is_haha=('haha' in p2.name), tune_subtasks="tuned" in p2.name)
                        eval_env.set_teammate(p2)
                        eval_env.set_reset_p_idx(p_idx)
                        for _ in range(5):
                            reward = run_game(eval_env, p1, deterministic=deterministic)
                            # mean_reward, std_reward = evaluate_policy(p1, eval_env, n_eval_episodes=10,
                            #                                           deterministic=True, warn=False, render=False)
                            score_matrices[eval_env.layout_name][i][j] += reward / 5
                    pbar.update(1)
                    score_matrices[eval_env.layout_name][i][j] /= 2
    pbar.close()

    print(score_matrices)

    print([p.name for p in agents_to_evaluate])
    print([('bc' if type(p) == dict else p.name) for p in teammates])
    print(score_matrices)

    return score_matrices

def load_agents_population(filepaths, args):
    agents = []
    for fn in filepaths:
        tuned = False
        if "+tuned" in str(fn):
            fn = Path(str(fn).replace("+tuned", ""))
            tuned = True
        agent = load_agent(fn, args=args)
        if tuned:
            agent.name = agent.name + "+tuned"
        print(f'loaded agent {agent.name}')
        agents.append(agent)
    return agents

if __name__ == "__main__":
    args = get_arguments()
    base_dir = args.base_dir / 'agent_models'
    main_agents_fns = ["HAHA_fcp_fcp+tuned"]#,"HAHA_fcp_fcp",  "BCP", "FCP"]#"HAHA_bcp_bcp+tuned"]#, "HAHA_fcp_fcp"]# "FCP", "BCP"]#, "HAHA+tuned", "HAHA_new36+tuned"]#"HAHA+tuned", "HAHA", "FCP"] #"FCP", "fcp/last_hope/agents_dir/agent_0", "bcp/last_hope/agents_dir/agent_0", "selfplay/best/agents_dir/agent_0"]
    main_agents_fns = [base_dir / fn for fn in main_agents_fns]

    main_agents = load_agents_population(main_agents_fns, args) #+[DummyAgent('random')] +
    bc, human_proxy = get_bc_and_human_proxy(args)

    # Load main agents again to avoid issues with hrl object
    # tms = [*load_agents_population(["SP"], args), DummyAgent('random'), human_proxy] # *load_agents_population(main_agents_fns, args),
    # tms = get_fcp_population(args, training_steps)

    # tm_fns = ["ck_0", "ck_4", "ck_8", "ck_12", "ck_16", "best"]
    # tm_fns = [base_dir / '2l_hd128_s1997' / fn / 'agents_dir' / 'agent_0' for fn in tm_fns]
    # tms = [bc, human_proxy]
    tms = [*load_agents_population([base_dir / "SP"], args), human_proxy, DummyAgent('random')] # [*load_agents_population(tm_fns, args)]

    score_matrices = eval_agents_with_various_teammates(main_agents, tms)
    for layout in args.layout_names:
        print(f"For env: {layout}")
        for i, agent in enumerate(main_agents):
            print(f"Agent {agent.name} had an average score of {np.mean(score_matrices[layout][i])}")


