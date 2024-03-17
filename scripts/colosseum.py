from oai_agents.agents.agent_utils import load_agent, DummyAgent
from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.arguments import get_arguments
from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv

import numpy as np
import pandas as pd
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

def eval_agents_with_various_teammates(agents_to_evaluate, teammates, use_self_as_tm=False, training_tms=None,
                                       agent_det=False, tm_det=False):
    eval_envs_kwargs = {'is_eval_env': True, 'args': args, 'horizon': 400, 'ret_completed_subtasks': True}
    # args.layout_names = ['counter_circuit_o_1order']
    args.layout_names = ['asymmetric_advantages', 'coordination_ring', 'counter_circuit_o_1order', 'cramped_room', 'forced_coordination']
    args.layout_names = [ln + '_mod' for ln in args.layout_names]
    num_teammates = len(teammates) + 1 if training_tms else len(teammates)
    if use_self_as_tm:
        num_teammates = 1
    score_matrices = np.zeros((len(agents_to_evaluate), len(args.layout_names), num_teammates))

    tot_rounds = len(args.layout_names) * len(agents_to_evaluate) * num_teammates
    with tqdm(total=tot_rounds) as pbar:
        for i, p1 in enumerate(agents_to_evaluate):
            eval_envs_kwargs['ret_completed_subtasks'] = ('haha' in p1.name)
            eval_envs = [OvercookedGymEnv(**{'env_index': i, **eval_envs_kwargs}) for i in range(len(args.layout_names))]
            for j, eval_env in enumerate(eval_envs):
                eval_env.deterministic = tm_det
                eval_env.encoding_fn = p1.encoding_fn
                tms = teammates + training_tms[i] if training_tms is not None else teammates
                if use_self_as_tm and len(teammates) == len(agents_to_evaluate):
                    tms = [teammates[i]]
                for k, p2 in enumerate(tms):
                    # if type(p2) != dict and p1.name == p2.name:
                    #     continue
                    p2 = p2[eval_env.layout_name][-1] if type(p2) == dict else p2
                    print(f'Now evaluating {p1.name} with teammates {p2.name}')
                    for p_idx in [0, 1]:
                        p1.set_encoding_params(p_idx, eval_env.args.horizon, eval_env, is_haha=('haha' in p1.name), tune_subtasks="tuned" in p1.name)
                        p2.set_encoding_params(1 - p_idx, eval_env.args.horizon, eval_env, is_haha=('haha' in p2.name), tune_subtasks="tuned" in p2.name)
                        eval_env.set_teammate(p2)
                        eval_env.set_reset_p_idx(p_idx)
                        num_trials = 5
                        for _ in range(num_trials):
                            reward = run_game(eval_env, p1, deterministic=agent_det)
                            # mean_reward, std_reward = evaluate_policy(p1, eval_env, n_eval_episodes=10,
                            #                                           deterministic=True, warn=False, render=False)
                            score_matrices[i][j][k] += reward / num_trials
                    pbar.update(1)
                    # div by 2 for both indices
                    score_matrices[i][j][k] /= 2
    pbar.close()

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
    base_dir = args.base_dir / 'agent_models_ICML'# / 'ent_aamas24'
    main_agents_names = ['HAHA_fcp']#,'HAHA_fcp',  'HAHA_bcp', 'bcp']#, "HAHA_fcp_fcp"]# "FCP", "BCP"]#, "HAHA+tuned", "HAHA_new36+tuned"]#"HAHA+tuned", "HAHA", "FCP"] #"FCP", "fcp/last_hope/agents_dir/agent_0", "bcp/last_hope/agents_dir/agent_0", "selfplay/best/agents_dir/agent_0"]
    seeds = ['61', '102', '219', '1811', '4573']

    bc, human_proxy = get_bc_and_human_proxy(args)
    use_self_as_tm = True
    if use_self_as_tm:
        num_tms = 1
    else:
        tms = [*load_agents_population([base_dir / "sp_test"], args), human_proxy, DummyAgent('random')]
        num_tms = len(tms)

    main_agent_scores = {k: np.zeros((len(args.layout_names) + 1, num_tms + 1)) for k in main_agents_names} # +1s for average
    main_agent_stds = {k: np.zeros((len(args.layout_names) + 1, num_tms + 1)) for k in main_agents_names} # +1s for average

    for main_agent in main_agents_names:
        agent_fns = [base_dir / f'{main_agent}_{seed}' for seed in seeds]

        main_agents = load_agents_population(agent_fns, args)
        if use_self_as_tm:
            tms = [*load_agents_population(agent_fns, args)]

        score_matrices = eval_agents_with_various_teammates(main_agents, tms, use_self_as_tm=use_self_as_tm)

        agent_means = np.mean(score_matrices, axis=(1, 2))
        agent_mean = np.mean(agent_means)
        agent_std = np.std(agent_means)

        # Average across seeds
        main_agent_scores[main_agent][:-1, :-1] = np.mean(score_matrices, axis=0)
        main_agent_stds[main_agent][:-1, :-1] = np.std(score_matrices, axis=0)

        # Average across layouts
        layout_avgs = np.mean(score_matrices, axis=1)
        main_agent_scores[main_agent][-1, :-1] = np.mean(layout_avgs, axis=0)
        main_agent_stds[main_agent][-1, :-1] = np.std(layout_avgs, axis=0)

        # Average across all teammates
        tm_avgs = np.mean(score_matrices, axis=2)
        main_agent_scores[main_agent][:-1, -1] = np.mean(tm_avgs, axis=0)
        main_agent_stds[main_agent][:-1, -1] = np.std(tm_avgs, axis=0)

        # Average acroos tms and layouts
        main_agent_scores[main_agent][-1, -1] = agent_mean
        main_agent_stds[main_agent][-1, -1] = agent_std

        column_names = ['agent_name'] + args.layout_names + ['average']
        with open(f'results_{main_agent}_mod.csv', 'w') as f:
            f.write(','.join(column_names))
            for i in range(num_tms):
                tm_name = tms[i].name if not isinstance(tms[i], dict) else 'human_proxy'
                f.write(f'\n{tm_name},')
                for mean, std in zip(main_agent_scores[main_agent][:, i], main_agent_stds[main_agent][:, i]):
                    f.write(f'{mean:.3f}+-({std:.3f}),')
            f.write('\naverage,')
            for mean, std in zip(main_agent_scores[main_agent][:, -1], main_agent_stds[main_agent][:, -1]):
                f.write(f'{mean:.3f}+-({std:.3f}),')




