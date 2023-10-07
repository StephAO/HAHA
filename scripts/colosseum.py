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

def eval_agents_with_various_teammates(agents_to_evaluate, teammates, training_tms=None, agent_det=False, tm_det=False):
    eval_envs_kwargs = {'is_eval_env': True, 'args': args, 'horizon': 400, 'ret_completed_subtasks': True}
    args.layout_names = ['asymmetric_advantages', 'coordination_ring', 'counter_circuit_o_1order', 'cramped_room', 'forced_coordination']
    # args.layout_names = [ln + '_mod' for ln in args.layout_names]
    num_teammates = len(teammates) + 1 if training_tms else len(teammates)
    score_matrices = {ln: np.zeros((len(agents_to_evaluate), num_teammates)) for ln in args.layout_names}

    tot_rounds = len(args.layout_names) * len(agents_to_evaluate) * num_teammates
    with tqdm(total=tot_rounds) as pbar:
        for i, p1 in enumerate(agents_to_evaluate):
            eval_envs_kwargs['ret_completed_subtasks'] = ('haha' in p1.name)
            eval_envs = [OvercookedGymEnv(**{'env_index': i, **eval_envs_kwargs}) for i in range(len(args.layout_names))]
            for eval_env in eval_envs:
                eval_env.deterministic = tm_det
                eval_env.encoding_fn = p1.encoding_fn
                eval_env.new_agent = p1.new_agent
                tms = teammates + training_tms[i] if training_tms is not None else teammates
                for j, p2 in enumerate(tms):
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
                            score_matrices[eval_env.layout_name][i][j] += reward / num_trials
                    pbar.update(1)
                    # div by 2 for both indices
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
    base_dir = args.base_dir / 'agent_models / ent_aamas24'
    main_agents_fns = ['fcp', 'fcp_det', "bcp", "bcp_det", "sp", "sp_det"]#, "HAHA_fcp_fcp"]# "FCP", "BCP"]#, "HAHA+tuned", "HAHA_new36+tuned"]#"HAHA+tuned", "HAHA", "FCP"] #"FCP", "fcp/last_hope/agents_dir/agent_0", "bcp/last_hope/agents_dir/agent_0", "selfplay/best/agents_dir/agent_0"]
    main_agents_fns = [base_dir / fn for fn in main_agents_fns]

    main_agents = load_agents_population(main_agents_fns, args) #+[DummyAgent('random')] +
    bc, human_proxy = get_bc_and_human_proxy(args)

    inc_training_tm = True
    print(main_agents_fns)

    if inc_training_tm:
        training_tms = []
        for fn in main_agents_fns:
            if 'bcp' in str(fn):
                teammates.append([bc])
            elif 'fcp' in str(fn):
                fcp_pop = {}
                for layout_name in args.layout_names:
                    fcp_pop[layout_name] = RLAgentTrainer.load_agents(args, name=f'fcp_pop_{layout_name}', tag='aamas24')[:3]
                training_tms.append([fcp_pop])
            else:
                load_agents_population([fn], args)
    else:
        training_tms = None
    # Load main agents again to avoid issues with hrl object
    # tms = [*load_agents_population(["SP"], args), DummyAgent('random'), human_proxy] # *load_agents_population(main_agents_fns, args),
    # tms = get_fcp_population(args, training_steps)

    # tm_fns = ["ck_0", "ck_4", "ck_8", "ck_12", "ck_16", "best"]
    # tm_fns = [base_dir / '2l_hd128_s1997' / fn / 'agents_dir' / 'agent_0' for fn in tm_fns]
    # tms = [bc, human_proxy]
    tms = [*load_agents_population([base_dir / "SP"], args), human_proxy]#, DummyAgent('random')] # [*load_agents_population(tm_fns, args)]
    # tms = load_agents_population(main_agents_fns, args)

    """
    I want a csv for each teammate and for overall across all teammates x 2 for deterministic and stochastic tms
    In each csv, I want a score with teammate on each layout and on average
    With 3 tms, this is 8 csv files
    """


    column_names = ['agent_name'] + args.layout_names + ['average']
    for tm_det in [True, False]:
        score_matrices = {}
        for agent_det in [True, False]:
            score_matrices[agent_det] = eval_agents_with_various_teammates(main_agents, tms, training_tms=training_tms, agent_det=agent_det, tm_det=tm_det)
            # print(f"For layouts: {args.layout_names}, agents{main_agents_fns}")
            # for layout in args.layout_names:
            #     for i, agent in enumerate(main_agents):
            #         print(np.mean(score_matrices[layout][i][:3]),end='' if (i == len(main_agents) - 1) else ',')
            #     # print(f"Agent {agent.name} had an average score of {np.mean(score_matrices[layout][i][:3])} with other agents and {score_matrices[layout][i][-1]} with itself")
            #     print()
            # for layout in args.layout_names:
            #     for i, agent in enumerate(main_agents):
            #         print(np.mean(score_matrices[layout][i][-1]),end='' if (i == len(main_agents) - 1) else ',')
            #     print()
        tm_names = [('human_proxy' if isinstance(a, dict) else a.name) for a in tms]
        if inc_training_tm:
            tm_names += ['training_partner']

        for i, tm_name in enumerate(tm_names):
            # SEPARATE CSVS
            data = []
            # ONE ROW IS 1 AGENT
            for j, agent_name in enumerate([a.name for a in main_agents]):
                # EACH AGENT NAME is two rows for det/sto
                row_data_det = [agent_name + '_d']
                row_data_sto = [agent_name + '_s']
                avg_det, avg_sto = [], []
                # EACH COL is a separate layout + average
                for layout_name in args.layout_names:
                    row_data_det.append(score_matrices[True][layout_name][j][i])
                    row_data_sto.append(score_matrices[False][layout_name][j][i])
                    if tm_name == 'training_partner':
                        continue
                    avg_det.append(score_matrices[True][layout_name][j][i])
                    avg_sto.append(score_matrices[False][layout_name][j][i])

                row_data_det.append(np.mean(avg_det))
                row_data_sto.append(np.mean(avg_sto))

                data.extend([row_data_det, row_data_sto])

            df = pd.DataFrame(data=data, columns=column_names)
            df.to_csv(f'data/ua_eval_{tm_name}_{tm_det}.csv')

        # AVERAGE OVER TMS CSV
        data = []
        # ONE ROW IS 1 AGENT
        for j, agent_name in enumerate([a.name for a in main_agents]):
            # EACH AGENT NAME is two rows for det/sto
            row_data_det = [agent_name + '_d']
            row_data_sto = [agent_name + '_s']
            avg_det, avg_sto = [], []
            # EACH COL is a separate layout + average
            for layout_name in args.layout_names:
                # :-1 to remove training teammate score
                row_data_det.append(np.mean(score_matrices[True][layout_name][j][:-1]))
                row_data_sto.append(np.mean(score_matrices[False][layout_name][j][:-1]))
                avg_det.append(np.mean(score_matrices[True][layout_name][j][:-1]))
                avg_sto.append(np.mean(score_matrices[False][layout_name][j][:-1]))

            row_data_det.append(np.mean(avg_det))
            row_data_sto.append(np.mean(avg_sto))

            data.extend([row_data_det, row_data_sto])

        df = pd.DataFrame(data=data, columns=column_names)
        df.to_csv(f'data/ua_eval_avg_over_tms_{tm_det}.csv')

