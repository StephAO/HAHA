from oai_agents.agents.il import BehavioralCloningTrainer
from oai_agents.agents.rl import RLAgentTrainer, SB3Wrapper, VEC_ENV_CLS
from oai_agents.agents.hrl import RLManagerTrainer, HierarchicalRL, DummyAgent
from oai_agents.common.arguments import get_arguments
from oai_agents.agents.agent_utils import load_agent
from oai_agents.gym_environments.worker_env import OvercookedSubtaskGymEnv

from overcooked_ai_py.mdp.overcooked_mdp import Action

from copy import deepcopy
from gym import Env, spaces
import numpy as np
from pathlib import Path
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env


def calculate_agent_pairing_score_matrix(agents, args):
    eval_envs_kwargs = {'is_eval_env': True, 'args': args}
    eval_envs = [OvercookedGymEnv(**{'env_index': i, **eval_envs_kwargs}) for i in range(len(args.layout_names))]
    score_matrix = np.zeros((len(agents), len(agents)))
    for eval_env in eval_envs:
        for i, p1 in enumerate(agents):
            for j, p2 in enumerate(agents):
                env.set_teammate(p2)
                mean_reward, std_reward = evaluate_policy(p1, eval_env, n_eval_episodes=10,
                                                          deterministic=False, warn=False, render=False)
                score_matrix[i][j] = mean_reward
    return score_matrix


def create_pop_from_agents(args):
    # WARNING: THIS IS JUST TEMPLATE CODE. This function requires hand figuring out each ck to use for the mid ck
    base_path = args.base_dir / 'agent_models'
    mid_indices = {'forced_coordination': [6, 6, 5, 4, 6, 4, 4, 4], 'counter_circuit_o_1order': [3, 4, 4, 4, 5, 4, 3, 3], 'asymmetric_advantages': [3, 1, 2, 2, 2, 2, 1, 2],
                   'cramped_room': [2, 2, 1, 1, 2, 2, 1, 1], 'coordination_ring': [3, 2, 2, 2, 2, 6, 3, 2]}
    agent_names = ['fcp_spfs_hd64_seed105', 'fcp_spfs_hd64_seed2907', 'fcp_spfs_hd256_seed105', 'fcp_spfs_hd256_seed2907',
                   'fcp_sphd64_seed105', 'fcp_sphd64_seed2907', 'fcp_sphd256_seed105', 'fcp_sphd256_seed2907']
    for layout_name in args.layout_names:
        pop_agents = []
        for agent_name, mi in zip(agent_names, mid_indices[layout_name]):
            worst = SB3Wrapper.load(base_path / agent_name / f'ck_0' / 'agents_dir' / 'agent_0', args)
            mid = SB3Wrapper.load(base_path / agent_name / f'ck_{mi}' / 'agents_dir' / 'agent_0', args)
            best = SB3Wrapper.load(base_path / agent_name / 'best' / 'agents_dir' / 'agent_0', args)
            pop_agents += [worst, mid, best]

        mat = RLAgentTrainer([], args, selfplay=True, name=f'fcp_pop_{layout_name}')
        mat.agents = pop_agents
        mat.save_agents()


def combine_populations(args):  # , pop_names, new_name):
    pop_names = ['sp_fs_hd32_seed105', 'sp_fs_hd32_seed1997', 'sp_fs_hd256_seed105', 'sp_fs_hd256_seed1997',
                 'sp_hd32_seed105', 'sp_hd32_seed1997', 'sp_hd256_seed105', 'sp_hd256_seed1997']
    full_pop = {k: [] for k in args.layout_names}
    for layout_name in args.layout_names:
        for pop_name in pop_names:
            full_pop[layout_name] += RLAgentTrainer.load_agents(args, name=f'fcp_pop_{layout_name}_{pop_name}')
        # verify = input('WARNING: You are about to overwrite fcp_pop. Press Y to continue, or anything else to cancel.')
        # if verify.lower() == 'y':
        mat = RLAgentTrainer([], args, selfplay=True, name=f'fcp_pop_{layout_name}', num_agents=0)
        mat.set_agents(full_pop[layout_name])
        print(len(mat.agents))
        mat.save_agents()


### EVALUATION AGENTS ###
def get_eval_teammates(args):
    sp = get_selfplay_agent(args, training_steps=1e7)
    bcs, human_proxies = get_bc_and_human_proxy(args)
    random_agent = DummyAgent('random')
    eval_tms = {}
    for ln in args.layout_names:
        eval_tms[ln] = [bcs[ln][0], sp[0], random_agent]
    return eval_tms


### BASELINES ###

# SP
def get_selfplay_agent(args, training_steps=1e7, tag=None):
    name = 'sp_det'
    try:
        tag = tag or 'best'
        agents = RLAgentTrainer.load_agents(args, name=name, tag=tag)
    except FileNotFoundError as e:
        print(f'Could not find saved selfplay agent, creating them from scratch...\nFull Error: {e}')
        selfplay_trainer = RLAgentTrainer([], args, selfplay=True, name=name, seed=678, use_frame_stack=False,
                                          use_lstm=False, use_cnn=False, deterministic=True)
        selfplay_trainer.train_agents(train_timesteps=training_steps)
        agents = selfplay_trainer.get_agents()
    return agents


# BC and Human Proxy
def get_bc_and_human_proxy(args, epochs=300):
    bcs, human_proxies = {}, {}
    # This is required because loading agents will overwrite args.layout_names
    all_layouts = deepcopy(args.layout_names)
    for layout_name in all_layouts:
        try:
            bc, human_proxy = BehavioralCloningTrainer.load_bc_and_human_proxy(args, name=f'bc_{layout_name}')
        except FileNotFoundError as e:
            print(f'Could not find saved BC and human proxy, creating them from scratch...\nFull Error: {e}')
            bct = BehavioralCloningTrainer(args.dataset, args, name=f'bc_{layout_name}', layout_names=[layout_name])
            bct.train_agents(epochs=epochs)
            bc, human_proxy = bct.get_agents()
        bcs[layout_name] = [bc]
        human_proxies[layout_name] = [human_proxy]

    args.layout_names = all_layouts
    return bcs, human_proxies


# BCP
def get_behavioral_cloning_play_agent(args, training_steps=1e7):
    name = 'bcp_det'
    try:
        bcp = RLAgentTrainer.load_agents(args, name=name)
    except FileNotFoundError as e:
        print(f'Could not find saved BCP, creating them from scratch...\nFull Error: {e}')
        teammates, _ = get_bc_and_human_proxy(args)
        self_play_trainer = RLAgentTrainer(teammates, args, name=name, deterministic=True)
        self_play_trainer.train_agents(train_timesteps=training_steps)
        bcp = self_play_trainer.get_agents()
    return bcp

def get_test_fcp_pop(args):
    fcp_pop = {layout_name: [] for layout_name in args.layout_names}
    agents = []
    num_layers = 2

    ck_rate = training_steps // 10

    print(f'Starting training for: {name}')
    rlat = RLAgentTrainer([], args, selfplay=True, name='fcp_test_pop', hidden_dim=128, use_frame_stack=False,
                          fcp_ck_rate=ck_rate, seed=seed, num_layers=num_layers)
    rlat.train_agents(train_timesteps=2e7)

    for layout_name in args.layout_names:
        agents = rlat.get_fcp_agents(layout_name)
        fcp_pop[layout_name] += agents

    for layout_name in args.layout_names:
        pop = RLAgentTrainer([], args, selfplay=True, name=f'fcp_test_pop_{layout_name}')
        pop.agents = fcp_pop[layout_name]
        pop.save_agents(tag='aamas24')


# FCP
def get_fcp_population(args, training_steps=2e7):
    try:
        fcp_pop = {}
        for layout_name in args.layout_names:
            fcp_pop[layout_name] = RLAgentTrainer.load_agents(args, name=f'fcp_pop_{layout_name}', tag='aamas24')
            print(f'Loaded fcp_pop with {len(fcp_pop[layout_name])} agents.')
    except FileNotFoundError as e:
        print(f'Could not find saved FCP population, creating them from scratch...\nFull Error: {e}')
        fcp_pop = {layout_name: [] for layout_name in args.layout_names}
        agents = []
        num_layers = 2
        for use_fs in [True]:#[False, True]:
            for seed, h_dim in [(2907, 64), (2907, 256)]:  #(105, 64), (105, 256),# [8,16], [32, 64], [128, 256], [512, 1024]
                ck_rate = training_steps // 10
                # name = f'cnn_{num_layers}l_' if use_cnn else f'eval_{num_layers}l_'
                name = 'fcp_sp'
                # name += 'pc_' if use_policy_clone else ''
                # name += 'tpl_' if taper_layers else ''
                name += f'fs_' if use_fs else ''
                name += f'hd{h_dim}_'
                name += f'seed{seed}'
                print(f'Starting training for: {name}')
                rlat = RLAgentTrainer([], args, selfplay=True, name=name, hidden_dim=h_dim, use_frame_stack=use_fs,
                                      fcp_ck_rate=ck_rate, seed=seed, num_layers=num_layers)
                rlat.train_agents(train_timesteps=training_steps)

                for layout_name in args.layout_names:
                    agents = rlat.get_fcp_agents(layout_name)
                    fcp_pop[layout_name] += agents

        for layout_name in args.layout_names:
            pop = RLAgentTrainer([], args, selfplay=True, name=f'fcp_pop_{layout_name}')
            pop.agents = fcp_pop[layout_name]
            pop.save_agents(tag='aamas24')
    return fcp_pop


def get_fcp_agent(args, training_steps=1e7):
    name = 'fcp_det'
    teammates = get_fcp_population(args, training_steps)
    fcp_trainer = RLAgentTrainer(teammates, args, name=name, use_subtask_counts=False, use_policy_clone=False,
                                 seed=2602, deterministic=True)
    fcp_trainer.train_agents(train_timesteps=training_steps)
    return fcp_trainer.get_agents()[0]


def get_hrl_worker(args, teammate_type='fcp', training_steps=1e7):
    name = f'worker_{teammate_type}'
    try:
        worker = RLAgentTrainer.load_agents(args, name=name, tag='best')[0]
    except FileNotFoundError as e:
        print(f'Could not find saved worker agent, creating them from scratch...\nFull Error: {e}')
        # eval_tms = get_eval_teammates(args)

        if teammate_type == 'bcp':
            teammates, _ = get_bc_and_human_proxy(args)
        elif teammate_type == 'fcp':
            teammates = get_fcp_population(args, training_steps)

        #teammates = get_fcp_population(args, 1e7)
        # Create subtask worker
        env_kwargs = {'stack_frames': False, 'full_init': False, 'args': args}
        env = make_vec_env(OvercookedSubtaskGymEnv, n_envs=args.n_envs, env_kwargs=env_kwargs,
                           vec_env_cls=VEC_ENV_CLS)
        env_kwargs['full_init'] = True
        eval_envs = [OvercookedSubtaskGymEnv(**{'env_index': n, 'is_eval_env': True, **env_kwargs})
                     for n in range(len(args.layout_names))]
        worker_trainer = RLAgentTrainer(teammates, args, name=name, env=env, eval_envs=eval_envs,
                                        use_subtask_eval=True)
        worker_trainer.train_agents(train_timesteps=training_steps)
        worker = worker_trainer.get_agents()[0]

    return worker


def get_hrl_agent(args, teammate_types=('bcp', 'bcp'), training_steps=1e7, num_iterations=10):
    """
    teammates args is a tuple of length 2, where each value can be either bcp of fcp. The first value indicates the
    teammates to use for the worker, the second the teammates to use for the manager
    """
    name = f'HAHA_{teammate_types}_fulltasks'
    # Get worker
    worker = get_hrl_worker(args, teammate_types[0])#, training_steps=15e6)
    # Get teammates
    if teammate_types[1] == 'bcp':
        teammates, _ = get_bc_and_human_proxy(args)
    elif teammate_types[1] == 'fcp':
        teammates = get_fcp_population(args, training_steps)

    # Create manager and manager env
    manager_trainer = RLManagerTrainer(worker, teammates, args, use_subtask_counts=False,
                                       name=f'manager_{teammate_types}', inc_sp=False, use_policy_clone=False, seed=2602)

    # Iteratively train worker and manager
    # import cProfile
    # from timeit import timeit
    # print(timeit(lambda: worker_trainer.train_agents(training_steps_per_agent_per_iter), number=1))
    # cProfile.runctx('worker_trainer.train_agents(training_steps_per_agent_per_iter)', None, locals(), sort='cumtime')
    manager_trainer.train_agents(1e8)

    hrl = HierarchicalRL(worker, manager_trainer.learning_agent, args, name=name)
    hrl.save(Path(Path(args.base_dir / 'agent_models' / hrl.name / args.exp_name)))
    return hrl


def get_all_agents(args, training_steps=1e7, agents_to_train='all'):
    agents = {}
    fcp_pop = None
    if agents_to_train == 'all' or 'selfplay' in agents_to_train:
        agents['selfplay'] = get_selfplay_agent(args, training_steps=5e6)
    if agents_to_train == 'all' or 'bcp' in agents_to_train:
        agents['bcp'] = get_behavioral_cloning_play_agent(args, training_steps=5e6)
    if agents_to_train == 'all' or 'fcp' in agents_to_train:
        agents['fcp'] = get_fcp_agent(args, training_steps)
    if agents_to_train == 'all' or 'hrl' in agents_to_train:
        agents['hrl'] = get_hrl_agent(args, training_steps)


if __name__ == '__main__':
    args = get_arguments()
    #get_selfplay_agent(args, training_steps=2e8)
    # print('GOT SP', flush=True)
    # get_bc_and_human_proxy(args, epochs=2)
    # print('GOT BC&HP', flush=True)
    #get_behavioral_cloning_play_agent(args, training_steps=2e8)
    # print('GOT BCP', flush=True)
    #get_fcp_agent(args, training_steps=2e8)
    get_test_fcp_pop(args)
    # print('GOT FCP', flush=True)
    #get_hrl_worker(args, training_steps=1e8)
    #print('GOT WORK', flush=True)
    #get_hrl_agent(args, training_steps=1e8)
    #print('GOT HAHA', flush=True)

    # get_bc_and_human_proxy(args)
    # get_behavioral_cloning_play_agent(args, training_steps=1e8)

    # get_fcp_population(args, 2e7)
    #get_fcp_agent(args, training_steps=1e8)
    # get_hrl_worker(args)
    # get_hrl_agent(args, 5e7)

    # create_test_population(args, 1e3)
    # create_pop_from_agents(args)

    # combine_populations(args)

    # worker = load_agent(args.base_dir / 'agent_models' / 'worker_bcp/best/agents_dir/agent_0', args)
    # manager = load_agent(args.base_dir / 'agent_models' / 'manager_bcp/best/agents_dir/agent_0', args)
    #
    # hrl = HierarchicalRL(worker, manager, args, name='HAHA_bcp_bcp')
    # hrl.save(Path(Path(args.base_dir / 'agent_models' / hrl.name / args.exp_name)))
