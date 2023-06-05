from oai_agents.agents.il import BehavioralCloningTrainer
from oai_agents.agents.rl import RLAgentTrainer, SB3Wrapper
from oai_agents.agents.hrl import MultiAgentSubtaskWorker, RLManagerTrainer, HierarchicalRL, DummyAgent
from oai_agents.common.arguments import get_arguments

from overcooked_ai_py.mdp.overcooked_mdp import Action

from copy import deepcopy
from gym import Env, spaces
import numpy as np
from pathlib import Path
from stable_baselines3.common.evaluation import evaluate_policy


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
    mid_indices = {'forced_coordination': [2, 9], 'counter_circuit_o_1order': [0, 7], 'asymmetric_advantages': [0, 1], 'cramped_room': [0, 2], 'coordination_ring': [0, 3]}
    for layout_name in args.layout_names:
        pop_agents = []
        for agent_name in ['2l_hd32_seed19950226', '2l_hd32_seed20220501', '2l_hd128_seed1997', '2l_hd128_seed219',
                            '2l_tpl_hd32_seed1004219', '2l_tpl_hd32_seed20220501', '2l_tpl_hd128_seed219', '2l_tpl_hd128_seed2191004']:
            best = SB3Wrapper.load(base_path / agent_name / 'best' / 'agents_dir' / 'agent_0', args)
            pop_agents += [best]
            for i in mid_indices[layout_name]:
                next_agent = SB3Wrapper.load(base_path / agent_name / f'ck_{i}' / 'agents_dir' / 'agent_0', args)
                pop_agents += [next_agent]

        mat = RLAgentTrainer([], args, selfplay=True, name=f'fcp_pop_{layout_name}')
        mat.set_agents(pop_agents)
        mat.save_agents()

def combine_populations(args): #, pop_names, new_name):
    pop_names = ['sp_fs_hd32_seed105', 'sp_fs_hd32_seed1997', 'sp_fs_hd256_seed105', 'sp_fs_hd256_seed1997',
                 'sp_hd32_seed105', 'sp_hd32_seed1997', 'sp_hd256_seed105', 'sp_hd256_seed1997']
    full_pop = {k: [] for k in args.layout_names}
    for layout_name in args.layout_names:
        for pop_name in pop_names:
            mat = RLAgentTrainer([], args, selfplay=True, name=f'fcp_pop_{layout_name}_{pop_name}', num_agents=0)
            full_pop[layout_name] += mat.load_agents()
        # verify = input('WARNING: You are about to overwrite fcp_pop. Press Y to continue, or anything else to cancel.')
        # if verify.lower() == 'y':
        mat = RLAgentTrainer([], args, selfplay=True, name=f'fcp_pop_{layout_name}', num_agents=0)
        mat.set_agents(full_pop[layout_name])
        print(len(mat.agents))
        mat.save_agents()

### EVALUATION AGENTS ###
def get_eval_teammates(args):
    sp = get_selfplay_agent(args, training_steps=1e7, tag='ijcai_tm')
    bcs, human_proxies = get_bc_and_human_proxy(args)
    random_agent = DummyAgent('random')
    eval_tms = {}
    for ln in args.layout_names:
        eval_tms[ln] = [bcs[ln][0], sp[0], random_agent]
    return eval_tms

### BASELINES ###

# SP
def get_selfplay_agent(args, training_steps=1e7, tag=None):
    self_play_trainer = RLAgentTrainer([], args, selfplay=True, name='selfplay', seed=499, use_cnn=False, use_frame_stack=False)
    try:
        tag = tag or 'testing3'
        self_play_trainer.load_agents(tag=tag)
    except FileNotFoundError as e:
        print(f'Could not find saved selfplay agent, creating them from scratch...\nFull Error: {e}')
        self_play_trainer.train_agents(total_timesteps=training_steps)
    return self_play_trainer.get_agents()

# BC and Human Proxy
def get_bc_and_human_proxy(args):
    bcs, human_proxies = {}, {}
    # This is required because loading agents will overwrite args.layout_names
    all_layouts = deepcopy(args.layout_names)
    for layout_name in all_layouts:
        bct = BehavioralCloningTrainer(args.dataset, args, name=f'bc_{layout_name}', layout_names=[layout_name])
        try:
            bct.load_bc_and_human_proxy(tag='ijcai')
        except FileNotFoundError as e:
            print(f'Could not find saved BC and human proxy, creating them from scratch...\nFull Error: {e}')
            bct.train_agents(epochs=500)
        bc, human_proxy = bct.get_agents()
        bcs[layout_name] = [bc]
        human_proxies[layout_name] = [human_proxy]

    args.layout_names = all_layouts
    return bcs, human_proxies

# BCP
def get_behavioral_cloning_play_agent(args, training_steps=1e7):
    bcs, human_proxies = get_bc_and_human_proxy(args)
    teammates = bcs
    self_play_trainer = RLAgentTrainer(teammates, args, name='bcp')
    try:
        bcp = self_play_trainer.load_agents(tag='ijcai')
    except FileNotFoundError as e:
        print(f'Could not find saved BCP, creating them from scratch...\nFull Error: {e}')
        self_play_trainer.train_agents(total_timesteps=training_steps)
        bcp = self_play_trainer.get_agents()
    return bcp

# FCP
def get_fcp_population(args, training_steps=2e7):
    try:
        fcp_pop = {}
        for layout_name in args.layout_names:
            mat = RLAgentTrainer([], args, selfplay=True, name=f'fcp_pop_{layout_name}')
            fcp_pop[layout_name] = mat.load_agents(tag='ijcai')
            print(f'Loaded fcp_pop with {len(fcp_pop)} agents.')
    except FileNotFoundError as e:
        print(f'Could not find saved FCP population, creating them from scratch...\nFull Error: {e}')
        fcp_pop = {}
        agents = []
        num_layers = 2
        seed = 105
        for use_fs in [True, False]:
            for h_dim in [32, 256]: # [8,16], [32, 64], [128, 256], [512, 1024]
                ck_rate = training_steps // 20
                # name = f'cnn_{num_layers}l_' if use_cnn else f'eval_{num_layers}l_'
                name = 'sp_'
                # name += 'pc_' if use_policy_clone else ''
                # name += 'tpl_' if taper_layers else ''
                name += f'fs_' if use_fs else ''
                name += f'hd{h_dim}_'
                name += f'seed{seed}'
                print(f'Starting training for: {name}')
                mat = RLAgentTrainer([], args, selfplay=True, name=name, hidden_dim=h_dim, use_frame_stack=use_fs,
                                     fcp_ck_rate=ck_rate, seed=seed, num_layers=num_layers)
                mat.train_agents(total_timesteps=training_steps)

                for layout_name in args.layout_names:
                    agents = mat.get_fcp_agents(layout_name)
                    pop = RLAgentTrainer([], args, selfplay=True, name=f'fcp_pop_{layout_name}_{name}')
                    pop.set_agents(agents)
                    pop.save_agents()
                    fcp_pop[layout_name] = pop.get_agents()
    return fcp_pop

def get_fcp_agent(args, training_steps=1e7):
    teammates = get_fcp_population(args, training_steps)
    eval_tms = get_eval_teammates(args)
    fcp_trainer = RLAgentTrainer(teammates, args, eval_tms=eval_tms, name='fcp_nips_idx', use_subtask_counts=False, inc_sp=False, use_policy_clone=False, seed=2602)
    fcp_trainer.train_agents(total_timesteps=training_steps)
    return fcp_trainer.get_agents()[0]

def get_hrl_worker(args):
    # Create subtask worker
    name = 'multi_agent_subtask_worker_nips'
    try:
        worker = MultiAgentSubtaskWorker.load(Path(args.base_dir / 'agent_models' / name / args.exp_name), args)
    except FileNotFoundError as e:
        print(f'Could not find saved subtask worker, creating them from scratch...\nFull Error: {e}')
        #worker = MultiAgentSubtaskWorker.create_model_from_pretrained_subtask_workers(args)
        eval_tms = get_eval_teammates(args)
        teammates = get_fcp_population(args, 1e7)
        worker, _ = MultiAgentSubtaskWorker.create_model_from_scratch(args, teammates=teammates, eval_tms=eval_tms)
    return worker

def get_hrl_agent(args, training_steps=1e7):
    teammates = get_fcp_population(args, training_steps)
    worker = get_hrl_worker(args)
    eval_tms = get_eval_teammates(args)
    # Create manager
    rlmt = RLManagerTrainer(worker, teammates, args, eval_tms=eval_tms, use_subtask_counts=False, name='HAHA_nips_idx', inc_sp=False, use_policy_clone=False, seed=2602)
    rlmt.train_agents(total_timesteps=training_steps)
    manager = rlmt.get_agents()[0]
    hrl = HierarchicalRL(worker, manager, args, name='HAHA')
    hrl.save(Path(Path(args.base_dir / 'agent_models' / hrl.name / args.exp_name)))
    return hrl

def get_all_agents(args, training_steps=1e7, agents_to_train='all'):
    agents = {}
    fcp_pop = None
    if agents_to_train == 'all' or 'sp' in agents_to_train:
        agents['sp'] = get_selfplay_agent(args, training_steps=5e6)
    if agents_to_train == 'all' or 'bcp' in agents_to_train:
        agents['bcp'] = get_behavioral_cloning_play_agent(args, training_steps=5e6)
    if agents_to_train == 'all' or 'fcp' in agents_to_train:
        agents['fcp'] = get_fcp_agent(args, training_steps)
    if agents_to_train == 'all' or 'hrl' in agents_to_train:
        agents['hrl'] = get_hrl_agent(args, training_steps)


if __name__ == '__main__':
    args = get_arguments()
    get_selfplay_agent(args, training_steps=5e4)
    # get_bc_and_human_proxy(args)
    #get_behavioral_cloning_play_agent(args, training_steps=1e8)

    # get_fcp_population(args, 2e7)
    # get_fcp_agent(args, training_steps=1e8)
    # get_hrl_worker(args)
    # get_hrl_agent(args, 5e7)

    # create_test_population(args, 1e3)
    # create_pop_from_agents(args)

    #combine_populations(args)
