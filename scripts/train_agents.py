from oai_agents.agents.il import BehavioralCloningTrainer
from oai_agents.agents.rl import SingleAgentTrainer, MultipleAgentsTrainer, SB3Wrapper
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
    pop_agents = []
    base_path = args.base_dir / 'agent_models'
    for agent_name, mid_ck in zip(['fs_16_s16384', 'fs_256_s16384', 'no_fs_16_s16384', 'no_fs_256_s16384'], ['ck_5', 'ck_4', 'ck_7', 'ck_7']):
        best = SB3Wrapper.load(base_path / agent_name / 'best' / 'agents_dir' / 'agent_0', args)
        worst = SB3Wrapper.load(base_path / agent_name / 'ck_0' / 'agents_dir' / 'agent_0', args)
        mid = SB3Wrapper.load(base_path / agent_name / mid_ck / 'agents_dir' / 'agent_0', args)
        pop_agents += [best, worst, mid]

    mat = MultipleAgentsTrainer(args, name='fcp_pop_seed=16384', num_agents=0)
    mat.set_agents(pop_agents)
    print(len(mat.agents))
    mat.save_agents()

def combine_populations(args, pop_names):
    full_pop = []
    for pop_name in pop_names:
        mat = MultipleAgentsTrainer(args, name=pop_name, num_agents=0)
        full_pop += mat.load_agents()
    verify = input('WARNING: You are about to overwrite fcp_pop. Press Y to continue, or anything else to cancel.')
    if verify.lower() == 'y':
        mat = MultipleAgentsTrainer(args, name='fcp_pop', num_agents=0)
        mat.set_agents(full_pop)
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
def get_selfplay_agent(args, training_steps=1e7):
    self_play_trainer = MultipleAgentsTrainer(args, name='selfplay', num_agents=1)
    try:
        self_play_trainer.load_agents(tag='final')
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
            bct.load_agents(tag='final')
        except FileNotFoundError as e:
            print(f'Could not find saved BC and human proxy, creating them from scratch...\nFull Error: {e}')
            bct.train_agents(epochs=300)
        bc, human_proxy = bct.get_agents()
        bcs[layout_name] = [bc]
        human_proxies[layout_name] = [human_proxy]

    args.layout_names = all_layouts
    return bcs, human_proxies

# BCP
def get_behavioral_cloning_play_agent(args, training_steps=1e7):
    bcs, human_proxies = get_bc_and_human_proxy(args)
    teammates = bcs
    self_play_trainer = SingleAgentTrainer(teammates, args, name='bcp')
    try:
        bcp = self_play_trainer.load_agents()
    except FileNotFoundError as e:
        print(f'Could not find saved BCP, creating them from scratch...\nFull Error: {e}')
        self_play_trainer.train_agents(total_timesteps=training_steps)
        bcp = self_play_trainer.get_agents()
    return bcp

# PP
def get_population_play_agent(args, pop_size=8, training_steps=1e7):
    pop_play_trainer = MultipleAgentsTrainer(args, name='pop_play', num_agents=pop_size, use_lstm=False, hidden_dim=64)
    pop_play_trainer.train_agents(total_timesteps=training_steps)
    all_agents = pop_play_trainer.get_agents()
    score_matrix = calculate_agent_pairing_score_matrix(all_agents, args)
    avg_score_per_agent = np.mean(score_matrix, axis=1)
    best_agent = all_agents[np.argmax(avg_score_per_agent)]
    return best_agent

# FCP
def get_fcp_population(args, training_steps=2e7):
    try:
        mat = MultipleAgentsTrainer(args, name='fcp_pop', num_agents=0)
        fcp_pop = mat.load_agents(tag='final')
        print(f'Loaded fcp_pop with {len(fcp_pop)} agents.')
    except FileNotFoundError as e:
        print(f'Could not find saved FCP population, creating them from scratch...\nFull Error: {e}')
        agents = []
        use_fs = False
        for h_dim in [8, 16]: # [32, 64], [128, 256], [512, 1024]
            seed = h_dim # 64, 1024, 16384
            ck_rate = training_steps // 20
            name = f'fs_{h_dim}' if use_fs else f'no_fs_{h_dim}'
            print(f'Starting training for: {name}')
            mat = MultipleAgentsTrainer(args, name=name, num_agents=1, hidden_dim=h_dim, use_frame_stack=use_fs,
                                        fcp_ck_rate=ck_rate, seed=seed)
            mat.train_agents(total_timesteps=training_steps)
            mat.save_agents(path=(args.base_dir / 'agent_models' / 'sp'), tag=name)
            agents.extend(mat.get_fcp_agents())
        pop = MultipleAgentsTrainer(args, name='fcp_pop', num_agents=0)
        pop.set_agents(agents)
        pop.save_agents()
        fcp_pop = pop.get_agents()
    return fcp_pop

def get_fcp_agent(args, training_steps=1e7):
    teammates = get_fcp_population(args, training_steps)
    eval_tms = get_eval_teammates(args)
    fcp_trainer = SingleAgentTrainer(teammates, args, eval_tms=eval_tms, name='fcp_sp_stc', inc_sp=True, use_subtask_counts=True)
    fcp_trainer.train_agents(total_timesteps=training_steps)
    return fcp_trainer.get_agents()[0]

def get_hrl_worker(args):
    # Create subtask worker
    name = 'multi_agent_subtask_worker'
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
    rlmt = RLManagerTrainer(worker, teammates, args, eval_tms=eval_tms, use_subtask_counts=True, name='hrl_manager')
    rlmt.train_agents(total_timesteps=training_steps)
    manager = rlmt.get_agents()[0]
    hrl = HierarchicalRL(worker, manager, args)
    hrl.save(Path(Path(args.base_dir / 'agent_models' / hrl.name / args.exp_name)))
    return hrl

def get_all_agents(args, training_steps=1e7, agents_to_train='all'):
    agents = {}
    fcp_pop = None
    if agents_to_train == 'all' or 'sp' in agents_to_train:
        agents['sp'] = get_selfplay_agent(args, training_steps=5e6)
    if agents_to_train == 'all' or 'bcp' in agents_to_train:
        agents['bcp'] = get_behavioral_cloning_play_agent(args, training_steps=5e6)
    # if agents_to_train == 'all' or 'pp' in agents_to_train:
    #     agents['pp'] = get_population_play_agent(args, training_steps=training_steps)
    if agents_to_train == 'all' or 'fcp' in agents_to_train:
        agents['fcp'] = get_fcp_agent(args, training_steps)
    if agents_to_train == 'all' or 'hrl' in agents_to_train:
        agents['hrl'] = get_hrl_agent(args, training_steps)


### TESTING STUFF ###
def create_test_population(args, training_steps=1e7):
    agents = []
    h_dim= 64
    seed = 8

    # name = 'frame_stack'
    # print(f'Starting training for: {name}')
    # mat = MultipleAgentsTrainer(args, name=name, num_agents=1, use_frame_stack=True, hidden_dim=h_dim, seed=seed)
    # mat.train_agents(total_timesteps=1e6)
    a = False
    if a:
        args.layout_names = ['counter_circuit_o_1order', 'forced_coordination', 'asymmetric_advantages']
        get_behavioral_cloning_play_agent(args, training_steps=3e6)

    # name = 'multi_env_uniform'
    # print(f'Starting training for: {name}')
    # args.layout_names = ['counter_circuit_o_1order','forced_coordination','asymmetric_advantages']
    # args.multi_env_mode = 'uniform'
    # mat = MultipleAgentsTrainer(args, name=name, num_agents=1, hidden_dim=h_dim, seed=seed)
    # mat.train_agents(total_timesteps=3e6)

    else:
        # args.layout_names = ['counter_circuit_o_1order']
        # # get subtask worker
        # name = 'multi_agent_subtask_worker'
        # worker = MultiAgentSubtaskWorker.load(Path(args.base_dir / 'agent_models' / name / args.exp_name), args)
        #
        # name = 'hrl_default'
        # mat = MultipleAgentsTrainer(args, name='fcp_pop', num_agents=0)
        # tms = mat.load_agents()
        # inc_sp = False

        from oai_agents.common.subtasks import Subtasks
        from oai_agents.gym_environments.worker_env import OvercookedSubtaskGymEnv
        name = 'multi_agent_subtask_worker'
        worker = MultiAgentSubtaskWorker.load(Path(args.base_dir / 'agent_models' / name / args.exp_name), args)
        tms = get_eval_teammates(args)
        total_f = 0
        for env_idx, ln in enumerate(args.layout_names):
            print('--------------------')
            print(f'layout: {ln}')
            print('--------------------')
            for i in range(Subtasks.NUM_SUBTASKS - 1):
                if ln == 'asymmetric_advantages' and  Subtasks.IDS_TO_SUBTASKS[i] in ['put_soup_closer', 'get_soup_from_counter', 'get_onion_from_counter', 'get_plate_from_counter']:
                    continue

                print(f"Subtask {i}: {Subtasks.IDS_TO_SUBTASKS[i]}, layout: {ln}")
                env_kwargs = {'single_subtask_id': i, 'stack_frames': False, 'full_init': True, 'args': args}
                eval_env1 = OvercookedSubtaskGymEnv(**{'env_index': env_idx, 'is_eval_env': True, **env_kwargs})
                # eval_env1.setup_visualization()
                tms = tms[ln] if type(tms) == dict else tms
                for tm in tms:
                # tm = DummyAgent('random')
                    eval_env1.set_teammate(tm)
                    # print("Running with determinism")
                    # w = SB3Wrapper.load(Path(f'/home/miguel/Documents/projects/oai_agents/agent_models/subtask_worker_{i}/best/agents_dir/agent_0'), args)
                    _, tot_f = eval_env1.evaluate(worker.agents[i]) #
                    total_f += tot_f
                # eval_env2 = OvercookedSubtaskGymEnv(**{'env_index': 0, 'is_eval_env': False, **env_kwargs})
                # eval_env2.set_teammate(tms)
                # print("Running without determinism")
                # eval_env2.evaluate(worker.agents[i])
        print(total_f)
        exit(0)

        # b = True
        # if b:
        #     inc_sp = True
        #     tms = []
        #     name = 'hrl_sp'
        #
        # # Create manager
        # rlmt = RLManagerTrainer(worker, tms, args, use_subtask_counts=True, inc_sp=inc_sp, name=name)
        # rlmt.train_agents(total_timesteps=training_steps)
        # manager = rlmt.get_agents()[0]
        # hrl = HierarchicalRL(worker, manager, args, name='hrl_sp')
        # hrl.save(Path(Path(args.base_dir / 'agent_models' / hrl.name / args.exp_name)))


if __name__ == '__main__':
    args = get_arguments()
    # create_test_population(args, 1e3)
    # get_hrl_agent(args, 1e7)

    # create_test_population(args, 1e3)
    # get_bc_and_human_proxy(args)
    #get_fcp_agent(args, training_steps=1e7)
    teammates = get_fcp_population(args, 2e7)
    # get_hrl_worker(args)
    # get_bc_and_human_proxy(args)
    # get_fcp_agent(args, training_steps=1e7)
