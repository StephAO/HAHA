from oai_agents.agents.agent_utils import load_agent, DummyAgent
from oai_agents.common.arguments import get_arguments
from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv

import numpy as np
from pathlib import Path
from tqdm import tqdm
from scripts.train_agents import get_bc_and_human_proxy
from scipy.stats import entropy
from torch.distributions.categorical import Categorical

def get_entropy(dist):
    if isinstance(dist, Categorical):
        probs = dist.probs.detach().squeeze().numpy()
    else:
        probs = dist.distribution.probs.squeeze().numpy()
    ent = entropy(probs)
    return ent

def run_game(env, agent,  deterministic=False):
    done = False
    cum_reward = 0
    obs = env.reset()
    # print(env.layout_name, env.teammate.name, agent.name)
    entropy_per_state = []
    states_visited = set()
    while not done:
        state_hash = env.state.specific_hash(env.p_idx)
        if state_hash not in states_visited:
            states_visited.add(state_hash)
        obs = {k: v for k, v in obs.items() if k in agent.policy.observation_space.keys()}
        entropy_per_state.append(get_entropy(agent.get_distribution(obs)))
        action = agent.predict(obs, deterministic=deterministic)[0]
        obs, reward, done, info = env.step(action)
        cum_reward += reward
    # print(cum_reward)
    # print(f'{env.layout_name}: {entropy_per_state}')
    return cum_reward, entropy_per_state, len(states_visited)

args = get_arguments()

agents = []
for name in ['sp', 'sp_det', 'bcp', 'bcp_det']:
    fn = args.base_dir / 'agent_models' / name
    agents.append(load_agent(fn, args=args))
    # if 'sp' in name:
    #     teammates.append(load_agent(fn, args=args))
    # else:
    #     teammates.append(get_bc_and_human_proxy(args)[0])

bc, human_proxy = get_bc_and_human_proxy(args)
agents.extend([bc, human_proxy])
# teammates.extend([hp, bc])

teammates = [load_agent(args.base_dir / 'agent_models' / 'SP', args=args), human_proxy, DummyAgent('random')]
# teammate = agent# load_agent(fn, args=args)


eval_envs_kwargs = {'is_eval_env': True, 'args': args, 'horizon': 400, 'ret_completed_subtasks': True}
args.layout_names = ['asymmetric_advantages', 'forced_coordination', 'counter_circuit_o_1order', 'cramped_room', 'coordination_ring']
eval_envs = [OvercookedGymEnv(**{'env_index': i, **eval_envs_kwargs}) for i in range(len(args.layout_names))]


for agent in agents:
    for agent_det in [True, False]:
        name = agent.name + '_' if not isinstance(agent, dict) else 'bc'
        if '_det' in name:
            name = name.replace('_det_', '_d')
        elif name != 'bc':
            name += 's'
        name += ('d' if agent_det else 's')
        csv_format = [name]
        for teammate_det in [True, False]:
            entropy_per_state, num_states_vis = [], []
            for env in eval_envs:
                env.deterministic = teammate_det
                for teammate in teammates:
                    tm = teammate[env.layout_name][0] if isinstance(teammate, dict) else teammate
                    a = agent[env.layout_name][0] if isinstance(agent, dict) else agent
                    env.encoding_fn = a.encoding_fn
                    env.new_agent = a.new_agent
                    env.set_teammate(tm)
                    for i in range(2 if agent_det and teammate_det else 10):
                        env.set_reset_p_idx(i % 2)
                        _, eps, sv = run_game(env, a, deterministic=agent_det)
                        entropy_per_state += eps
                        num_states_vis.append(sv)
            # csv_format.append(np.mean(entropy_per_state))
            csv_format.append(np.mean(num_states_vis))
        print(','.join([str(c) for c in csv_format]))
                # csv_format.extend([np.mean(epspl), np.std(epspl), np.min(epspl), np.max(epspl)])
            # print(','.join([str(c) for c in csv_format]))
                # print(f'{env.layout_name:>23}: avg:{np.mean(epspl):.3f}, std{np.std(epspl):.3f}, min: {np.min(epspl):.3f}, max:{np.max(epspl):.3f}')

            # print(f'Over all layouts       : avg:{np.mean(entropy_per_state):.3f}, std{np.std(entropy_per_state):.3f}, min: {np.min(entropy_per_state):.3f}, max:{np.max(entropy_per_state):.3f}')
            # print(f'{np.mean(entropy_per_state)},{np.std(entropy_per_state)},{np.min(entropy_per_state)},{np.max(entropy_per_state)}')