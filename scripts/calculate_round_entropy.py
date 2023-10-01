from oai_agents.agents.agent_utils import load_agent
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
    # probs = np.delete(probs, -2)
    # probs = probs / np.sum(probs)
    ent = entropy(probs)
    return ent

def run_game(env, agent, deterministic):
    done = False
    cum_reward = 0
    obs = env.reset()
    entropy_per_state = []
    while not done:
        obs = {k: v for k, v in obs.items() if k in agent.policy.observation_space.keys()}
        entropy_per_state.append(get_entropy(agent.get_distribution(obs)))
        action = agent.predict(obs, deterministic=deterministic)[0]
        obs, reward, done, info = env.step(action)
        cum_reward += reward
    print(f'{env.layout_name}: {entropy_per_state}')
    return cum_reward, entropy_per_state

args = get_arguments()

# fn = args.base_dir / 'agent_models' / 'SP'
# agent = load_agent(fn, args=args)
# teammate = load_agent(fn, args=args)
agent, teammate = get_bc_and_human_proxy(args)

deterministic = False


eval_envs_kwargs = {'is_eval_env': True, 'args': args, 'horizon': 400, 'ret_completed_subtasks': True}
args.layout_names = ['asymmetric_advantages']#'forced_coordination', 'counter_circuit_o_1order', 'asymmetric_advantages', 'cramped_room',
                     # 'coordination_ring']
eval_envs = [OvercookedGymEnv(**{'env_index': i, **eval_envs_kwargs}) for i in range(len(args.layout_names))]


entropy_per_state = []
for env in eval_envs:
    env.deterministic = deterministic
    tm = teammate[env.layout_name][0] if isinstance(teammate, dict) else teammate
    a = agent[env.layout_name][0] if isinstance(agent, dict) else agent
    env.encoding_fn = a.encoding_fn
    env.set_teammate(tm)
    _, eps = run_game(env, a, deterministic)
    entropy_per_state += eps

print(f'AVERAGE ENTROPY: {np.mean(entropy_per_state)}')