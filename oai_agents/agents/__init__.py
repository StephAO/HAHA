from oai_agents.agents.base_agent import *
from oai_agents.agents.il import *
from oai_agents.agents.rl import *
from oai_agents.agents.hrl import *
import torch as th

def load_agent(agent_path, args):
    try:
        load_dict = th.load(agent_path)
    except FileNotFoundError as e:
        raise ValueError(f'Could not find file:{e}') # TODO print options
    agent = load_dict['agent_type'].load(agent_path, args)
    assert isinstance(agent, OAIAgent)
    return agent

if __name__ == '__main__':
    from oai_agents.common.arguments import get_arguments
    args = get_arguments()
    agent = load_agent('test_data/test', args)
    print(type(agent))

