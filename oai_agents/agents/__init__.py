from oai_agents.agents.base_agent import *
from oai_agents.agents.il import *
from oai_agents.agents.rl import *
from oai_agents.agents.hrl import *
from oai_agents.common.arguments import get_arguments
from pathlib import Path
import torch as th

if __name__ == '__main__':
    from oai_agents.common.arguments import get_arguments
    args = get_arguments()
    agent = load_agent(Path('test_data'), args)
    print(type(agent))

