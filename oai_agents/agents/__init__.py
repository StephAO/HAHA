from oai_agents.agents.base_agent import *
from oai_agents.agents.il import *
from oai_agents.agents.rl import *
from oai_agents.agents.hrl import *
from oai_agents.common.arguments import get_arguments
from pathlib import Path
import torch as th

def fix_save_pickle(agent_path, args=None):
    """ Starter code if pickles has modules '__main__.xxxx' insated of 'oai_agents.xxxx. This should not be a problem is
    scripts/train_agents is used."""
    import oai_agents
    args = args or get_arguments()
    agent_path = Path(agent_path)

    try:
        load_dict = th.load(agent_path / 'agent_file')
        print(load_dict)
    except FileNotFoundError as e:
        raise ValueError(f'Could not find file:{e}') # TODO print options

    save_dict = {}
    for k, v in load_dict.items():
        # print(k, v)
        # print('__main__' in str(v))
        if '__main__.SubtaskWorker' in str(v):
            v = oai_agents.agents.SubtaskWorker
        elif '__main__.HierarchicalRL' in str(v):
            v = oai_agents.agents.HierarchicalRL
        elif '__main__' in str(v):
            print(v)
            # cls = oai_agents.agents.HierarchicalRL
            # save_dict[k.replace('__main__', 'oai_agents.agents')] = v
        save_dict[k] = v
    print(load_dict)
    print(save_dict)
    th.save(save_dict, agent_path / 'agent_file_fixed')



if __name__ == '__main__':
    from oai_agents.common.arguments import get_arguments
    fix_save_pickle('/home/miguel/Documents/projects/overcooked-demo/server/static/assets/agents/oai_hrl/manager')

    # args = get_arguments()
    # agent = load_agent(Path('test_data'), args)
    # print(type(agent))

