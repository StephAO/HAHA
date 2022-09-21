# Overcooked AI Agents
Repo to train and deploy agents to play overcooked ai

## Set up guide
1. Create conda environment:`conda create -n oai python=3.9`
2. Activate env: `conda activate oai`
3. Install [pytorch](https://pytorch.org/get-started/locally/) based on your use case 
4. Install overcooked-ai: `pip install git+https://github.com/StephAO/overcooked_ai.git`  
***Note***: The above repo is a modified version of the original [overcooked ai repo](https://github.com/HumanCompatibleAI/overcooked_ai)
5. Clone modified overcooked-ai repo: `git clone https://github.com/StephAO/oai_agents.git'   
6. Move to repo dir: `cd oai_agents`
7. Install this package: `pip install -e .`

## Training Agents
### TODO enable training with just arguments
`python -m oai_agents.agents.rl`
