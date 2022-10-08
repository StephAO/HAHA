class TypeBasedAdaptor(OAIAgent):
    def __init__(self, p1_agents, p2_agents, selfplay_table, idx, args):
        super(TypeBasedAdaptor, self).__init__()
        self.args = args
        self.set_player_idx(idx)
        self.p1_agents = p1_agents
        self.p2_agents = p2_agents
        self.selfplay_table = selfplay_table
        self.encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
        self.policy_selection = args.policy_selection
        self.name = f'type_based_adaptor_{self.policy_selection}_p{idx + 1}'
        self.env = OvercookedGymEnv(p1=p1_agents[0], p2=p2_agents[0], args=args)
        self.policy = np.random.choice(self.p1_agents if idx == 0 else self.p2_agents)

    @staticmethod
    def create_models(args, bc_epochs=2, rl_epochs=2):
        # TODO add options to each kind of agent (e.g. reward shaping / A2C vs PPO for RL agents, using subtasks for BC Agents
        p1_agents = []
        p2_agents = []

        # BC agents
        for dataset_file in ['tf_test_5_5.1.pickle', 'tf_test_5_5.2.pickle', 'tf_test_5_5.3.pickle',
                             'tf_test_5_5.4.pickle', 'tf_test_5_5.5.pickle', 'all_trials.pickle']:
            bct = BehavioralCloningTrainer(dataset_file, args)
            bce = bc_epochs  # (bc_epochs // 5) if dataset_file == 'all_trials.pickle' else bc_epochs
            bct.train_agents(epochs=bce)
            bc_p1 = bct.get_agents(idx=0)
            bc_p2 = bct.get_agents(idx=1)
            p1_agents.append(bc_p1)
            p2_agents.append(bc_p2)

        # RL two single agents
        rl_tsat = TwoSingleAgentsTrainer(args)
        rl_tsat.train_agents(total_timesteps=rl_steps)
        p1_agents.append(rl_tsat.get_agents(idx=0))
        p2_agents.append(rl_tsat.get_agents(idx=1))

        # RL double agent
        rl_odat = OneDoubleAgentTrainer(args)
        rl_odat.train_agents(total_timesteps=rl_steps)
        p1_agents.append(rl_odat.get_agents(idx=0))
        p2_agents.append(rl_odat.get_agents(idx=1))

        # RL single agents trained with BC partner
        rl_sat = SingleAgentTrainer(bc_p2, 1, args)
        rl_sat.train_agents(total_timesteps=rl_steps)
        p1_agents.append(rl_sat.get_agents(idx=0))
        rl_sat = SingleAgentTrainer(bc_p1, 0, args)
        rl_sat.train_agents(total_timesteps=rl_steps)
        p2_agents.append(rl_sat.get_agents(idx=1))

        # TODO deal with different layouts logic
        selfplay_table = TypeBasedAdaptor.calculate_selfplay_table(p1_agents, p2_agents, args)
        return p1_agents, p2_agents, selfplay_table

    @staticmethod
    def calculate_selfplay_table(p1_agents, p2_agents, args):
        selfplay_table = np.zeros((len(p1_agents), len(p2_agents)))
        for i, p1 in enumerate(p1_agents):
            for j, p2 in enumerate(p2_agents):
                selfplay_table[i, j] = TypeBasedAdaptor.run_full_episode((p1, p2), args)
        print(selfplay_table)
        return selfplay_table

    @staticmethod
    def run_full_episode(players, args):
        args.horizon = 1200
        env = OvercookedGymEnv(args=args)
        env.reset()
        for player in players:
            player.policy.eval()

        done = False
        total_reward = 0
        while not done:
            if env.visualization_enabled:
                env.render()
            joint_action = [None, None]
            for i, player in enumerate(players):
                if isinstance(player, DoubleAgentWrapper):
                    joint_action[i] = player.predict(env.get_obs(p_idx=None))[0]
                else:
                    joint_action[i] = player.predict(env.get_obs(p_idx=i))[0]
            obs, reward, done, info = env.step(joint_action)

            total_reward += np.sum(info['sparse_r_by_agent'])
        return total_reward

    def predict(self, obs: th.Tensor, state=None, episode_start=None, deterministic=False) -> Tuple[int, Union[th.Tensor, None]]:
        """
        Given an observation return the index of the action and the agent state if the agent is recurrent.
        """
        return self.policy.predict(obs, state=state, episode_start=episode_start, deterministic=deterministic)

    def get_distribution(self, obs: th.Tensor):
        return self.policy.get_distribution(obs)

    def step(self, state, joint_action):
        if self.p_idx is None:
            raise ValueError('Player idx must be set before TypeBasedAdaptor.step can be called')
        self.trajectory.append((state, joint_action))
        self.update_beliefs(state, joint_action)

    def reset(self, state, player_idx):
        # NOTE this is for a reset between episodes. Create a new TypeBasedAdaptor if this is for a new human
        super().reset(state, player_idx)
        self.policy = self.select_policy()
        self.init_behavior_dist()
        self.trajectory = []

    def init_behavior_dist(self):
        # initialize probability distribution as uniform
        self.eta = self.args.eta
        assert len(self.p1_agents) == len(self.p2_agents)
        n = len(self.p1_agents)
        self.behavior_dist = np.full((2, n), 1 / n)

    def update_beliefs(self, state, joint_action):
        # Based on algorithm 4 (PLASTIC-Policy) of https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/AIJ16-Barrett.pdf
        prior_teammate_policies = self.p1_agents if self.t_idx == 0 else self.p2_agents
        action = joint_action[self.t_idx]
        for i, policy in enumerate(prior_teammate_policies):
            action_idx = Action.ACTION_TO_INDEX[action]
            obs = self.encoding_fn(self.env.mdp, state, self.env.grid_shape, self.args.horizon, p_idx=self.t_idx)
            prob_dist = model.get_distribution(obs)
            action_prob = prob_dist[action_idx]
            # calculate loss for model
            loss_model = 1 - action_prob
            # Update Probability Distribution according to loss for that model
            self.behavior_dist[self.t_idx][i] *= (1 - self.eta * loss_model)
        # Normalize Probabiity Distribution
        self.behavior_dist[self.t_idx] = self.behavior_dist[self.t_idx] / np.sum(self.behavior_dist[self.t_idx])

    def select_policy_plastic(self):
        best_match_policy_idx = np.argmax(self.behavior_dist[self.t_idx])
        return self.get_best_complementary_policy(best_match_policy_idx)

    def select_policy_using_cross_entropy_metric(self, horizon=10):
        # teammate_idx === t_idx
        prior_teammate_policies = self.p1_agents if self.t_idx == 0 else self.p2_agents
        horizon = min(horizon, len(self.trajectory))
        trajectory = self.trajectory[-horizon:]
        best_cem = 0
        best_match_policy_idx = None
        for policy_idx, policy in enumerate(prior_teammate_policies):
            cem = 0
            for t in range(horizon):
                state, joint_action = trajectory[t]
                action_idx = Action.ACTION_TO_INDEX[joint_action[self.t_idx]]
                obs = self.encoding_fn(self.env.mdp, state, self.env.grid_shape, self.args.horizon, p_idx=self.t_idx)
                dist = policy.get_distribution(obs)
                cem += dist.log_prob(action_idx)
            cem = cem / horizon
            if cem > best_cem:
                best_match_policy_idx = policy_idx
                best_cem = cem

        return self.get_best_complementary_policy(best_match_policy_idx)

    def get_best_complementary_policy(self, policy_idx):
        # NOTE if t_idx == 0 then p_idx == 1 and vice versa
        team_scores = self.selfplay_table[policy_idx, :] if self.t_idx == 0 else self.selfplay_table[:, policy_idx]
        own_policies = self.p1_agents if self.p_idx == 0 else self.p2_agents
        return own_policies[np.argmax(team_scores)]

    def select_policy(self):
        # TODO use distribution to find the most similar model to human,
        #      then select the most complementary model using the selfplay_table
        if self.policy_selection == 'CEM':
            self.curr_policy = self.select_policy_using_cross_entropy_metric()
        elif self.policy_selection == 'PLASTIC':
            self.curr_policy = self.select_policy_plastic()
        else:
            raise NotImplementedError(f'{self.policy_selection} is not an implemented policy selection algorithm')
        print(f'Now using policy {self.curr_policy.name}.')

    def save(self, path: str = None):
        """Save all models and selfplay table"""
        base_dir = args.base_dir / 'agent_models' / 'type_based_agents'
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        for model in self.p1_agents:
            save_path = base_dir / model.name
            model.save(save_path)

        for model in self.p2_agents:
            save_path = base_dir / model.name
            model.save(save_path)

        self.selfplay_table.to_pickle(base_dir / 'selfplay_table.pickle')

    def load(self, path: str = None):
        """Load all models and selfplay table"""
        base_dir = args.base_dir / 'agent_models' / 'type_based_agents'
        for model in self.p1_agents:
            load_path = base_dir / model.name
            model.load(load_path)

        for model in self.p2_agents:
            load_path = base_dir / model.name
            model.load(load_path)

        self.selfplay_table.read_pickle(base_dir / 'selfplay_table.pickle')


# TODO wandb init each agent at the start of their training


if __name__ == '__main__':
    from oai_agents.common.arguments import get_arguments

    args = get_arguments()
    p1_agents, p2_agents, sp_table = TypeBasedAdaptor.create_models(args)
    tba = TypeBasedAdaptor(p1_agents, p2_agents, sp_table, 0, args)