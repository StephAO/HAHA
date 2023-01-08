import argparse
from pathlib import Path
import torch as th

ARGS_TO_SAVE_LOAD = ['layout_names', 'encoding_fn']

def get_arguments(additional_args=[]):
    """
    Arguments for training agents
    :return:
    """
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--layout-names', default='counter_circuit_o_1order,forced_coordination,asymmetric_advantages,cramped_room,coordination_ring',  help='Overcooked maps to use')
    parser.add_argument('--policy-selection', type=str, default='CEM',
                        help='Which policy selection algorithm to use. Options: "CEM", "PLASTIC". Default: "CEM"')
    parser.add_argument('--multi-env-mode', type=str, default='uniform',
                        help='What distributions to use when training on multiple environments. Options: "uniform", "random", "splits", "weighted_decay". Default: "uniform"')
    parser.add_argument('--subtask-selection', type=str, default='value_based',
                        help='Which subtask selection algorithm to use. Options: "value_based", "dist". Default: "value_based"')
    parser.add_argument('--horizon', type=int, default=1200, help='Max timesteps in a rollout')
    parser.add_argument('--n-envs', type=int, default=50, help='Number of environments to use while training')
    parser.add_argument('--num_stack', type=int, default=3, help='Number of frame stacks to use in training')
    parser.add_argument('--encoding-fn', type=str, default='OAI_egocentric',
                        help='Encoding scheme to use. '
                             'Options: "dense_lossless", "OAI_lossless", "OAI_feats", "OAI_egocentric"')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--eta', type=float, default=0.1, help='eta used in plastic policy update.')
    parser.add_argument('--exp-name', type=str, default='default_exp',
                        help='Name of experiment. Used to tag save files.')
    parser.add_argument('--base-dir', type=str, default=Path.cwd(),
                        help='Base directory to save all models, data, tensorboard metrics.')
    parser.add_argument('--data-path', type=str, default='data',
                        help='Path from base_dir to where the expert data is stored')
    parser.add_argument('--dataset', type=str, default='2019_hh_trials_all.pickle',
                        help='Which set of expert data to use. '
                             'See https://github.com/HumanCompatibleAI/human_aware_rl/tree/master/human_aware_rl/static/human_data for options')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='number of workers for pytorch train_dataloader (default: 4)')
    parser.add_argument('--wandb-mode', type=str, default='online',
                        help='Wandb mode. One of ["online", "offline", "disabled"')
    parser.add_argument('--wandb-ent', type=str, default='stephaneao',
                        help='Wandb entity to log to.')

    parser.add_argument('-c', type=str, default='', help='for stupid reasons, but dont delete')
    parser.add_argument('args', nargs='?', type=str, default='', help='')
    #parser.add_argument('gunicorn_config', type=str, default='', help='for stupid reasons, but dont delete')

    for parser_arg, parser_kwargs in additional_args:
        parser.add_argument(parser_arg, **parser_kwargs)


    args = parser.parse_args()
    args.base_dir = Path(args.base_dir)
    args.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    args.layout_names = args.layout_names.split(',')
    if len(args.layout_names) > 1 and args.encoding_fn != 'OAI_egocentric':
        raise ValueError("Encoding function must be OAI_egocentric if training on multiple layouts")

    return args

def get_args_to_save(curr_args):
    arg_dict = vars(curr_args)
    arg_dict = {k: v for k, v in arg_dict.items() if k in ARGS_TO_SAVE_LOAD}
    return arg_dict

def set_args_from_load(loaded_args, curr_args):
    for arg in ARGS_TO_SAVE_LOAD:
        setattr(curr_args, arg, loaded_args[arg])
