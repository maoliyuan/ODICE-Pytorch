import argparse, yaml
import gym
import os
import d4rl
import numpy as np
import torch
from tqdm import trange
from collections import defaultdict
from odice import ODICE
from policy import GaussianPolicy
from value_functions import ValueFunction, TwinV
from util import return_range, set_seed, sample_batch, torchify, evaluate
import wandb
import time



def get_env_and_dataset(env_name, max_episode_steps, normalize):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)
    dataset_length = len(dataset['terminals'])
    if any(s in env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
        min_ret, max_ret = return_range(dataset, max_episode_steps)
        print(f'Dataset returns have range [{min_ret}, {max_ret}]')
        dataset['rewards'] /= (max_ret - min_ret)
        dataset['rewards'] *= max_episode_steps
    elif 'antmaze' in env_name:
        dataset['rewards'] = np.where(dataset['rewards'] == 0., -3.0, 0)

    print("***********************************************************************")
    print(f"Normalize for the state: {normalize}")
    print("***********************************************************************")
    if normalize:
        mean = dataset['observations'].mean(0)
        std = dataset['observations'].std(0) + 1e-3
        dataset['observations'] = (dataset['observations'] - mean)/std
        dataset['next_observations'] = (dataset['next_observations'] - mean)/std
    else:
        obs_dim = dataset['observations'].shape[1]
        mean, std = np.zeros(obs_dim), np.ones(obs_dim)

    for k, v in dataset.items():
        dataset[k] = torchify(v)
    for k, v in list(dataset.items()):
        assert len(v) == dataset_length, 'Dataset values must have same length'

    return env, dataset, mean, std


def main(args):
    args.log_dir = '/'.join(__file__.split('/')[: -1]) + '/' + args.log_dir
    args.model_dir = '/'.join(__file__.split('/')[: -1]) + '/' + args.model_dir
    if 'antmaze' in args.env_name:
        args.eval_period = 20000 if args.eval_period < 20000 else args.eval_period
        args.n_eval_episodes = 50
        args.layer_norm = False
        if 'large' in args.env_name or 'umaze-diverse' in args.env_name:
            args.use_twin_v = False
    
    wandb.init(project=f"odice_offline_RL",
               entity="your name",
               name=f"{args.env_name}_ODICE",
               config={
                   "env_name": args.env_name,
                   "type": args.type,
                   "seed": args.seed,
                   "normalize": args.normalize,
                   "Lambda": args.Lambda,
                   "eta": args.eta,
                   "use_twin_v": args.use_twin_v,
                   "use_tanh": args.use_tanh,
                   "f_name": args.f_name,
                   "weight_decay": args.weight_decay,
                   "gamma": args.discount,
               })
    torch.set_num_threads(1)

    env, dataset, mean, std = get_env_and_dataset(args.env_name,
                                                  args.max_episode_steps,
                                                  args.normalize)
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]   # this assume continuous actions
    set_seed(args.seed, env=env)

    policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=1024, n_hidden=2, use_tanh=args.use_tanh)
    vf = TwinV(obs_dim, layer_norm=args.layer_norm, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden) if args.use_twin_v else ValueFunction(obs_dim, layer_norm=args.layer_norm, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)

    odice = ODICE(
        vf=vf,
        policy=policy,
        max_steps=args.train_steps,
        f_name=args.f_name,
        Lambda=args.Lambda,
        eta=args.eta, 
        discount=args.discount,
        value_lr=args.value_lr,
        policy_lr=args.policy_lr,
        weight_decay=args.weight_decay,
        use_twin_v = args.use_twin_v,
    )
    if os.path.exists(f"{args.model_dir}/{args.env_name}" + f"/eta_{args.eta}_Lambda_{args.Lambda}_checkpoint_{args.load_step}.pth"):
        odice.load(f"{args.model_dir}/{args.env_name}", args.load_step)

    def eval(step):
        eval_returns = np.array([evaluate(env, policy, mean, std) \
                                 for _ in range(args.n_eval_episodes)])
        normalized_returns = d4rl.get_normalized_score(args.env_name, eval_returns) * 100.0
        return_info = {}
        return_info["normalized return mean"] = normalized_returns.mean()
        return_info["normalized return std"] = normalized_returns.std()
        return_info["percent difference 10"] = (normalized_returns[: 10].min() - normalized_returns[: 10].mean()) / normalized_returns[: 10].mean()
        wandb.log(return_info, step=step)

        print("---------------------------------------")
        print(f"Env: {args.env_name}, Evaluation over {args.n_eval_episodes} episodes: D4RL score: {normalized_returns.mean():.3f}")
        print("---------------------------------------")

        return normalized_returns.mean()

    algo_name = f"{args.type}_lambda-{args.Lambda}_gamma-{args.discount}_eta-{args.eta}_f_name-{args.f_name}_use_tanh-{args.use_tanh}_normalize-{args.normalize}_use_twin_v-{args.use_twin_v}"
    os.makedirs(f"{args.log_dir}/{args.env_name}/{algo_name}", exist_ok=True)
    eval_log = open(f"{args.log_dir}/{args.env_name}/{algo_name}/seed-{args.seed}.txt", 'w')
    for step in trange(args.train_steps):
        if args.type == 'orthogonal_true_g':
            odice.orthogonal_true_g_update(**sample_batch(dataset, args.batch_size))
        elif args.type == 'true_g':
            odice.true_g_update(**sample_batch(dataset, args.batch_size))
        elif args.type == 'semi_g':
            odice.semi_g_update(**sample_batch(dataset, args.batch_size))

        if (step+1) % args.eval_period == 0:
            average_returns = eval(odice.step)
            eval_log.write(f'{step + 1}\tavg return: {average_returns}\t\n')
            eval_log.flush()
    eval_log.close()
    os.makedirs(f"{args.model_dir}/{args.env_name}", exist_ok=True)
    odice.save(f"{args.model_dir}/{args.env_name}")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env_name', type=str, default="hopper-medium-replay-v2")
    parser.add_argument('--Lambda', type=float, default=0.6)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument("--type", type=str, choices=['orthogonal_true_g', 'true_g', 'semi_g'], default='orthogonal_true_g')
    with open("configs/offline_RL.yaml", "r") as file:
        config = yaml.safe_load(file)
    now = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    args = parser.parse_args(namespace=argparse.Namespace(**config))

    main(args)