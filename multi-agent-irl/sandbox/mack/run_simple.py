#!/usr/bin/env python3
import logging
import os
import itertools
import click
import gym
import make_env
import sys
sys.path.append('MA-AIRL/multi-agent-irl/')
from rl import bench
from rl import logger
from rl.common import set_global_seeds
from rl.common.vec_env.subproc_vec_env import SubprocVecEnv
from sandbox.mack.acktr_disc import learn as learn_disc
from sandbox.mack.acktr_cont import learn as learn_cont
from sandbox.mack.policies import CategoricalPolicy, GaussianPolicy, MultiCategoricalPolicy

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
identical = None


def train(logdir, env_id, num_timesteps, lr, timesteps_per_batch, seed, num_cpu):
    global identical
    def create_env(rank):
        def _thunk():
            global identical
            if ':' in env_id:
                if 'assistive_gym' in env_id:
                    import importlib
                    module = importlib.import_module('assistive_gym.envs')
                    env_class = getattr(module, env_id.split(':')[1].split('-')[0] + 'Env')
                    env = env_class()
                else:
                    env = gym.make(env_id)
            else:
                env = make_env.make_env(env_id)
                env.seed(seed + rank)
                env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
                                allow_early_resets=True)
                identical=make_env.get_identical(env_id)
            gym.logger.setLevel(logging.WARN)
            return env
        return _thunk

    logger.configure(logdir, format_strs=['stdout', 'log', 'json', 'tensorboard'])
    
    set_global_seeds(seed)
    env = SubprocVecEnv([create_env(i) for i in range(num_cpu)], is_multi_agent=True)
    if isinstance(env.action_space[0], gym.spaces.Discrete):
        policy_fn = CategoricalPolicy
        learn_disc(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), nprocs=num_cpu,
                nsteps=timesteps_per_batch // num_cpu, lr=lr, ent_coef=0.00, identical=identical)
        env.close()
    else: 
        policy_fn = GaussianPolicy
        learn_cont(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), nprocs=num_cpu,
                nsteps=timesteps_per_batch // num_cpu, lr=lr, ent_coef=0.00, identical=identical)
        env.close()
    

from pathlib import Path
@click.command()
@click.option('--logdir', type=click.STRING, default=str(Path(__file__).parent.parent.parent.parent)+'/logs/ma-particle/')
@click.option('--env', type=click.Choice(['simple', 'simple_speaker_listener',
                                          'simple_crypto', 'simple_push',
                                          'simple_tag', 'simple_spread', 'simple_adversary', 'ma_gym:HuRoSorting-v0', 'assistive_gym:FeedingSawyerHuman-v0']))
@click.option('--lr', type=click.FLOAT, default=0.1)
@click.option('--seed', type=click.INT, default=1)
@click.option('--batch_size', type=click.INT, default=1000)
@click.option('--atlas', is_flag=True, flag_value=True)
def main(logdir, env, lr, seed, batch_size, atlas):
    env_ids = [env]
    lrs = [lr]
    seeds = [seed]
    batch_sizes = [batch_size]

    print('logging to: ' + logdir)

    for env_id, seed, lr, batch_size in itertools.product(env_ids, seeds, lrs, batch_sizes):
        train(logdir + '/exps/mack/' + env_id + '/l-{}-b-{}/seed-{}'.format(lr, batch_size, seed),
              env_id, 5e7, lr, batch_size, seed, batch_size // 1000) # change last arg to 250 instead of 1000 for 4 parallel processes


if __name__ == "__main__":
    main()
