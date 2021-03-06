import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from baselines.ppo2 import ppo2
from baselines.ppo2_options import ppo2_options
from baselines.ppo2_options_new import ppo2_options as ppo2_options_new
from baselines.common.models import build_impala_cnn,nature_cnn
from baselines.common.mpi_util import setup_mpi_gpus
from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
from baselines import logger
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
import argparse

from baselines.common import set_global_seeds
from baselines.run import parse_cmdline_kwargs


def main():
    num_envs = 64
    learning_rate = 5e-4
    ent_coef = .01
    gamma = .999
    lam = .95
    nsteps = 256
    nminibatches = 8
    ppo_epochs = 3
    clip_range = .2

    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='coinrun')
    parser.add_argument('--distribution_mode', type=str, default='easy', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=200)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--log_path', type=str, default='/tmp/procgen')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timesteps_per_proc', type=float, default=25e6)
    parser.add_argument('--algo', type=str, default='ppo2')
   
    args,unk_args = parser.parse_known_args()
    unk_args = parse_cmdline_kwargs(unk_args)
    
    set_global_seeds(args.seed)
    test_worker_interval = args.test_worker_interval
    timesteps_per_proc = int(args.timesteps_per_proc)

    comm = None #MPI.COMM_WORLD
    #rank = comm.Get_rank()

    is_test_worker = False

    #if test_worker_interval > 0:
    #    is_test_worker = comm.Get_rank() % test_worker_interval == (test_worker_interval - 1)

    mpi_rank_weight = 1#0 if is_test_worker else 1
    num_levels = args.num_levels #0 if is_test_worker else args.num_levels


    #log_comm = comm.Split(1 if is_test_worker else 0, 0)
    format_strs = ['csv', 'stdout'] #if log_comm.Get_rank() == 0 else []

    logger.configure(dir=args.log_path, format_strs=format_strs)

    logger.info("creating environment")
    venv = ProcgenEnv(num_envs=num_envs, env_name=args.env_name, num_levels=num_levels, start_level=args.start_level, distribution_mode=args.distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100,)
    venv = VecNormalize(venv=venv, ob=False)

    logger.info("creating eval environment")
    eval_venv = ProcgenEnv(num_envs=num_envs, env_name=args.env_name, num_levels=0, start_level=args.start_level, distribution_mode=args.distribution_mode)
    eval_venv = VecExtractDictObs(eval_venv, "rgb")
    eval_venv = VecMonitor(venv=eval_venv, filename=None, keep_buf=100,    )
    eval_venv = VecNormalize(venv=eval_venv, ob=False)
    # eval_venv=None
    print("created gym environment")

    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)
    # conv_fn = lambda x: nature_cnn(x)


    logger.info("training on {}".format(args.algo))
    if args.algo == 'ppo2':
        algo = ppo2 
    elif args.algo == 'ppo2_options':
        algo = ppo2_options
    else:
        algo = ppo2_options_new
    algo.learn(
        env=venv,
        eval_env=eval_venv,
        network=conv_fn,
        total_timesteps=timesteps_per_proc,
        save_interval=0,
        nsteps=nsteps,
        nminibatches=nminibatches,
        lam=lam,
        gamma=gamma,
        noptepochs=ppo_epochs,
        log_interval=1,
        ent_coef=ent_coef,
        mpi_rank_weight=mpi_rank_weight,
        comm=comm,
        lr=learning_rate,
        cliprange=clip_range,
        update_fn=None,
        init_fn=None,
        vf_coef=0.5,
        max_grad_norm=0.5,
        eval_interval=2,
        **unk_args
    )

if __name__ == '__main__':
    main()

