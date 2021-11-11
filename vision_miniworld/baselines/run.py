import sys
import re
import multiprocessing
import os.path as osp
import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np

from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module


try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def train(args, extra_args):
    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type))


    total_timesteps = int(args.num_timesteps)
    seed = args.seed
    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)


    env = build_env(args)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    if env_type=='bsuite':
        alg_kwargs.update({'total_episodes':env.envs[0]._env.bsuite_num_episodes})

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed
    env_type, env_id = get_env_type(args)

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)

    else:
        if env_type == 'miniworld':
            frame_stack_size = 1
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)


        if env_type == 'bsuite':
            import bsuite
            from baselines.common.vec_env import dummy_vec_env
            from bsuite.utils import gym_wrapper


            SAVE_PATH = '/tmp/bsuite/ppo'

            def _load_env():
              raw_env = bsuite.load_and_record(
                  bsuite_id=env_id, 
                  save_path=SAVE_PATH, logging_mode='csv', overwrite=True)
              return gym_wrapper.GymFromDMEnv(raw_env)
            env = dummy_vec_env.DummyVecEnv([_load_env])
        else:
            config = tf.ConfigProto(allow_soft_placement=True,
                                   intra_op_parallelism_threads=1,
                                   inter_op_parallelism_threads=1)
            config.gpu_options.allow_growth = True
            get_session(config=config)

            flatten_dict_observations = alg not in {'her'}
            env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)
            
            if env_type == 'mujoco':
                env = VecNormalize(env, use_tf=True)

    return env


def get_env_type(args):
    env_id = args.env


    if args.env_type is not None:
        return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    elif 'bsuite' in env_id:
        env_type='bsuite'
        env_id=env_id[7:]
    elif 'MiniWorld' in env_id:
        env_type='miniworld'
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro','miniworld'}:
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs



def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])
    
    model, env = train(args, extra_args)

    
    # if args.save_path is not None and rank == 0:
        # save_path = osp.expanduser(args.save_path)
    if rank == 0:
        # save_path = osp.join(args.log_path, 'checkpoints','final')
        checkdir= logger.get_dir().replace("results","checkpoints")
        save_path = osp.join(checkdir, 'final')
        model.save(save_path)

    if args.play:
        logger.log("Running trained model")
        obs = env.reset()
        

        if args.alg == 'ppo2_options':
            option, _, _, _, _, _,\
                 _, _, _ = model.option_step(obs)
        else:
            option=None

        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))
        force_option=0
        option=np.array([0])
        episode_rew = np.zeros(env.num_envs) if isinstance(env, VecEnv) else np.zeros(1)
        while True:
            time.sleep(0.03)
            print(option)
            if state is not None:
                actions, _, state, _ = model.step(obs,S=state, M=dones)
            else:
                if args.alg == 'ppo2_options':
                    actions, _, _, _, _ = model.step(obs, sampled_options=option)
                else:
                    actions, _, _, _ = model.step(obs)

            obs, rew, done, _ = env.step(actions)

            if args.alg == 'ppo2_options' and not force_option:
                next_option, neglogop, term, _, _, tp,\
                     curr_o_p, _, _ = model.option_step(obs,sampled_options=option)
                
                if term[0]:
                    option=next_option
                elif done[0]:
                    option=next_option

            episode_rew += rew
            env.render()
            done_any = done.any() if isinstance(done, np.ndarray) else done
            if done_any:
                for i in np.nonzero(done)[0]:
                    print('episode_rew={}'.format(episode_rew[i]))
                    episode_rew[i] = 0

    env.close()

    return model

if __name__ == '__main__':
    main(sys.argv)
