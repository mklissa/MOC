# !/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
import gym, logging
from baselines import logger
from half_cheetah import *
from walker2d import *


def train(env_id, num_timesteps, seed, num_options, app, saves, wsaves, epoch, w_intfc, switch, mainlr, intlr, piolr, multi, eta):
    import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    
    if env_id=="AntWalls":
        from antwalls import AntWallsEnv
        env=AntWallsEnv()
        env.seed(seed) 
    else:
        env = gym.make(env_id)
        env._seed(seed)


    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2, num_options=num_options, w_intfc=w_intfc)

    gym.logger.setLevel(logging.WARN)

    if not multi:
        if num_options ==1:
            optimsize=64
        elif num_options ==2:
            optimsize=32
        else:
            optimsize=int(64/num_options)
    else:
        optimsize=64


    num_timesteps = num_timesteps
    tperbatch = 2048 if not epoch else int(1e4)
    pposgd_simple.learn(env, policy_fn, 
            max_timesteps=num_timesteps,
            timesteps_per_batch=tperbatch,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=mainlr, optim_batchsize=optimsize,
            gamma=0.99, lam=0.95, schedule='constant', num_options=num_options,
            app=app, saves=saves, wsaves=wsaves, epoch=epoch, seed=seed,
            w_intfc=w_intfc,switch=switch,intlr=intlr,piolr=piolr,multi=multi,eta=eta
        )
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='AntWalls')
    parser.add_argument('--timesteps', help='number of timesteps', type=int, default=2e6) 
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--opt', help='number of options', type=int, default=2) 
    parser.add_argument('--app', help='Append to folder name', type=str, default='')        
    parser.add_argument('--saves', help='Save the returns at each iteration', dest='saves', action='store_true', default=False)
    parser.add_argument('--wsaves', help='Save the weights',dest='wsaves', action='store_true', default=False)    
    parser.add_argument('--switch', help='Switch task after 150 iterations', dest='switch', action='store_true', default=False)    
    parser.add_argument('--nointfc', help='Disables interet functions', dest='w_intfc', action='store_false', default=True)    
    parser.add_argument('--epoch', help='Load weights from a certain epoch', type=int, default=0) 
    parser.add_argument('--mainlr', type=float, default=1e-4)
    parser.add_argument('--intlr', type=float, default=1e-4)
    parser.add_argument('--piolr', type=float, default=1e-4)
    parser.add_argument('--multi', help='Multi updates', dest='multi', action='store_true', default=False)  
    parser.add_argument('--eta', type=float, default=0.1, help='trade off updates')




    args = parser.parse_args()

    train(args.env, num_timesteps=args.timesteps, seed=args.seed, num_options=args.opt, app=args.app,
     saves=args.saves, wsaves=args.wsaves, epoch=args.epoch,w_intfc=args.w_intfc,
     switch=args.switch,mainlr=args.mainlr,intlr=args.intlr,piolr=args.piolr,multi=args.multi, eta=args.eta)


if __name__ == '__main__':
    main()
