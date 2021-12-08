import gym
import argparse
import numpy as np
from fourrooms import Fourrooms
import random
import time
import os

from scipy.stats import entropy
from scipy.special import expit
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib 
from utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--discount', help='Discount factor', type=float, default=0.99)
    parser.add_argument('--lr_term', help="Termination gradient learning rate", type=float, default=0.8)
    parser.add_argument('--lr_intra', help="Intra-option gradient learning rate", type=float, default=0.8)
    parser.add_argument('--lr_critic', help="Learning rate", type=float, default=0.8)
    parser.add_argument('--nepisodes', help="Number of episodes per run", type=int, default=1000)
    parser.add_argument('--nruns', help="Number of runs", type=int, default=1)
    parser.add_argument('--nsteps', help="Maximum number of steps per episode", type=int, default=1000)
    parser.add_argument('--noptions', help='Number of options', type=int, default=4)
    parser.add_argument('--option_temperature', help="Temperature parameter for softmax", type=float, default=1e-1)
    parser.add_argument('--action_temperature', help="Temperature parameter for softmax", type=float, default=1e-2)
    parser.add_argument('--seed', help="seed", type=int, default=1)


    parser.add_argument('--multi_option', help="multi updates", default=False, action='store_true')
    parser.add_argument('--eta', help="Multi-updates hypereparameter", type=float, default=0.3)
    parser.add_argument('--new_randomness', help="new degree of randomness", type=float, default=0.45)

    args = parser.parse_args()
    eta=args.eta
    args.lr_term =args.lr_intra
    total_steps=0
    start=time.time()
    possible_next_goals = [74,75,84,85]
    history_steps = np.zeros((args.nruns, args.nepisodes))
    for run in range(args.nruns):
        env = gym.make('Fourrooms-v0')
        env.set_goal(62)
        env.set_seed(args.seed+run)


        np.random.seed(args.seed+run)
        random.seed(args.seed+run)

        features = Tabular(env.observation_space.n)
        nfeatures, nactions = len(features), env.action_space.n


        # The intra-option policies are linear-softmax functions
        option_policies = [SoftmaxPolicy(nfeatures, nactions, args.action_temperature) for _ in range(args.noptions)]

        # The termination function are linear-sigmoid functions
        option_terminations = [SigmoidTermination(nfeatures) for _ in range(args.noptions)]

        # Policy over options
        meta_policy = SoftmaxPolicy(nfeatures, args.noptions, args.option_temperature)

        # Different choices are possible for the critic. Here we learn an
        # option-value function and use the estimator for the values upon arrival
        critic = IntraOptionQLearning(args.discount, args.lr_critic, option_terminations, meta_policy.weights, meta_policy,args.noptions) 

        # Improvement of the termination functions based on gradients
        termination_improvement= TerminationGradient(option_terminations, critic, args.lr_term,args.noptions)

        # Intra-option gradient improvement with critic estimator
        intraoption_improvement = IntraOptionGradient(option_policies, args.lr_intra, args.discount, critic,args.noptions)




        tot_steps=0.
        for episode in range(args.nepisodes):
            if episode > 0 and episode == int(args.nepisodes/2.): ############################# Change time #############################
                goal=possible_next_goals[args.seed % len(possible_next_goals)]
                env.set_goal(goal)
                env.set_randomness(args.new_randomness)
                print('************* New goal : ', env.goal)



            
            last_opt=None
            phi = features(env.reset())
            option = meta_policy.sample(phi)
            action = option_policies[option].sample(phi)
            critic.start(phi, option)



            action_ratios_avg=[]
            
            for step in range(args.nsteps):
                observation, reward, done, _ = env.step(action)
                next_phi = features(observation)

                if option_terminations[option].sample(next_phi): 
                    next_option = meta_policy.sample(next_phi)
                else:
                    next_option=option

                next_action = option_policies[next_option].sample(next_phi)



                ###Action ratios
                action_ratios=np.zeros((args.noptions))
                for o in range(args.noptions):
                    action_ratios[o] = option_policies[o].pmf(phi)[action]
                action_ratios= action_ratios / action_ratios[option]
                action_ratios_avg.append(action_ratios)


                # Prob of current option
                one_hot = np.zeros(args.noptions)
                if last_opt is not None:
                    bet = option_terminations[last_opt].pmf(phi)
                    one_hot[last_opt] = 1.
                else:
                    bet = 1.0
                prob_curr_opt = bet * meta_policy.pmf(phi) + (1-bet)*one_hot
                one_hot_curr_opt= np.zeros(args.noptions)
                one_hot_curr_opt[option] = 1.
                sampled_eta = float(np.random.rand() < eta)
                prob_curr_opt= eta * prob_curr_opt + (1-eta) * one_hot_curr_opt


            
                # Critic updates
                critic.update(next_phi, next_option, reward, done, one_hot_curr_opt)


                # Intra-option policy update
                critic_feedback = reward + args.discount * critic.value(next_phi, next_option)
                critic_feedback -= critic.value(phi, option)
                if args.multi_option:
                    intraoption_improvement.update(phi, option, action, reward, done, next_phi, next_option, critic_feedback,
                        action_ratios, prob_curr_opt  )   
                else:
                    intraoption_improvement.update(phi, option, action, reward, done, next_phi, next_option, critic_feedback,
                        np.ones_like(action_ratios), one_hot_curr_opt  ) 

                # Termination update
                if not done:
                    termination_improvement.update(next_phi, option, one_hot_curr_opt )


                last_opt=option
                phi=next_phi
                option=next_option
                action=next_action


                if done:
                    break


            tot_steps+=step
            history_steps[run, episode] = step
            end=time.time()
            print('Run {} Total steps {} episode {} steps {} FPS {:0.0f} '.format(run,tot_steps, episode, step,   int(tot_steps/ (end- start)) )  )

