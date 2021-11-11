from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time


from baselines.common.mpi_adam import MpiAdam
MPI=None
# from mpi4py import MPI
from collections import deque
import os
import shutil
from scipy import spatial
import gym
import matplotlib.pyplot as plt


def traj_segment_generator(pi, env, horizon, stochastic, num_options, saves, rewbuffer, epoch, seed, w_intfc, switch, gamma, eta):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()

    if hasattr(env,'NAME')and env.NAME=='AntWalls':
        switch_iter = 240
    elif env.spec.id == 'HalfCheetahDir-v1':
        switch_iter = 150
    elif env.spec.id == 'Walker2dStand-v1':
        switch_iter = 240


    render=0
    iters_so_far=0

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of completed episodes in this segment

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    realrews = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    opts = np.zeros(horizon, 'int32')
    activated_options = np.zeros((horizon, num_options), 'float32')
    last_options=np.zeros(horizon, 'int32')

    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()


    option,active_options_t = pi.get_option(ob)
    last_option=option


    ep_states=[[] for _ in range(num_options)] 
    ep_states[option].append(ob)
    ep_states_term=[[] for _ in range(num_options)] 
    ep_num =0
    while True:
        prevac = ac
        ac = pi.act(stochastic, ob, option)
        if render:
            option=1
            env.render()
            time.sleep(0.05)
            print(option)#,cur_ep_ret)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            vpreds, op_vpreds, vpred, op_vpred, op_probs, intfc, pi_I = pi.get_allvpreds(obs, ob)
            term_ps, term_p, all_term_ps = pi.get_alltpreds(obs, ob)
            last_betas=term_ps[range(len(last_options)),last_options]

            all_opts = np.append(opts,option)
            term_ratios=np.zeros((len(all_opts),num_options))
            for o in range(num_options):
                one_hot = np.zeros(len(all_opts))
                one_hot[np.where(all_opts==o)] = 1.
                term_ratios[:,o]=(all_term_ps[:,o] * pi_I[range(len(all_opts)),all_opts] + (1-all_term_ps[:,o]) * one_hot)
            term_ratios = np.log(term_ratios[1:]) - np.log(term_ratios[range(1,len(all_opts)),all_opts[:-1]][...,None])
            

            logps = np.zeros( (len(obs),num_options))
            for o in range(num_options):
                logps[:,o] = pi._logps(True,obs,[o],acs)[0]
            action_ratios = logps - logps[range(len(obs)), opts][...,None] 
            
            prev_action_ratios = np.vstack((action_ratios[0],action_ratios[:-1])) # a little bias here

            last_options_onehot = np.zeros((len(last_options),num_options))
            last_options_onehot[range(len(last_options)),last_options] = 1.
            prob_curr_opt = last_betas[...,None] * pi_I[:-1] + (1-last_betas[...,None]) * last_options_onehot
            prob_prev_opt = np.vstack((prob_curr_opt[0],prob_curr_opt[:-1])) # a little bias here


            sampled_eta = float(np.random.rand()<eta)
            options_onehot = np.zeros((len(opts),num_options))
            options_onehot[range(len(opts)),opts] = 1.
            prob_curr_opt = sampled_eta * prob_curr_opt + (1-sampled_eta) * options_onehot
            prob_prev_opt= sampled_eta * prob_prev_opt + (1-sampled_eta) * last_options_onehot



            yield {"ob" : obs, "rew" : rews, "realrew": realrews, "vpred" : vpreds, "op_vpred": op_vpreds, "new" : news,
                    "ac" : acs, "opts" : opts, "opt": option, "prevac" : prevacs, "nextvpred": vpred * (1 - new), "nextop_vpred": op_vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens, 'term_p': term_ps, 'next_term_p':term_p,
                     "op_probs":op_probs, "last_betas":last_betas, "intfc":intfc, 
                      "action_ratios": action_ratios, "term_ratios":term_ratios, "prev_action_ratios": prev_action_ratios,
                      "last_options": last_options, "last_option":last_option, "prob_curr_opt": prob_curr_opt, "prob_prev_opt":prob_prev_opt}


            ep_rets = []
            ep_lens = []
            iters_so_far+=1

            ###### Switching Goal ##########

            if iters_so_far==switch_iter and switch:
                # import pdb;pdb.set_trace()
                if hasattr(env,'NAME') and env.NAME=='AntWalls': # Switch the goal for AntWalls
                    from antwalls import AntWallsEnv
                    env=AntWallsEnv(num_walls=2)
                    env.seed(seed) 
                elif env.spec.id == 'HalfCheetahDir-v1':
                    env.env.env.reset_task({'direction':-1})
                elif env.spec.id == 'Walker2dStand2-v1':
                    env.env.reset_task('run')
                    
            ################################

        i = t % horizon
        obs[i] = ob
        last_options[i]=last_option

        news[i] = new
        opts[i] = option
        acs[i] = ac
        prevacs[i] = prevac
        activated_options[i] = active_options_t



        ## RL loop ##
        ob, rew, new, _ = env.step(ac) 
        rews[i] = rew
        realrews[i] = rew
        ## RL loop ##

        
        candidate_option,active_options_t = pi.get_option(ob)
        term = pi.get_term([ob],[option])
        last_option=option
        if term:
            ep_states_term[option].append(ob)
            option = candidate_option


        ep_states[option].append(ob)
        cur_ep_ret += rew
        cur_ep_len += 1

        
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0

            ep_num +=1
            ob = env.reset()
            option,active_options_t = pi.get_option(ob)
            last_option=option
            ep_states[option].append(ob)
        t += 1


def add_vtarg_and_adv(seg, gamma, lam, num_options):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    T = len(seg["rew"])
    arrival_options = np.append(seg["last_options"],seg["last_option"])
    opts = np.append(seg["opts"],seg["opt"])
    rew = seg["rew"]

    op_vpred = np.append(seg["op_vpred"], seg["nextop_vpred"])
    term_p = np.vstack((np.array(seg["term_p"]),np.array(seg["next_term_p"])))
    q_sw = np.vstack((seg["vpred"],seg["nextvpred"]))
    all_u_sw = (1-term_p) * q_sw + term_p * np.tile(op_vpred[:,None],num_options)
    u_sw = all_u_sw[range(len(all_u_sw)),arrival_options]
    
    
    seg["op_adv"] = gaelam = np.empty(T, 'float32')
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * u_sw[t+1] * nonterminal - u_sw[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam


    seg["adv"] = gaelam = np.empty(T, 'float32')
    vpred= q_sw[range(len(opts)),opts]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam

    seg["tdlamret"] = seg["adv"] + vpred[:-1]

    seg["term_adv"] = seg["vpred"] - np.tile(seg["op_vpred"][:,None],num_options)




def learn(env, policy_func, *,
        timesteps_per_batch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        num_options=1,
        app='',
        saves=False,
        wsaves=False,
        epoch=0,
        seed=1,
        w_intfc=True,switch=False,intlr=1e-4,piolr=1e-4,multi=False,eta=0.1,
        ):


    optim_batchsize_ideal = optim_batchsize 
    np.random.seed(seed)
    tf.set_random_seed(seed)
    


    ### Book-keeping
    if hasattr(env,'NAME'):
        gamename = env.NAME.lower() #change this for plots
    else:
        gamename = env.spec.id[:-3].lower()
    gamename += 'seed' + str(seed)

    ###


    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_func("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return
    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon



    prob_cur_opt = tf.placeholder(dtype=tf.float32, shape=[None]) # Probability of current option
    is_ratio = tf.placeholder(dtype=tf.float32, shape=[None]) # IS ratio for correcting off-policyness


    ob = U.get_placeholder_cached(name="ob")
    option = U.get_placeholder_cached(name="option")
    term_adv = U.get_placeholder(name='term_adv', dtype=tf.float32, shape=[None])
    op_adv = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    betas = tf.placeholder(dtype=tf.float32, shape=[None]) # Probability of termination (Used to weight meta-updates)
    oldvpred = tf.placeholder(tf.float32, [None])
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = U.mean(kloldnew)
    meanent = U.mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac) + is_ratio)
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg 
    pol_surr = - U.mean(tf.minimum(surr1, surr2)  * prob_cur_opt )  # PPO's pessimistic surrogate (L^CLIP)


    vf_loss = U.mean(tf.square(pi.vpred - ret) * tf.exp(is_ratio) * prob_cur_opt)

    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    # Loss for termination function
    option_hot = tf.one_hot(option,depth=num_options)
    term_loss= U.mean(( tf.reduce_sum(pi.tpred * option_hot, axis=1) * term_adv) )

    # Loss for interest function
    pi_w = tf.placeholder(dtype=tf.float32, shape=[None,num_options])
    pi_I = (pi.intfc ) * pi_w / tf.expand_dims(tf.reduce_sum((pi.intfc ) * pi_w,axis=1),1)
    pi_I = tf.clip_by_value(pi_I,1e-6,1-1e-6)
    int_loss = - tf.reduce_sum(betas *tf.reduce_sum(pi_I * option_hot,axis=1)    * op_adv)

    # Loss for policy over options
    intfc = tf.placeholder(dtype=tf.float32, shape=[None,num_options])
    pi_I = (intfc ) * pi.op_pi / tf.expand_dims(tf.reduce_sum( (intfc ) * pi.op_pi,axis=1),1)
    pi_I = tf.clip_by_value(pi_I,1e-6,1-1e-6)
    op_loss = - tf.reduce_sum(betas *tf.reduce_sum(pi_I * option_hot,axis=1)    * op_adv)
    log_pi = tf.log(tf.clip_by_value(pi.op_pi, 1e-20, 1.0))
    op_entropy = -tf.reduce_mean(pi.op_pi * log_pi, reduction_indices=1)
    op_loss -= 0.01*tf.reduce_sum(op_entropy)



    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult, option,  prob_cur_opt, is_ratio], losses + [U.flatgrad(total_loss, var_list)])
    termgrad = U.function([ob, option, term_adv], [U.flatgrad(term_loss, var_list)]) # Since we might use a different step size.
    opgrad = U.function([ob, option, betas, op_adv, intfc], [U.flatgrad(op_loss, var_list)]) # Since we might use a different step size.
    intgrad = U.function([ob, option, betas, op_adv, pi_w], [U.flatgrad(int_loss, var_list)]) # Since we might use a different step size.
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult, option], losses)


    U.initialize()
    adam.sync()


    saver = tf.train.Saver(max_to_keep=10000)




    episodes_so_far = 0
    timesteps_so_far = 0
    global iters_so_far
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=10) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=10) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"








    ######################################################### Prepare for rollouts #########################################################
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True, num_options=num_options,saves=saves,rewbuffer=rewbuffer,epoch=epoch,seed=seed,w_intfc=w_intfc,switch=switch,gamma=gamma,eta=eta)

    datas = [0 for _ in range(num_options)]

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)
        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam,num_options)


        ob, ac, opts, atarg, tdlamret, op_atarg  = seg["ob"], seg["ac"], seg["opts"], seg["adv"], seg["tdlamret"], seg["op_adv"] 
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy
        assign_old_eq_new() # set old parameter values to new parameter values


        

        min_batch=160 # Arbitrary
        for opt in range(num_options):
            
            

            if multi: ### multi-updates here ###
                inds = np.arange(len(ob))
                is_ratios =seg["action_ratios"] + seg["term_ratios"]
                is_ratios=is_ratios[:,opt]
                prob_curr_opt= seg["prob_curr_opt"][:,opt]
                d = Dataset(dict(ob=ob[inds], ac=ac[inds], atarg=atarg[inds], vtarg=tdlamret[inds],  prob_curr_opt=prob_curr_opt[inds], is_ratios=is_ratios[inds], oldvpred=seg["vpred"][inds,opt]), shuffle=not pi.recurrent)

                logger.log("Optimizing...")
                # Here we do a bunch of optimization epochs over the data
                for _ in range(optim_epochs):
                    losses = [] # list of tuples, each of which gives the loss for a minibatch
                    for batch in d.iterate_once(optim_batchsize):
                        *newlosses, grads = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult, [opt], batch["prob_curr_opt"], batch["is_ratios"])
                        adam.update(grads, optim_stepsize * cur_lrmult) 
                        losses.append(newlosses)

            else:
                indices = np.where(opts==opt)[0]
                print("batch size:",indices.size)
                if not indices.size:
                    continue

                if datas[opt] != 0:

                    if (indices.size < min_batch and datas[opt].n > min_batch):
                        datas[opt] = Dataset(dict(ob=ob[indices], ac=ac[indices], atarg=atarg[indices], vtarg=tdlamret[indices]), shuffle=not pi.recurrent)
                        continue

                    elif indices.size + datas[opt].n < min_batch:
                        oldmap = datas[opt].data_map

                        cat_ob = np.concatenate((oldmap['ob'],ob[indices]))
                        cat_ac = np.concatenate((oldmap['ac'],ac[indices]))
                        cat_atarg = np.concatenate((oldmap['atarg'],atarg[indices]))
                        cat_vtarg = np.concatenate((oldmap['vtarg'],tdlamret[indices]))
                        datas[opt] = Dataset(dict(ob=cat_ob, ac=cat_ac, atarg=cat_atarg, vtarg=cat_vtarg), shuffle=not pi.recurrent)
                        continue

                    elif (indices.size + datas[opt].n > min_batch and datas[opt].n < min_batch) or (indices.size > min_batch and datas[opt].n < min_batch):

                        oldmap = datas[opt].data_map
                        cat_ob = np.concatenate((oldmap['ob'],ob[indices]))
                        cat_ac = np.concatenate((oldmap['ac'],ac[indices]))
                        cat_atarg = np.concatenate((oldmap['atarg'],atarg[indices]))
                        cat_vtarg = np.concatenate((oldmap['vtarg'],tdlamret[indices]))
                        datas[opt] = d = Dataset(dict(ob=cat_ob, ac=cat_ac, atarg=cat_atarg, vtarg=cat_vtarg), shuffle=not pi.recurrent)

                    if (indices.size > min_batch and datas[opt].n > min_batch):
                        datas[opt] = d = Dataset(dict(ob=ob[indices], ac=ac[indices], atarg=atarg[indices], vtarg=tdlamret[indices]), shuffle=not pi.recurrent)

                elif datas[opt] == 0:
                    datas[opt] = d = Dataset(dict(ob=ob[indices], ac=ac[indices], atarg=atarg[indices], vtarg=tdlamret[indices]), shuffle=not pi.recurrent)



                optim_batchsize = optim_batchsize or ob.shape[0]

                logger.log("Optimizing...")
                # Here we do a bunch of optimization epochs over the data
                for _ in range(optim_epochs):
                    for batch in d.iterate_once(optim_batchsize):
                        *newlosses, grads = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult, [opt], np.ones_like(batch["vtarg"]), np.zeros_like(batch["vtarg"]))
                        adam.update(grads, optim_stepsize * cur_lrmult) 


        termg = termgrad(seg["ob"], seg['last_options'], seg["term_adv"][range(len(seg["last_options"])),seg["last_options"]] )[0]
        adam.update(termg, piolr)

        if w_intfc:
            intgrads = intgrad(seg['ob'],seg['opts'], seg["last_betas"], op_atarg, seg["op_probs"])[0]
            adam.update(intgrads, intlr)

        opgrads = opgrad(seg['ob'],seg['opts'], seg["last_betas"], op_atarg, seg["intfc"])[0]
        adam.update(opgrads, intlr)
        



        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs=[lrlocal]
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        logger.dump_tabular()




def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
