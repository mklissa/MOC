import tensorflow as tf
import functools
import numpy as np

from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.tf_util import initialize

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None

class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, noptions=1, mpi_rank_weight=1, comm=None, microbatch_size=None, eta=0.0,op_ent_coeff=0.001):
        self.sess = sess = get_session()
        self.eta=eta

        if MPI is not None and comm is None:
            comm = MPI.COMM_WORLD

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, noptions=noptions, sess=sess)

            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, nsteps, noptions=noptions, sess=sess)
            else:
                train_model = policy(microbatch_size, nsteps, noptions=noptions, sess=sess)

        # CREATE THE POLICY/VALUE PLACEHOLDERS
        self.A = A = train_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])


        self.ALL_OPTIONS = ALL_OPTIONS = []
        self.ALL_VF_OPTIONS = ALL_VF_OPTIONS = []
        self.CURR_O_PROB = CURR_O_PROB = []
        self.LOG_RATIO = LOG_RATIO = []
        self.NOT_START_EP = NOT_START_EP = []
        for o in range(noptions):
            ALL_OPTIONS.append(tf.placeholder(tf.int32, [None]) )
            ALL_VF_OPTIONS.append(tf.placeholder(tf.int32, [None]) )
            CURR_O_PROB.append(tf.placeholder(tf.float32, [None]))
            LOG_RATIO.append(tf.placeholder(tf.float32, [None]))
            NOT_START_EP.append(tf.placeholder(tf.int32, [None]) )
        

        # CREATE THE PLACEHOLDERS for Policy Over Options 
        # self.O = O = train_model.pdtype_op.sample_placeholder([None])
        self.O = O = tf.placeholder(tf.int32, [None])
        self.ADV_OP = ADV_OP = tf.placeholder(tf.float32, [None])
        self.BETAS = BETAS = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        self.OLDNEGLOGPOP = OLDNEGLOGPOP = tf.placeholder(tf.float32, [None])


        # CREATE THE PLACEHOLDERS for Termination function
        self.T = T = train_model.pdtype.sample_placeholder([None])
        self.ADV_T = ADV_T = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        self.OLDNEGLOGP_T = OLDNEGLOGP_T = tf.placeholder(tf.float32, [None])
        

        neglogpop = train_model.pd_op.neglogp(O)
        ratio_op = tf.exp(OLDNEGLOGPOP - neglogpop)
        # Defining Loss = - J is equivalent to max J
        pg_losses_op = -ADV_OP * ratio_op
        pg_losses2_op = -ADV_OP * tf.clip_by_value(ratio_op, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        # Final PG loss
        pg_loss_op = tf.reduce_mean(tf.maximum(pg_losses_op, pg_losses2_op))
        entropy_op = tf.reduce_mean(train_model.pd_op.entropy())


        loss= pg_loss_op - entropy_op * op_ent_coeff

        
        # self.stats_list = []
        for i,(pd,pd_t) in enumerate(zip(train_model.pds,train_model.pds_term)):
            
            # Calculate ratio (pi current policy / pi old policy)
            neglogpac = tf.gather( pd.neglogp(A), ALL_OPTIONS[i])
            OLDNEGLOGPAC_O = tf.gather(OLDNEGLOGPAC, ALL_OPTIONS[i])
            ADV_O = tf.gather(ADV, ALL_OPTIONS[i])

            ratio = tf.exp(OLDNEGLOGPAC_O - neglogpac + LOG_RATIO[i])
            # Defining Loss = - J is equivalent to max J
            pg_losses = -ADV_O * ratio
            pg_losses2 = -ADV_O * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
            # Final PG loss
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2)  * CURR_O_PROB[i] )
            entropy = tf.gather(pd.entropy(), ALL_VF_OPTIONS[i])
            entropy = tf.reduce_mean(entropy)



            # Clip the value to reduce variability during Critic training
            # Get the predicted value
            vpred = tf.gather( train_model.allvf[i], ALL_VF_OPTIONS[i]) 
            OLDVPRED_O = tf.gather(OLDVPRED, ALL_VF_OPTIONS[i])
            R_O = tf.gather(R, ALL_VF_OPTIONS[i])

            vpredclipped = OLDVPRED_O + tf.clip_by_value(vpred - OLDVPRED_O, - CLIPRANGE, CLIPRANGE)
            # Unclipped value
            vf_losses1 = tf.square(vpred - R_O)
            # Clipped value
            vf_losses2 = tf.square(vpredclipped - R_O)
            vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2)) 


            #Termination loss
            ADV_T_O = tf.gather( ADV_T, NOT_START_EP[i] )

            # Calculate ratio (pi current policy / pi old policy)
            neglogp_t = tf.gather( pd_t.neglogp(T), NOT_START_EP[i])
            OLDNEGLOGP_T_O = tf.gather(OLDNEGLOGP_T, NOT_START_EP[i])

            ratio_t = tf.exp(OLDNEGLOGP_T_O - neglogp_t)
            # Defining Loss = - J is equivalent to max J
            term_losses = -ADV_T_O * ratio_t
            term_losses2 = -ADV_T_O * tf.clip_by_value(ratio_t, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
            # Final term loss
            term_loss = tf.reduce_mean(tf.maximum(term_losses, term_losses2))
            entropy_t = tf.gather(pd_t.entropy(), NOT_START_EP[i])
            entropy_t = tf.reduce_mean(entropy_t)


            # Total loss
            loss += (pg_loss - entropy * ent_coef + vf_loss * vf_coef + term_loss - entropy_t * 0.001)



        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params = tf.trainable_variables('ppo2_model')
        # 2. Build our trainer
        if comm is not None and comm.Get_size() > 1:
            self.trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)
        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da



        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)


        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.option_step = act_model.option_step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.noptions = noptions

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)


        initialize()
        
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables, comm=comm) #pylint: disable=E1101

    def train(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, neglogpops, options, advs_op, terms, neglogpts, advs_term, terms_p, curr_opt_prob, log_action_ratios, log_option_ratios, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        advs_op = (advs_op - advs_op.mean()) / (advs_op.std() + 1e-8)
        # advs_term = (advs_term - advs_term.mean()) / (advs_term.std() + 1e-8)


        # options=np.zeros_like(options)
        td_map = {
            self.train_model.X : obs,
            self.A : actions,
            self.O : options,
            self.T : terms,
            self.ADV : advs,
            self.ADV_OP : advs_op,
            self.ADV_T : advs_term,
            self.BETAS : terms_p,
            self.R : returns,
            self.LR : lr,
            self.CLIPRANGE : cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDNEGLOGPOP : neglogpops,
            self.OLDNEGLOGP_T : neglogpts,
            self.OLDVPRED : values,
        }

        # Determine where to train what
        sampled_eta=np.random.rand() < self.eta
        if sampled_eta:
            for o in range(self.noptions):
                minval=np.min(log_option_ratios[:,o] + log_action_ratios[:,o])
                options_indices = np.where(options == o)[0]
                # Indicate which index refers to which option
                td_map[self.ALL_OPTIONS[o]] = list(range(len(options)))
                td_map[self.ALL_VF_OPTIONS[o]] = options_indices
                td_map[self.CURR_O_PROB[o]] = curr_opt_prob[:,o]
                td_map[self.LOG_RATIO[o]] = log_option_ratios[:,o] + log_action_ratios[:,o]

                # Check where there was episode reset, used for termination function
                masks_o = masks[options_indices]
                masks_o_inds= np.where(1. -  masks_o)[0] 
                td_map[self.NOT_START_EP[o]] = options_indices[masks_o_inds]
        else:
            for o in range(self.noptions):
                options_indices = np.where(options == o)[0]
                # Indicate which index refers to which option
                td_map[self.ALL_OPTIONS[o]] = options_indices
                td_map[self.ALL_VF_OPTIONS[o]] = options_indices
                td_map[self.CURR_O_PROB[o]] = np.ones_like(curr_opt_prob[options_indices,o])
                td_map[self.LOG_RATIO[o]] = np.zeros_like(log_option_ratios[options_indices,o])

                # Check where there was episode reset, used for termination function
                masks_o = masks[options_indices]
                masks_o_inds= np.where(1. -  masks_o)[0] 
                td_map[self.NOT_START_EP[o]] = options_indices[masks_o_inds]
            

        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

        return self.sess.run(
            self._train_op,
            td_map
        )


