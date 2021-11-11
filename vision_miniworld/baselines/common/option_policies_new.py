import tensorflow as tf
from baselines.common import tf_util
from baselines.a2c.utils import fc
from baselines.common.distributions import make_pdtype,fc_options
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.tf_util import adjust_shape
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.models import get_network_builder
from baselines.a2c.utils import fc, ortho_init

import gym
from gym import spaces
import numpy as np

class PolicyWithValue(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, env, observations, latent,  vf_latent, op_latent, term_latent, estimate_q=False, noptions=1, sess=None, **tensors):
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        latent          latent state from which policy distribution parameters should be inferred

        vf_latent       latent state from which value function should be inferred (if None, then latent is used)

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask

        """

        self.X = observations
        self.state = tf.constant([])
        self.initial_state = None
        self.__dict__.update(tensors)
        self.noptions = noptions


        # Based on the action space, will select what probability distribution type
        latent = tf.layers.flatten(latent)
        self.actions = []
        self.neglogps = []     
        self.action_dists = []   
        self.pdtype = make_pdtype(env.action_space,options=True)
        self.pds, self.pis = self.pdtype.pdfromlatent(latent, noptions=noptions, init_scale=0.01)
        for pd in self.pds:
            # Take an action
            action = pd.sample()
            # Calculate the neg log of our probability
            neglogp = pd.neglogp(action)
            self.actions.append(action)
            self.neglogps.append(neglogp)


        self.sess = sess or tf.get_default_session()


        # Calcualte value functions
        vf_latent = tf.layers.flatten(vf_latent)
        self.allvf = []
        for o in range(noptions):
            self.allvf.append( tf.squeeze(fc(vf_latent, 'vf_{}'.format(o), 1),axis=1) )  

        
        # Calculate policy over options
        op_latent = tf.layers.flatten(op_latent)
        # op_latent = tf.stop_gradient(op_latent) #optional gradient stop
        self.pdtype_op = make_pdtype(spaces.Discrete(noptions))
        self.pd_op, self.pi_op = self.pdtype_op.pdfromlatent(op_latent, init_scale=0.01,name='pi_op')
        # # Take an option
        self.option = self.pd_op.sample()
        # # Calculate the neg log of our probability
        self.neglogp_op = self.pd_op.neglogp(self.option)
        self.op_dist = self.pd_op.mean

        # import pdb;pdb.set_trace()
        #self.intfc = tf.sigmoid(fc(op_latent, 'pi_intfc', noptions, init_scale=0.01, init_bias=0.0))
        #self.op_dist = tf.nn.softmax(fc(op_latent, 'pi_op', noptions, init_scale=0.01, init_bias=0.0))
        # self.I = tf.stop_gradient(self.intfc)
        #self.pi_I = (self.intfc * self.op_dist) / tf.expand_dims(tf.reduce_sum( self.intfc * self.op_dist , axis=-1),axis=1)
        #self.pi_I_logits = tf.log( self.pi_I )
        # u = tf.random_uniform(tf.shape(self.pi_I_logits), dtype=self.pi_I_logits.dtype)
        # self.option = tf.argmax(self.pi_I_logits - tf.log(-tf.log(u)), axis=-1)
        #self.pdtype_op = make_pdtype(spaces.Discrete(noptions))
        #self.pd_op= self.pdtype_op.pdfromflat(self.pi_I_logits)
        # Take an option
        #self.option = self.pd_op.sample()
        # Calculate the neg log of our probability
        #self.neglogp_op = self.pd_op.neglogp(self.option)
        # self.op_dist = self.pd_op.mean




        #Calculate the termination function
        term_latent = tf.layers.flatten(term_latent)
        # term_latent = tf.stop_gradient(term_latent) #optional gradient stop

        # self.terms = []
        # for o in range(noptions):
        #     self.terms.append( tf.squeeze(tf.nn.sigmoid(fc(term_latent, 'term_{}'.format(o), 1)),axis=1)  )

        self.terms = []
        self.neglogps_term = []
        self.terms_dist = []
        self.pdtype_term = make_pdtype(spaces.Discrete(2),options=True) # hard code 2 since either termination or not
        self.pds_term, self.pis_term = self.pdtype_term.pdfromlatent(term_latent, noptions=noptions, init_scale=0.01, name='term')
        for pd_t in self.pds_term:
            # Take an termination action
            term = pd_t.sample()
            # Calculate the neg log of our probability
            neglogp_term = pd_t.neglogp(term)
            self.terms.append(term)
            self.neglogps_term.append(neglogp_term)
            self.terms_dist.append(pd_t.mean)


        

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def step(self, observation, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """
        
        acts, qs, state, neglogps   = self._evaluate([self.actions, self.allvf, self.state, self.neglogps], observation, **extra_feed)
        opts = extra_feed['sampled_options']
        a = np.array(acts)[opts, range(len(observation))]
        neglogp = np.array(neglogps)[opts, range(len(observation))]
        q = np.array(qs)[opts, range(len(observation))]
        if state.size == 0:
            state = None
        log_action_ratios = neglogp - np.array(neglogps)
        return a, q, state, neglogp, log_action_ratios

    def option_step(self, observation, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """
        
        o, neglogp_op, ts, qs, op, ts_p, neglogpts = self._evaluate([self.option, self.neglogp_op, self.terms, self.allvf, self.op_dist, self.terms_dist, self.neglogps_term], observation, **extra_feed)
        opts = o if 'sampled_options' not in extra_feed else extra_feed['sampled_options'] # Only used for the first timestep
        v = (np.array(qs).T * op).sum(axis=1)
        t = np.array(ts)[opts, range(len(observation))]
        tp = np.array(ts_p)[opts, range(len(observation))][:,1]
        neglogpt = np.array(neglogpts)[opts, range(len(observation))]            
        arrival_q = np.array(qs)[opts, range(len(observation))]



        n_envs = op.shape[0]
        ts_p= np.array(ts_p)
        prev_options_ts_p = np.zeros_like(ts_p)
        prev_options_ts_p[:,np.arange(n_envs),:] = ts_p[opts,np.arange(n_envs),:]

        prev_options_one_hot = np.zeros((self.noptions,op.shape[0])) 
        prev_options_one_hot[opts,np.arange(n_envs)] =1.
        one_hot_and_curr_op = np.stack((prev_options_one_hot,op.T),axis=2)
        curr_opt_prob = (one_hot_and_curr_op * prev_options_ts_p).sum(axis=-1)


        log_option_ratios_with_done = np.zeros_like(curr_opt_prob) # placeholder

        if 'dones' in extra_feed:
            dones=extra_feed['dones'].astype(int)
            previous_and_candidate_opt = np.stack((extra_feed['sampled_options'],o))
            next_options = previous_and_candidate_opt[t, np.arange(len(t))]
            next_op_probs = op[np.arange(len(next_options)),next_options]
            next_op_probs_stack = np.tile(next_op_probs,(self.noptions,1))

            next_options_one_hot = np.zeros((self.noptions,n_envs)) 
            next_options_one_hot[next_options,np.arange(len(next_options))] =1.
            one_hot_and_next_op = np.stack((next_options_one_hot,next_op_probs_stack),axis=2)
            next_opt_prob = (one_hot_and_next_op * ts_p).sum(axis=-1)
            

            log_option_ratios = np.log(next_opt_prob) - np.log(next_opt_prob[next_options,np.arange(n_envs)])
            done_ratios = np.zeros_like(log_option_ratios)
            log_ratios_and_done_ratios = np.stack((log_option_ratios,done_ratios),axis=2)
            log_option_ratios_with_done = log_ratios_and_done_ratios[:,np.arange(n_envs),dones]



        
        return o, neglogp_op, t, arrival_q, v, tp, neglogpt, curr_opt_prob, log_option_ratios_with_done



    def value(self, observation, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        value estimate
        """
        qs = self._evaluate(self.allvf, observation, *args, **kwargs)
        opts = kwargs['sampled_options']
        q = np.array(qs)[opts,range(len(observation))]
        return q

    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)

def build_policy(env, policy_network, value_network=None, term_network=None, op_network=None,  normalize_observations=False, estimate_q=False, **policy_kwargs):
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_network_builder(network_type)(**policy_kwargs)

    def policy_fn(nbatch=None, nsteps=None, noptions=1, sess=None, observ_placeholder=None):
        ob_space = env.observation_space
        
        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch)

        extra_tensors = {}

        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X

        encoded_x = encode_observation(ob_space, encoded_x)

        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            policy_latent = policy_network(encoded_x)
            if isinstance(policy_latent, tuple):
                policy_latent, recurrent_tensors = policy_latent

                if recurrent_tensors is not None:
                    # recurrent architecture, need a few more steps
                    nenv = nbatch // nsteps
                    assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
                    policy_latent, recurrent_tensors = policy_network(encoded_x, nenv)
                    extra_tensors.update(recurrent_tensors)


        _v_net = value_network

        if _v_net is None or _v_net == 'shared':
            vf_latent = policy_latent
        else:
            if _v_net == 'copy':
                _v_net = policy_network
            else:
                assert callable(_v_net)

            with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
                # TODO recurrent architectures are not supported with value_network=copy yet
                vf_latent = _v_net(encoded_x)


        _t_net = term_network

        if _t_net is None or _t_net == 'shared':
            term_latent = policy_latent
        else:
            if _t_net == 'copy':
                _t_net = policy_network
            else:
                assert callable(_t_net)

            with tf.variable_scope('term', reuse=tf.AUTO_REUSE):
                # TODO recurrent architectures are not supported with term_network=copy yet
                term_latent = _t_net(encoded_x)



        _op_net = op_network

        if _op_net is None or _op_net == 'shared':
            op_latent = policy_latent
        else:
            if _op_net == 'copy':
                _op_net = policy_network
            else:
                assert callable(_op_net)

            with tf.variable_scope('op', reuse=tf.AUTO_REUSE):
                # TODO recurrent architectures are not supported with op_network=copy yet
                op_latent = _op_net(encoded_x)
        
        policy = PolicyWithValue(
            env=env,
            observations=X,
            latent=policy_latent,
            vf_latent=vf_latent,
            op_latent=op_latent, 
            term_latent=term_latent,
            sess=sess,
            estimate_q=estimate_q,
            noptions=noptions,
            **extra_tensors
        )
        return policy

    return policy_fn


def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms

