from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
import numpy as np



def dense3D2(x, size, name, option, num_options=1, weight_init=None, bias=True):
    w = tf.get_variable(name + "/w", [num_options, x.get_shape()[1], size], initializer=weight_init)
    ret = tf.matmul(x, w[option[0]])
    if bias:
        b = tf.get_variable(name + "/b", [num_options,size], initializer=tf.zeros_initializer())
        return ret + b[option[0]]

    else:
        return ret


class MlpPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True, num_options=2, w_intfc=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.w_intfc =w_intfc
        self.state_in = []
        self.state_out = []
        self.num_options = num_options
        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        option =  U.get_placeholder(name="option", dtype=tf.int32, shape=[None])


        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        self.vpred = dense3D2(last_out, 1, "vffinal", option, num_options=num_options, weight_init=U.normc_initializer(1.0))[:,0]
        

        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "termfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        self.tpred = tf.sigmoid(U.dense(last_out, num_options, "termhead", weight_init=U.normc_initializer(1.0)))

        


        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "polfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = dense3D2(last_out, pdtype.param_shape()[0]//2, "polfinal", option, num_options=num_options, weight_init=U.normc_initializer(0.01))
            logstd = tf.get_variable(name="logstd", shape=[num_options, 1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            pdparam = U.concatenate([mean, mean * 0.0 + logstd[option[0]]], axis=1)
        else:
            pdparam = U.dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))
        self.pd = pdtype.pdfromflat(pdparam)
        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        sampled_ac = self.pdtype.sample_placeholder([None])
        logps = self.pd.logp(sampled_ac)

        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "intfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        self.intfc = tf.sigmoid(U.dense(last_out, num_options, "intfcfinal", weight_init=U.normc_initializer(1.0)))

        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "OP%i"%(i+1), weight_init=U.normc_initializer(1.0))) 
        self.op_pi = tf.nn.softmax(U.dense(last_out, num_options, "OPfinal", weight_init=U.normc_initializer(1.0)))



        self._act = U.function([stochastic, ob, option], [ac])
        self._logps = U.function([stochastic, ob, option, sampled_ac], [logps])
        self.get_tpreds = U.function([ob], [self.tpred])
        self.get_vpred = U.function([ob, option], [self.vpred])        
        self._get_op_int = U.function([ob], [self.op_pi,self.intfc])
        self._get_intfc = U.function([ob], [self.intfc])
        self._get_op = U.function([ob], [self.op_pi])


    def get_tpred(self,ob,option):
        tpreds = self.get_tpreds(ob)[0]
        return tpreds[0,option]

    def get_term(self,ob,option):
        tpred = self.get_tpred(ob,option)
        term = int(np.random.uniform() < tpred)
        return term

    def act(self, stochastic, ob, option):
        ac1 =  self._act(stochastic, ob[None], [option])
        return ac1[0]

    def ac_probs(self, ob, ac):
        probs=[]
        for o in range(self.num_options):
            prob =  np.exp(self._logps(True, ob[None],[o], ac)[0][0])
            probs.append(prob)
        return probs

    def get_int_func(self,obs):
        return self._get_intfc(obs)[0]

    def get_alltpreds(self,obs,ob):
        obs=np.vstack((obs,ob[None]))
        betas = self.get_tpreds(obs)[0]
        return betas[:-1],betas[-1], betas

    def get_allvpreds(self,obs,ob):
        obs=np.vstack((obs,ob[None]))

        # Get Q(s,w)
        vals = []
        for opt in range(self.num_options):
            vals.append(self.get_vpred(obs,[opt])[0])
        vals=np.array(vals).T

        op_prob,int_func = self._get_op_int(obs)
        if self.w_intfc:
            pi_I = op_prob * int_func / np.sum(op_prob * int_func,axis=1)[:,None]     
        else:
            pi_I = op_prob
        op_vpred = np.sum((pi_I*vals),axis=1) # Get V(s)
        
        return vals[:-1], op_vpred[:-1],vals[-1],op_vpred[-1], op_prob[:-1], int_func[:-1], pi_I

    def get_vpreds(self,obs):
        vals = []
        for opt in range(self.num_options):
            vals.append(self.get_vpred(obs,[opt])[0])
        vals=np.array(vals).T
        return vals 

    def get_option(self,ob):
        
        op_prob,int_func = self._get_op_int([ob])
        activated_options = []
        for int_val in int_func[0]:
            activated_options.append(1.)   
        indices=(-int_func[0]).argsort()[:2]
        if 1. not in activated_options:
            for i in indices:
                activated_options[i] = 1.

        if self.w_intfc:
            pi_I = op_prob * (activated_options*int_func) / np.sum(op_prob * (activated_options*int_func),axis=1)[:,None]
        else:
            pi_I = op_prob  

        return np.random.choice(range(len(op_prob[0])), p=pi_I[0]),activated_options

    def get_op_prob(self,ob):
        op_prob,int_func = self._get_op_int([ob])
        if self.w_intfc:
            pi_I = op_prob * int_func / np.sum(op_prob * int_func,axis=1)[:,None]
        else:
            pi_I = op_prob       
        return pi_I  


    def get_intvals(self,ob):
        op_prob,int_func = self._get_op_int([ob])
        return op_prob,int_func

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []
