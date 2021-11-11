import numpy as np
from baselines.common.runners import AbstractEnvRunner

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam, noptions):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        # Initiate options
        self.options, self.neglogpops, self.terms, _, self.state_values,\
         self.terms_p, self.neglogpts, self.curr_opt_prob, self.log_option_ratios = self.model.option_step(self.obs)
        self.arrival_values = np.zeros_like(self.state_values) # Dummy values
        # self.terms_p = np.zeros_like(self.state_values) # Dummy values
        self.dones = [True for _ in range(self.nenv)] 
        self.noptions=noptions

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_svalues, mb_arrival_values, mb_dones, mb_neglogpacs,\
                            mb_neglogpops, mb_opts, mb_terms, mb_neglogpts, mb_terms_p, mb_curr_opt_prob,\
                             mb_log_action_ratios, mb_log_option_ratios = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
        mb_states = self.states
        epinfos = []

        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states,\
             neglogpacs, log_action_ratios  = self.model.step(self.obs, S=self.states, M=self.dones, sampled_options=self.options)

            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_log_action_ratios.append(log_action_ratios)
            mb_log_option_ratios.append(self.log_option_ratios)
            mb_dones.append(self.dones)
            mb_opts.append(self.options.copy())
            mb_neglogpops.append(self.neglogpops)
            mb_svalues.append(self.state_values)
            mb_arrival_values.append(self.arrival_values)   
            mb_terms.append(self.terms)
            mb_neglogpts.append(self.neglogpts)
            mb_terms_p.append(self.terms_p)
            mb_curr_opt_prob.append(self.curr_opt_prob)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            
            # Sample options
            next_options, self.neglogpops, self.terms, self.arrival_values, self.state_values, self.terms_p,\
             self.neglogpts, self.curr_opt_prob, self.log_option_ratios = self.model.option_step(self.obs, sampled_options=self.options,dones=self.dones)


            for e,(info,term) in enumerate(zip(infos,self.terms)):
                maybeepinfo = info.get('episode')
                if maybeepinfo: 
                    epinfos.append(maybeepinfo)
                    self.options[e] = next_options[e]
                elif term:
                    self.options[e] = next_options[e]


            mb_rewards.append(rewards)
        
        
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions, dtype=np.float32)
        mb_log_action_ratios = np.asarray(mb_log_action_ratios, dtype=np.float32)
        mb_log_option_ratios.append(self.log_option_ratios)
        mb_log_option_ratios = np.asarray(mb_log_option_ratios[1:], dtype=np.float32)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_svalues = np.asarray(mb_svalues, dtype=np.float32)
        mb_arrival_values = np.asarray(mb_arrival_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_neglogpops = np.asarray(mb_neglogpops, dtype=np.float32)
        mb_neglogpts = np.asarray(mb_neglogpts, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_opts = np.asarray(mb_opts)
        mb_terms = np.asarray(mb_terms)
        mb_terms_p = np.asarray(mb_terms_p)
        mb_curr_opt_prob = np.asarray(mb_curr_opt_prob, dtype=np.float32)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones, sampled_options=self.options)


        
        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values



        # discount/bootstrap off value fn for the policy over options
        mb_advs_op = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues =  self.terms_p * self.state_values + (1-self.terms_p) * self.arrival_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_terms_p[t+1] * mb_svalues[t+1] + (1-mb_terms_p[t+1]) * mb_arrival_values[t+1]
            baseline = mb_terms_p[t] * mb_svalues[t] + (1-mb_terms_p[t]) * mb_arrival_values[t]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - baseline
            mb_advs_op[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam


        mb_advs_term = mb_arrival_values - mb_svalues

        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions,\
         mb_values, mb_neglogpacs, mb_neglogpops, mb_opts, mb_advs_op,\
          mb_terms, mb_neglogpts, mb_advs_term, mb_terms_p,)),\
            mb_states, epinfos, \
             *map(sf12, (mb_curr_opt_prob, mb_log_action_ratios, mb_log_option_ratios)))

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])




def sf12(arr):
    """
    swap axes 1 and 2 and then swap and flatten axes 0 and 1
    """
    s = arr.swapaxes(1,2).shape
    return arr.swapaxes(1,2).swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


