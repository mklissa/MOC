import gym
import argparse
import numpy as np
from fourrooms import Fourrooms
import random

from scipy.special import expit
from scipy.special import logsumexp


class Tabular:
    def __init__(self, nstates):
        self.nstates = nstates

    def __call__(self, state):
        return np.array([state,])

    def __len__(self):
        return self.nstates

class EgreedyPolicy:
    def __init__(self, nfeatures, nactions, epsilon):
        self.epsilon = epsilon
        self.weights = np.zeros((nfeatures, nactions))

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def sample(self, phi):
        if np.random.uniform() < self.epsilon:
            return int(np.random.randint(self.weights.shape[1]))
        return int(np.argmax(self.value(phi)))




class SoftmaxPolicy:
    def __init__(self, nfeatures, nactions, temp=1.):
        self.weights = np.zeros((nfeatures, nactions))
        self.temp = temp

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def pmf(self, phi):
        v = self.value(phi)/self.temp
        return np.exp(v - logsumexp(v))

    def all_pmfs(self,):
        all_pmfs=[]
        for phi in range(len(self.weights)):
            v = self.value([phi])/self.temp
            all_pmfs.append(np.exp(v - logsumexp(v)))
        return np.array(all_pmfs)

    def sample(self, phi):
        return int(np.random.choice(self.weights.shape[1], p=self.pmf(phi)))

class SigmoidTermination:
    def __init__(self, nfeatures):
        self.weights = np.zeros((nfeatures,))

    def pmf(self, phi):
        return expit(np.sum(self.weights[phi]))

    def sample(self, phi):
        return int(np.random.uniform() < self.pmf(phi))

    def grad(self, phi):
        terminate = self.pmf(phi)
        return terminate*(1. - terminate), phi


class SigmoidInterestFunction:
    def __init__(self,):
        self.room0 = list(range(5)) + list(range(10, 15)) + list(range(20, 26)) + list(range(31, 36)) + list(range(41, 46)) + [51]
        self.room1 = list(range(5, 10)) + list(range(15, 20)) + list(range(26, 31)) + list(range(36, 41)) + list(range(46, 51)) + list(range(52, 57))
        self.room2 = list(range(68, 73)) + list(range(78, 83)) + list(range(89, 94)) + list(range(99, 104)) + [62]
        self.room3 = list(range(57, 62)) + list(range(63, 68)) + list(range(73, 78)) + list(range(83, 89)) + list(range(94, 99))
        self.rooms = [self.room0, self.room1, self.room2, self.room3]

    def get(self, phi, option):
        interest = float(phi in self.rooms[option])
        if interest == 0.:
            interest=0.1
        return interest

    def getall(self, phi):
        interest= np.ones((4)) * 0.1
        for o in range(4):
            if phi in self.rooms[o]:
                interest[o] = 1.
        return interest


class IntraOptionQLearning:
    def __init__(self, discount, lr, terminations, weights, meta_policy, noptions, initiation_policy=None):
        self.lr = lr
        self.discount = discount
        self.terminations = terminations
        self.weights = weights
        self.meta_policy = meta_policy
        self.noptions=noptions
        self.initiation_policy=initiation_policy

    def start(self, phi, option):
        self.last_phi = phi
        self.last_option = option
        self.last_value = self.value(phi)
        

    def value(self, phi, option=None):
        if option is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, option], axis=0)

    def advantage(self, phi, option=None):
        values = self.value(phi)
        advantages = values - self.meta_policy.pmf(phi).dot(values)
        if option is None:
            return advantages
        return advantages[option]

    def uponarrival(self, next_phi, option):
        qvalues = self.value(next_phi,option)
        all_values = self.value(next_phi)
        if self.initiation_policy is  None:
            values = self.meta_policy.pmf(next_phi).dot(all_values)
            beta = self.terminations[option].pmf(next_phi)
        else:
            values = np.array(self.initiation_policy.pmf(next_phi)).dot(all_values)
            beta = self.terminations.pmf(next_phi,option)
        u_so = qvalues * (1-beta) + values * beta
        return u_so


    def update(self, next_phi, next_option, reward, done, multiplier):

        # One-step update target
        update_target = reward
        current_values = self.value(next_phi)
        if not done:
            update_target += self.discount * current_values[next_option]

        
        # Dense gradient update step
        tderror = update_target - self.last_value[self.last_option]


        for o in range (self.noptions):
            self.weights[self.last_phi, o] += self.lr*multiplier[o] * \
            (reward + self.discount * float(not done) * self.uponarrival(next_phi,o) - self.last_value[o] )
            

        self.last_value = self.value(next_phi)
        self.last_option = next_option
        self.last_phi = next_phi




class TerminationGradient:
    def __init__(self, terminations, critic, lr,noptions):
        self.terminations = terminations
        self.critic = critic
        self.lr = lr
        self.noptions=noptions

    def update(self, phi, option, multiplier):
        for o in range(self.noptions):
            magnitude, direction = self.terminations[o].grad(phi)
            self.terminations[o].weights[direction] -= \
                    self.lr*multiplier[o]*magnitude*(self.critic.advantage(phi, o))             

class IntraOptionGradient:
    def __init__(self, option_policies, lr, discount, critic,noptions):
        self.lr = lr
        self.option_policies = option_policies
        self.discount = discount
        self.critic= critic
        self.noptions=noptions

    def update(self, phi, option, action, reward, done, next_phi, next_option, critic, multiplier, prob_cur_option=1.):

        for o in range(self.noptions):

            adv =(reward + self.discount * float(not done) * self.critic.uponarrival(next_phi,o) - self.critic.value(phi,o))
            actions_pmf = self.option_policies[o].pmf(phi)
            mult_o = multiplier[o]
            self.option_policies[o].weights[phi, :] -= self.lr* mult_o *prob_cur_option[o]*adv*actions_pmf
            self.option_policies[o].weights[phi, action] += self.lr* mult_o*prob_cur_option[o]*adv



class FixedTermination:
    def __init__(self,eps=0.4):
        self.room0 =  list(range(0, 5)) + list(range(10, 15)) + list(range(20, 25)) + list(range(31, 36)) + list(range(41, 46)) 
        self.room1 = list(range(5, 10)) + list(range(15, 20)) + list(range(26, 31)) + list(range(36, 41)) + list(range(46, 51)) + list(range(52, 57))
        self.room2 = list(range(68, 73)) + list(range(78, 83)) + list(range(89, 94)) + list(range(99, 104)) 
        self.room3 = list(range(57, 62)) + list(range(63, 68)) + list(range(73, 78)) + list(range(83, 88)) + list(range(94, 99))
        self.rooms = [self.room0, self.room0, self.room1, self.room1, self.room2,  self.room2, self.room3,self.room3]
        self.eps =eps

    def sample(self, phi, option):
        if option>=len(self.rooms): # Primitive actions
            termination=True
        else:                       # Options
            if phi in self.rooms[option]:
                termination = np.random.uniform() < self.eps
            else:
                termination = True
        return termination

    def pmf(self,phi, option):
        if option>=len(self.rooms): # Primitive actions
            termination_prob=1.0
        else:                       # Options
            if phi in self.rooms[option]:
                termination_prob = self.eps
            else:
                termination_prob = 1.0
        return termination_prob



class FixedInitiationSet:
    def __init__(self,):
        self.option0 = list(range(0, 5)) + list(range(10, 15)) + list(range(20, 25)) + list(range(31, 36)) + list(range(41, 46)) + [51]
        self.option1 = list(range(0, 5)) + list(range(10, 15)) + list(range(20, 25)) + list(range(31, 36)) + list(range(41, 46)) + [25]
        self.option2 = list(range(5, 10)) + list(range(15, 20)) + list(range(26, 31)) + list(range(36, 41)) + list(range(46, 51)) + list(range(52, 57)) + [62]
        self.option3 = list(range(5, 10)) + list(range(15, 20)) + list(range(26, 31)) + list(range(36, 41)) + list(range(46, 51)) + list(range(52, 57)) + [25]
        self.option4 = list(range(68, 73)) + list(range(78, 83)) + list(range(89, 94)) + list(range(99, 104)) + [62]
        self.option5 = list(range(68, 73)) + list(range(78, 83)) + list(range(89, 94)) + list(range(99, 104)) + [88]
        self.option6 = list(range(57, 62)) + list(range(63, 68)) + list(range(73, 78)) + list(range(83, 88)) + list(range(94, 99)) + [51]
        self.option7 = list(range(57, 62)) + list(range(63, 68)) + list(range(73, 78)) + list(range(83, 88)) + list(range(94, 99)) + [88]
        self.options = [self.option0, self.option1, self.option2, self.option3, self.option4,  self.option5, self.option6,self.option7]

    def get(self, phi, option):
        if option > 7:
            interest= 1.
        else:
            interest = float(phi in self.options[option])
        return interest



class InitiationSetSoftmaxPolicy:
    def __init__(self, noptions, initiationset, poveroptions):
        self.noptions = noptions
        self.initiationset = initiationset
        self.poveroptions = poveroptions

    def pmf(self, phi, option=None):
        list1 = [self.initiationset.get(phi,opt) for opt in range(self.noptions)]   
        list2 = self.poveroptions.pmf(phi)
        normalizer = sum([x * y for x, y in zip(list1, list2)])
        pmf = [float(list1[i] * list2[i])/normalizer for i in range(self.noptions)]
        return pmf

    def sample(self, phi):
        # import pdb;pdb.set_trace()
        return int(np.random.choice(self.noptions, p=self.pmf(phi)))

    def all_pmfs(self,):
        all_pmfs=[]
        for phi in range(len(self.poveroptions.weights)):
            pmf = self.pmf(np.array([phi]))
            all_pmfs.append(pmf)
        return np.array(all_pmfs)


# up,down,left,right
class FixedPolicy:
    def __init__(self, nfeatures, nactions, option, max_val=0.5):
        min_val = (1.-max_val)/ 4
        self.weights = np.zeros((nfeatures, nactions))
        self.option=option

        if option==0:
            option0_ranges = [range(0,4), range(10,14),range(20,25),range(31,35),range(41,45)]
            for opt0_range in option0_ranges:
                self.weights[opt0_range,3] = max_val
                self.weights[opt0_range,:] += min_val
            self.weights[51,0] = max_val
            self.weights[51,:] += min_val
            self.weights[35,0] = max_val
            self.weights[35,:] += min_val
            self.weights[45,0] = max_val
            self.weights[45,:] += min_val
            self.weights[4,1] = max_val
            self.weights[4,:] += min_val
            self.weights[14,1] = max_val
            self.weights[14,:] += min_val
        elif option==1:
            option1_ranges = [range(0,5), range(10,15),range(20,25),range(31,36)]
            for opt1_range in option1_ranges:
                self.weights[opt1_range,1] = max_val
                self.weights[opt1_range,:] += min_val
            self.weights[25,2] = max_val
            self.weights[25,:] += min_val
            self.weights[45,2] = max_val
            self.weights[45,:] += min_val
            self.weights[44,2] = max_val
            self.weights[44,:] += min_val
            self.weights[43,2] = max_val
            self.weights[43,:] += min_val
            self.weights[42,1] = max_val
            self.weights[42,:] += min_val
            self.weights[41,3] = max_val
            self.weights[41,:] += min_val

        elif option==2:
            option2_ranges = [range(6,10), range(16,20),range(26,31),range(37,41),range(47,51),range(53,57)]
            for opt2_range in option2_ranges:
                self.weights[opt2_range,2] = max_val
                self.weights[opt2_range,:] += min_val
            self.weights[5,1] = max_val
            self.weights[5,:] += min_val
            self.weights[15,1] = max_val
            self.weights[15,:] += min_val
            self.weights[36,0] = max_val
            self.weights[36,:] += min_val
            self.weights[46,0] = max_val
            self.weights[46,:] += min_val
            self.weights[52,0] = max_val
            self.weights[52,:] += min_val
            self.weights[62,0] = max_val
            self.weights[62,:] += min_val

        elif option==3:
            option3_ranges = [range(5,10), range(15,20),range(26,31),range(36,41),range(46,51)]
            for opt3_range in option3_ranges:
                self.weights[opt3_range,1] = max_val
                self.weights[opt3_range,:] += min_val
            self.weights[25,3] = max_val
            self.weights[25,:] += min_val
            self.weights[52,3] = max_val
            self.weights[52,:] += min_val
            self.weights[53,3] = max_val
            self.weights[53,:] += min_val
            self.weights[54,1] = max_val
            self.weights[54,:] += min_val
            self.weights[55,2] = max_val
            self.weights[55,:] += min_val
            self.weights[56,2] = max_val
            self.weights[56,:] += min_val


        elif option==4:
            option4_ranges = [range(69,73), range(79,83),range(89,94),range(100,104)]
            for opt4_range in option4_ranges:
                self.weights[opt4_range,2] = max_val
                self.weights[opt4_range,:] += min_val
            self.weights[62,1] = max_val
            self.weights[62,:] += min_val
            self.weights[68,1] = max_val
            self.weights[68,:] += min_val
            self.weights[78,1] = max_val
            self.weights[78,:] += min_val
            self.weights[99,0] = max_val
            self.weights[99,:] += min_val

        elif option==5:
            option5_ranges = [range(78,83), range(89,94),range(99,104)]
            for opt5_range in option5_ranges:
                self.weights[opt5_range,0] = max_val
                self.weights[opt5_range,:] += min_val
            self.weights[88,3] = max_val
            self.weights[88,:] += min_val
            self.weights[68,3] = max_val
            self.weights[68,:] += min_val
            self.weights[69,3] = max_val
            self.weights[69,:] += min_val
            self.weights[70,0] = max_val
            self.weights[70,:] += min_val
            self.weights[71,2] = max_val
            self.weights[71,:] += min_val
            self.weights[72,2] = max_val
            self.weights[72,:] += min_val




        elif option==6:
            option6_ranges = [range(57,61), range(63,67),range(73,77),range(83,88),range(94,98)]
            for opt6_range in option6_ranges:
                self.weights[opt6_range,3] = max_val
                self.weights[opt6_range,:] += min_val
            self.weights[51,1] = max_val
            self.weights[51,:] += min_val
            self.weights[61,1] = max_val
            self.weights[61,:] += min_val
            self.weights[67,1] = max_val
            self.weights[67,:] += min_val
            self.weights[77,1] = max_val
            self.weights[77,:] += min_val
            self.weights[98,0] = max_val
            self.weights[98,:] += min_val


        elif option==7:
            option7_ranges = [range(63,68), range(73,78),range(83,88),range(94,99)]
            for opt7_range in option7_ranges:
                self.weights[opt7_range,0] = max_val
                self.weights[opt7_range,:] += min_val
            self.weights[88,2] = max_val
            self.weights[88,:] += min_val
            self.weights[61,2] = max_val
            self.weights[61,:] += min_val
            self.weights[60,2] = max_val
            self.weights[60,:] += min_val
            self.weights[59,2] = max_val
            self.weights[59,:] += min_val
            self.weights[58,0] = max_val
            self.weights[58,:] += min_val
            self.weights[57,3] = max_val
            self.weights[57,:] += min_val

        elif option==8:
            self.weights[:,0] = 1.
        elif option==9:
            self.weights[:,1] = 1.
        elif option==10:
            self.weights[:,2] = 1.
        elif option==11:
            self.weights[:,3] = 1.



    def sample(self, phi):
        action = int(np.random.choice(self.weights.shape[1], p=self.weights[phi][0]))
        return action

    def pmf(self,phi):
        return self.weights[phi][0]

    def all_pmfs(self):
        return self.weights


create_folder = lambda f: [ os.makedirs(f) if not os.path.exists(f) else False ]