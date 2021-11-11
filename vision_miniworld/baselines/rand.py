import bsuite

import numpy as np

# SAVE_PATH_RAND = '/tmp/bsuite/rand'
# env = bsuite.load_and_record('bandit_noise/0', save_path=SAVE_PATH_RAND, overwrite=True)

# for episode in range(env.bsuite_num_episodes):
#   timestep = env.reset()
#   while not timestep.last():
#     action = np.random.choice(env.action_spec().num_values)
#     timestep = env.step(action)



from baselines.common.vec_env import dummy_vec_env
from baselines.ppo2 import ppo2
from bsuite.utils import gym_wrapper
import tensorflow as tf


SAVE_PATH_PPO = '/tmp/bsuite/ppo'

def _load_env():
  raw_env = bsuite.load_and_record(
      bsuite_id='mountain_car/0', 
      save_path=SAVE_PATH_PPO, logging_mode='csv', overwrite=True)
  return gym_wrapper.GymFromDMEnv(raw_env)
env = dummy_vec_env.DummyVecEnv([_load_env])


import pdb;pdb.set_trace()
for episode in range(env.envs[0]._env.bsuite_num_episodes):
  done=False  
  obs = env.reset()
  while not done:
    action = env.envs[0].action_space.sample()
    obs, rewards, done, infos = env.step(action)