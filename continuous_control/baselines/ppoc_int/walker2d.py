import numpy as np
from gym.envs.mujoco import Walker2dEnv as Walker2dEnv_



def _sigmoids(x, value_at_1):
    scale = np.sqrt(-2 * np.log(value_at_1))
    return np.exp(-0.5 * (x*scale)**2)

class Walker2dStandEnv(Walker2dEnv_):

    def __init__(self):
        self.task = 'stand'
        self.stand_bound = 0.55
        self.walk_bound = 2.0
        self.stand_margin= self.stand_bound/2
        self.walk_margin= self.walk_bound/2
        super(Walker2dStandEnv, self).__init__()


    def _step(self, a):
        posbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.model.data.qpos[0:3, 0]

        
        torso_height=self.get_body_com("torso")[-1]
        torso_upright=self.get_body_xmat("torso")[-1,-1]
        in_bounds = (self.stand_bound <= torso_height)
        dist = (self.stand_bound - torso_height) / self.stand_margin
        standing = float(np.where(in_bounds, 1.0, _sigmoids(dist, 0.1)))
        upright = (1 + torso_upright) / 2
        stand_reward = (3*standing + upright) / 4
        done = not (height > 0.4 and height < 2.0 and
                    ang > -2.0 and ang < 2.0)


        if self.task == 'stand': #reward for standing up
            reward = stand_reward
        else: 
            speed = ((posafter - posbefore) / self.dt)
            
            in_bounds = (self.walk_bound <= speed)
            dist = (self.walk_bound - speed) / self.walk_margin
            move_reward = float(np.where(in_bounds, 1.0, _sigmoids(dist, 0.1)))
            reward = stand_reward * (5*move_reward + 1) / 6


        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20

    def reset_task(self, task):
        self.task = task




from gym.envs.registration import register
register(
    'Walker2dStand-v1',
    entry_point='walker2d:Walker2dStandEnv',
    max_episode_steps=1000
)