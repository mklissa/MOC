import os
import numpy as np
import gym
from gym import utils
from gym.envs.mujoco.ant import AntEnv
from gym.envs.mujoco import mujoco_env
import xml.etree.ElementTree as ET
import tempfile
import seeding

class WallsEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(
            self,
            file_path,
            wall_height = 3,
            wall_pos_range = ([2.8, 0.0], [2.8, 0.0]),
            num_walls=2,
            *args,
            **kwargs):


        tree = ET.parse(file_path)
        worldbody = tree.find(".//worldbody")

        height = wall_height
        self.w_height= wall_height
        self.wall_pos_range = wall_pos_range
        rand_x = np.random.uniform(wall_pos_range[0][0], wall_pos_range[1][0]) #self.np_np.random.uniform(low=wall_pos_range[0][0], high=wall_pos_range[1][0], size=1)[0]
        rand_y = np.random.uniform(wall_pos_range[0][1], wall_pos_range[1][1]) #self.np_np.random.uniform(low=wall_pos_range[0][1], high=wall_pos_range[1][1], size=1)[0]
        self.wall_pos = wall_pos = (rand_x, rand_y)
        torso_x, torso_y = 0, 0
        self._init_torso_x = torso_x
        self._init_torso_y = torso_y


        self.wall_size = (0.25, 2., height)
        self.side_wall_size = (20., .25, .75)


        self.num_walls = num_walls
        self.space_between = 9
        self.init_y = 1.5

        for i in range(self.num_walls):
            ET.SubElement(
                worldbody, "geom",
                name="wall %i" % i,
                pos="%f %f %f" % (wall_pos[0]+i*self.space_between,
                                  self.init_y * (-1)**i,
                                  height / 2.),
                size="%f %f %f" % self.wall_size,
                type="box",
                material="",
                density="5.",
                rgba="1.0 0. 1. 1",
                contype="1",
                conaffinity="1",
                condim="1",
            )

        for i in range(2):
            ET.SubElement(
                worldbody, "geom",
                name="sidewall %i" % i,
                pos="%f %f %f" % (self.side_wall_size[0]/2,
                                  (self.init_y+self.wall_size[1]) * (-1)**i,
                                  self.side_wall_size[2]/2),
                size="%f %f %f" % self.side_wall_size,
                type="box",
                material="",
                density="5.",
                rgba=".0 .0 .0 .2",
                contype="1",
                conaffinity="1",
                condim="1",
            )        


        _, file_path = tempfile.mkstemp(text=True)
        tree.write(file_path)
        self.file_path=file_path

        mujoco_env.MujocoEnv.__init__(self, file_path, 5)
        utils.EzPickle.__init__(self)

        



    def _get_readings(self):
        # The observation would include both information about the robot itself as well as the sensors around its
        # environment
        max_x=1
        max_y=5
        terrain_read = np.zeros((max_y,max_x))


        index_ratio = 1/1 # number of indices per meter. a ratio of 2 means each index is 0.5 long in mujoco coordinates

        robot_x, robot_y, robot_z = robot_coords = self.get_body_com("torso")
        

        wall_length = self.wall_size[1] * 2

        for i in range(self.num_walls):

            diff_x = self.wall_pos[0]+i*self.space_between - robot_x

            index_x =  0 if diff_x < 4.5 and diff_x > 0 else -1


            

            if index_x < 2 and index_x >=0 and i%2==0: 

                wall_starty =   self.init_y * (-1)**i - self.wall_size[1]
                diff = (robot_y +  2/index_ratio) - wall_starty

                if diff >= 0.:
                    end_index = int(round(diff * index_ratio))
                    terrain_read[:end_index,index_x] = 1.

            elif index_x < 2 and index_x >=0 and i%2==1: 

                wall_endy =   self.init_y * (-1)**i + self.wall_size[1]
                diff = wall_endy - (robot_y -  2/index_ratio) 

                if diff >= 0.:
                    end_index = int(round(diff * index_ratio))
                    if end_index == 0:
                        end_index = 1 # due to the way negative slicing works
                    terrain_read[-end_index:,:] = 1.
            



        return terrain_read.flatten()




class AntWallsEnv(WallsEnv):
    NAME= "AntWalls"


    def __init__(self,num_walls=0):

        self.steps = 0
        file_path = "assets/ant.xml"
        WallsEnv.__init__(self,file_path,num_walls=num_walls)


    def _step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        self.steps += 1
        done = self.steps >= 1000 or done
        return ob, reward, done, {}

    def _get_obs(self):

        terrain_read = self._get_readings()
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            terrain_read
        ])
        
    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.steps = 0
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
