import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box
from ..params import DEFAULT_PARAMS

class MyWayHome(MiniWorldEnv):
    """
    Classic four rooms environment
    """

    def __init__(self, sparse=False, verysparse=False, tv=False, **kwargs):

        params = DEFAULT_PARAMS.no_random()
        params.set('forward_step', 0.18)
        params.set('turn_step', 30)
        self.sparse=sparse
        self.verysparse=verysparse
        self.tv=tv

        super().__init__(
            max_episode_steps=2100,
            params=params,
        )

        # Movement actions 
        self.action_space = spaces.Discrete(self.actions.move_forward+1)


    def _gen_world(self):

        ### top-down view is:
        ### -x → +x
        ### -z ↓ +z

        wall_height=3.5
        # Bot-left room
        room0 = self.add_rect_room(
            min_x=-5, max_x=-1,
            min_z=1 , max_z=5,
            wall_height= wall_height,
            wall_tex='doom_COMPBLUE',
            floor_tex='doom_BRICK7',
            ceil_tex='doom_BRICK7'
        )
        # Bot-right room
        room1 = self.add_rect_room(
            min_x=1, max_x=5,
            min_z=1, max_z=5,
            wall_height= wall_height,
            wall_tex='doom_DOOR',
            floor_tex='doom_BRICK7',
            ceil_tex='doom_BRICK7'
        )
        # Top-right room
        room2 = self.add_rect_room(
            min_x=1 , max_x=5,
            min_z=-5, max_z=-1,
            wall_height= wall_height,
            wall_tex='doom_BRNSMAL2',
            floor_tex='doom_BRICK7',
            ceil_tex='doom_BRICK7'
        )
        # Top-left room
        room3 = self.add_rect_room(
            min_x=-5, max_x=-1,
            min_z=-5, max_z=-1,
            wall_height= wall_height,
            wall_tex='doom_ICKWALL1' ,
            floor_tex='doom_BRICK7',
            ceil_tex='doom_BRICK7'
        )

        roombot = self.add_rect_room(
            min_x=-5, max_x=-1,
            min_z=7 , max_z=11,
            wall_height= wall_height,
            wall_tex='doom_REDWALL' ,
            floor_tex='doom_BRICK7',
            ceil_tex='doom_BRICK7'
        )

        roomverysparse = self.add_rect_room(
            min_x=-4, max_x=-1,
            min_z=13, max_z=15,
            wall_height= wall_height,
            wall_tex='doom_BIGBRIK1',
            floor_tex='doom_BRICK7',
            ceil_tex='doom_BRICK7'
        )
        roomsparse = self.add_rect_room(
            min_x=-11, max_x=-7,
            min_z=1 , max_z=5,
            wall_height= wall_height,
            wall_tex='doom_BIGDOOR2',
            floor_tex='doom_BRICK7',
            ceil_tex='doom_BRICK7'
        )

        roomright = self.add_rect_room(
            min_x=7, max_x=11,
            min_z=-5 , max_z=-1,
            wall_height= wall_height,
            wall_tex='doom_SILVER2',
            floor_tex='doom_BRICK7',
            ceil_tex='doom_BRICK7'
        )

        roompregoal = self.add_rect_room(
            min_x=7, max_x=11,
            min_z=1 , max_z=5,
            wall_height= wall_height,
            wall_tex='doom_WOOD9',
            floor_tex='doom_BRICK7',
            ceil_tex='doom_BRICK7'
        )

        roomgoal = self.add_rect_room(
            min_x=8, max_x=10,
            min_z=5 , max_z=7,
            wall_height= wall_height,
            wall_tex='doom_BIGBRIK1',
            floor_tex='doom_BRICK7',
            ceil_tex='doom_BRICK7'
        )





        # Add openings to connect the rooms together
        self.connect_rooms(room0, room1, min_z=2, max_z=4, max_y=wall_height,wall_tex='doom_BIGBRIK1')
        self.connect_rooms(room1, room2, min_x=2, max_x=4, max_y=wall_height,wall_tex='doom_BIGBRIK1')
        self.connect_rooms(room2, room3, min_z=-4, max_z=-2, max_y=wall_height,wall_tex='doom_BIGBRIK1')
        self.connect_rooms(room3, room0, min_x=-4, max_x=-2, max_y=wall_height,wall_tex='doom_BIGBRIK1')

        self.connect_rooms(roombot, room0, min_x=-4, max_x=-2, max_y=wall_height,wall_tex='doom_BIGBRIK1')
        self.connect_rooms(roombot, roomverysparse, min_x=-4, max_x=-2, max_y=wall_height,wall_tex='doom_BIGBRIK1')
        self.connect_rooms(roomsparse, room0, min_z=2, max_z=4, max_y=wall_height,wall_tex='doom_BIGBRIK1')        
        self.connect_rooms(room2, roomright,  min_z=-2, max_z=-4, max_y=wall_height,wall_tex='doom_BIGBRIK1')
        self.connect_rooms(roomright, roompregoal,  min_x=8, max_x=10, max_y=wall_height,wall_tex='doom_BIGBRIK1')
        self.connect_rooms(roompregoal, roomgoal,  min_x=8, max_x=10, max_y=wall_height,wall_tex='doom_BIGBRIK1')


        self.box = self.place_entity(Box(color='red'),pos=[9,0,6])


        if self.sparse:
            self.place_agent(room=roomsparse,dir=0)
        elif self.verysparse:
            self.place_agent(room=roomverysparse,dir=np.pi)
        else:
            self.place_agent()
    def step(self, action):
        obs, reward, done, info = super().step(action)

        if action == self.actions.toggle and self.tv:
            self.tv.randomize(None,None)
            del self.entities[-1]
            self.entities.append(self.tv)

        if self.near(self.box):
            reward += self._reward()
            done = True

        return obs, reward, done, info


class MyWayHomeSparse(MyWayHome):
    def __init__(self):
        super().__init__(sparse=True)



class MyWayHomeVerySparse(MyWayHome):
    def __init__(self):
        super().__init__(verysparse=True)


class MyWayHomeSparseNoisyTv(MyWayHome):
    def __init__(self):
        super().__init__(sparse=True,tv=True)
        self.action_space = spaces.Discrete(len(self.actions))
        # self.action_space = spaces.Discrete(self.actions.move_forward+3)



class MyWayHomeNoisyTv(MyWayHome):
    def __init__(self):
        super().__init__(tv=True)
        # Movement actions + Toggle
        self.action_space = spaces.Discrete(self.actions.move_forward+2)
