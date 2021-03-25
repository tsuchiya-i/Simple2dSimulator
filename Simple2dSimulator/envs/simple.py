#coding:utf-8

import numpy as np
import math
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import sys
import os
import cv2,time

import random

from Simple2dSimulator.envs.function.raycast import *

wall_switch = False
human_n = 10
human_mode = 2 #0:stop 1:straight 2:random 3:bound 4:onedirection

iscale = 4

#mapファイル読み込み
#im = cv2.imread('../maps/test_map.jpg')
#im = cv2.imread('../maps/nakano_11f_sim.png')
im = cv2.imread('../maps/paint_map/free_sim.png')
orgHeight, orgWidth = im.shape[:2]
size = (int(orgWidth/iscale), int(orgHeight/iscale))
im = cv2.resize(im, size)
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
threshold = 25
# 二値化(閾値100を超えた画素を255白にする。)
ret, img_thresh = cv2.threshold(im_gray, threshold, 255, cv2.THRESH_BINARY)

#mapファイル読み込み
#lim = cv2.imread('../maps/nakano_11f_line025.png')
lim = cv2.imread('../maps/paint_map/free_025.png')
lim_gray = cv2.cvtColor(lim, cv2.COLOR_BGR2GRAY)
lthreshold = 250
# 二値化(閾値100を超えた画素を255白にする。)
lret, limg_thresh = cv2.threshold(lim_gray, lthreshold, 255, cv2.THRESH_BINARY)

"""
cv2.imshow('image', im_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

class Simple(gym.Env):
    metadata = {'render.modes' : ['human', 'rgb_array']}

    def __init__(self):
        # world param
        self.map_height= img_thresh.shape[0]
        self.map_width = img_thresh.shape[1]
        self.map_size = 200
        self.xyreso = 0.05*4
        self.dt = 0.1
        self.world_time=0.0
        # state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)
        self.state = np.array([9, 10, math.radians(90), 0.0, 0.0])

        # robot param
        self.robot_radius = 0.3 #[m]
        self.human_radius = 0.6 #[m]

        # action param
        self.max_velocity = 0.8   # [m/s]
        self.min_velocity = -0.4  # [m/s]
        self.max_velocity_acceleration = 0.2  # [m/ss]
        self.min_velocity_acceleration = -0.2 # [m/ss]
        self.min_angular_velocity = math.radians(-40)  # [rad/s]
        self.max_angular_velocity = math.radians(40) # [rad/s]
        self.min_angular_acceleration = math.radians(-40)  # [rad/ss]
        self.max_angular_acceleration = math.radians(40) # [rad/ss]

        # lidar param
        self.yawreso = math.radians(10) # [rad]
        self.min_range = 0.20 # [m]
        self.max_range = 10.0 # [m]
        self.lidar_num = int(round(math.radians(360)/self.yawreso)+1)

        # set action_space (velocity[m/s], omega[rad/s])
        self.action_low  = np.array([self.min_velocity, self.min_angular_velocity]) 
        self.action_high = np.array([self.max_velocity, self.max_angular_velocity]) 
        self.action_space = spaces.Box(self.action_low, self.action_high, dtype=np.float32)

        # set observation_space
        self.min_yawrate  = math.radians(0)  # [rad]
        self.max_yawrate  = math.radians(360) # [rad]
        # state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        self.state_low  = np.array([0.0, 0.0, self.min_yawrate, self.min_velocity, self.min_angular_velocity])
        self.state_high = np.array([0.0, 0.0, self.max_yawrate, self.max_velocity, self.max_angular_velocity])

        # map
        if self.map_height < self.map_width:
            max_size = self.map_width
            min_size = self.map_height 
        else:
            max_size = self.map_height
            min_size = self.map_width

        self.map_low = np.full((self.map_height, self.map_width), 0)
        self.map_high = np.full((self.map_height, self.map_width), 100)
        self.map_low = np.full((self.map_size, self.map_size), 0)
        self.map_high = np.full((self.map_size, self.map_size), 100)

        if max_size*self.xyreso < math.pi:
            max_dist = math.pi
        else:
            max_dist = max_size*self.xyreso
        
        self.observation_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -math.pi])
        self.observation_high = np.array([max_dist,max_dist,max_dist,max_dist,max_dist,max_dist,max_dist,max_dist,max_dist,max_dist,math.pi])
        self.observation_space = spaces.Box(low = self.observation_low, high = self.observation_high, dtype=np.float32)
        #self.observation_low = np.array([0.0, -math.pi])
        #self.observation_high = np.array([max_dist,math.pi])
        #self.observation_space = spaces.Box(low = self.observation_low, high = self.observation_high, dtype=np.float32)
        
        #way point
        self.way_pioint_set()

        self.human_state = []

        self.viewer = None
        self.vis_lidar = True
        self.start_p_num = random.randint(0, len(self.waypoints)-1)
    
    # 状態を初期化し、初期の観測値を返す
    def reset(self):
        self.map = self.set_image_map(img_thresh)
        self.original_map = self.set_image_map(img_thresh)
        self.linemap = self.set_image_map(limg_thresh)
        
        #self.start_p_num = random.randint(0, len(self.waypoints)-1)
        if self.start_p_num == 19:
            self.start_p_num = 123#random.randint(0, (len(self.waypoints)-1)/2)
        else:
            self.start_p_num = 19
        self.start_p_num = 19
        self.start_p = self.waypoints[self.start_p_num]
        self.state = np.array([self.waypoints[self.start_p_num][0], self.waypoints[self.start_p_num][1], math.radians(random.uniform(179, -179)),0,0])
        self.state = np.array([self.waypoints[self.start_p_num][0], self.waypoints[self.start_p_num][1], math.radians(0),0,0])
        #self.state = np.array([self.waypoints[self.start_p_num][0], self.waypoints[self.start_p_num][1], math.radians(0),0,0])
        #goal_p_num = random.randint(0, len(self.waypoints)-1)
        #goal_p_num = 109+13*random.randint(-3, 3)+random.randint(-3, 3)
        if self.start_p_num == 19:
            goal_p_num = 19+13*7#123#random.randint((len(self.waypoints)-1)/2+1,len(self.waypoints)-1)
        else:
            goal_p_num = 19
        while self.start_p_num==goal_p_num:
            goal_p_num= random.randint(0, len(self.waypoints)-1)
        self.goal = self.waypoints[goal_p_num]

        #initial human pose
        self.ini_human(human_n)
        #reset goal_d and goal_a 
        self.reset_goal_info()
        #reset observe
        self.observation = self.observe()
        self.done = False
        # world_time reset
        self.world_time = 0.0
        return self.observation


    # actionを実行し、結果を返す
    def step(self, action):
        self.world_time += self.dt

        self.human_step()

        for i in range(len(action)):
            if action[i] < self.action_low[i]:
                action[i] = self.action_low[i]
            if action[i] > self.action_high[i]:
                action[i] = self.action_high[i]

        self.state[0] += action[0] * math.cos(self.state[2]) * self.dt
        self.state[1] += action[0] * math.sin(self.state[2]) * self.dt
        self.state[2] += action[1] * self.dt
        if self.state[2]<0.0:
            self.state[2] += math.pi * 2.0
        elif math.pi * 2.0 < self.state[2]:
            self.state[2] -= math.pi * 2.0
        self.state[3] = action[0]
        self.state[4] = action[1]


        d = np.linalg.norm(self.goal-self.state[:2])
        t_theta = np.angle(complex(self.goal[0]-self.state[0], self.goal[1]-self.state[1]))
        if t_theta < 0:
            t_theta = math.pi + (math.pi+t_theta)
        theta = t_theta
        if self.state[2] < math.pi:
            if t_theta > self.state[2]+math.pi:
                theta=t_theta-2*math.pi
        if self.state[2] > math.pi:
            if 0 < t_theta < self.state[2]-math.pi:
                theta=t_theta+2*math.pi
        anglegoal = theta-self.state[2]

        self.distgoal = np.array([d, anglegoal])
        self.observation = self.observe()
        reward = self.reward(action)
        self.done = self.is_done(True)
        self.old_distgoal = self.distgoal
        return self.observation, reward, self.done, {}
    
    def set_image_map(self, thresh_img):
        grid_map = np.zeros((self.map_height, self.map_width), dtype=np.int32)
        grid_map = np.where(thresh_img>100, 0, 1)
        return grid_map

    # ゴールに到達したかを判定
    def is_goal(self, show=False):
        if math.sqrt( (self.state[0]-self.goal[0])**2 + (self.state[1]-self.goal[1])**2 ) <= self.robot_radius*3:
            if show:
                print("Goal")
            return True
        else:
            return False

    # 移動可能範囲内に存在するか
    def is_movable(self, show=False):
        x = int(self.state[0]/self.xyreso)
        y = int(self.state[1]/self.xyreso)
        if(0<=x<self.map_width and 0<=y<self.map_height and self.map[self.map_height-1-y,x] == 0):
            return True
        else:
            if show:
                print("(%f, %f) is not movable area" % (x*self.xyreso, y*self.xyreso))
            return False

    # 高速衝突判定
    def is_collision(self, show=False):
        x = int(self.state[0]/self.xyreso) #[cell]
        y = int(self.state[1]/self.xyreso) #[cell]
        #print("robot:"+str(x))
        robot_radius_cell = int(self.robot_radius/self.xyreso) #[cell]
        sx = x - robot_radius_cell
        fx = x + robot_radius_cell
        sy = (self.map_height-1)-(y+robot_radius_cell)
        fy = (self.map_height-1)-(y-robot_radius_cell)
        if sx<0:
            sx = 0
        if fx>self.map_width-1:
            fx = self.map_width-1
        if sy<0:
            sy = 0
        if fy>self.map_height-1:
            fy = self.map_height-1

        obstacle = np.where(0<self.map[sy:fy,sx:fx])
        if len(obstacle[0]) > 0:
            print("(%f, %f) of collision" % (x*self.xyreso, y*self.xyreso))
            return True
        return False

    # 報酬値を返す
    def reward(self,  action):
        if self.is_goal():
            #return 25
            return 35
        elif self.is_collision():
            #return -25
            return -10
        elif not self.is_movable():
            #return -25
            return -10
        else:
            if self.lidar[np.argmin(self.lidar[:, 3]), 3] < self.robot_radius*2:
                wall_rwd = -1.0
            else:
                wall_rwd = 0.0
            vel_rwd = (action[0]-self.max_velocity)/self.max_velocity
            dist_rwd = (self.old_distgoal[0]-self.distgoal[0])/(self.max_velocity*self.dt)
            if dist_rwd<0:
                dist_rwd *= 10
            angle_rwd = (abs(self.old_distgoal[1])-abs(self.distgoal[1]))/(self.max_angular_velocity*self.dt)
            if angle_rwd<0:
                angle_rwd *= 10
            rwd = (vel_rwd*10 + dist_rwd + 2*angle_rwd)/4 + wall_rwd

            #rwd = 
            rwd = (vel_rwd + 2*dist_rwd + 2*angle_rwd)/5 + wall_rwd
            #rwd = (2*vel_rwd + dist_rwd + angle_rwd)/3
            #print("max:"+str(self.max_velocity))
            #print("vel_rwd  :"+str(vel_rwd))
            #print("dist_rwd :"+str(dist_rwd))
            #print("angle_rwd:"+str(angle_rwd))
            #print("===rwd===:"+str(rwd))
            return rwd


    # 終端状態か確認
    def is_done(self, show=False):
        return (not self.is_movable(show)) or self.is_collision(show) or self.is_goal(show)

    # 観測結果を表示
    def observe(self):
        # Raycasting
        Raycast = raycast(self.state[0:3], self.map, self.map_height,self.map_width, 
                                self.xyreso, self.yawreso,
                                self.min_range, self.max_range)
        self.lidar = Raycast.raycasting()
        observation = np.concatenate([self.lidar[:, 3], self.distgoal], 0)
        #observation = self.distgoal
        return observation

    def ini_human(self,num):
        self.human_state = []
        rand_num_his = []
        self.xxx = []
        self.yyy = []
        self.inwall = []
        
        while len(rand_num_his) < num:
            rand_num = random.randint(0, len(self.waypoints)-1)
            if (not rand_num in rand_num_his) and (rand_num != self.start_p_num):
                rand_num_his.append(rand_num)
                self.hstart_p = (self.waypoints[rand_num])
                #self.human_state.append(np.array([self.hstart_p[0], self.hstart_p[1],math.radians(random.uniform(179, -179))]))
                if (rand_num//13)%2 == 1:
                    self.human_state.append(np.array([self.hstart_p[0], self.hstart_p[1],math.radians(90)]))
                else:
                    self.human_state.append(np.array([self.hstart_p[0], self.hstart_p[1],-math.radians(90)]))
                self.xxx.append(0)
                self.yyy.append(0)
                self.inwall.append(False)
        """
        for i in range(5):
            hstart_p = self.waypoints[i*13+31]
            self.human_state.append(np.array([hstart_p[0], hstart_p[1], math.pi/2]))
            self.xxx.append(0)
            self.yyy.append(0)
            self.inwall.append(False)
        """

    def way_pioint_set(self):
        self.waypoints = []
        squre = 15
        self.dwidth = self.map_width*self.xyreso/squre
        self.dheight = self.map_height*self.xyreso/squre
        #self.waypoints = np.array([[9, 8.5], [9, 12], [9, 16], [9, 19], [15, 19], [20, 19], [25, 19], [30, 19], [35, 19], [40.5, 18], [40.5, 16], [40.5, 13], [40.5, 11], [40.5, 8.5], [35, 8.5], [30, 8.5], [25, 8.5], [20, 8.5], [15, 8.5], [48, 11], [57, 11], [63, 11], [48, 16], [57, 16], [63, 16], [48, 13], [3, 14]])
        for i in range(squre):
            for j in range(squre):
                if not (i==0 or i==squre-1 or j==0 or j==squre-1):
                    self.waypoints.append([i*self.dwidth,j*self.dheight])

    def reset_goal_info(self):
        d = np.linalg.norm(self.goal-self.state[:2])
        t_theta = np.angle(complex(self.goal[0]-self.state[0], self.goal[1]-self.state[1]))
        if t_theta < 0:
            t_theta = math.pi + (math.pi+t_theta)
        theta = t_theta
        if self.state[2] < math.pi:
            if t_theta > self.state[2]+math.pi:
                theta=t_theta-2*math.pi
        if self.state[2] > math.pi:
            if 0 < t_theta < self.state[2]-math.pi:
                theta=t_theta+2*math.pi
        anglegoal = theta-self.state[2]
        self.distgoal = np.array([d, anglegoal])
        self.start_distgoal = self.distgoal
        self.old_distgoal = self.distgoal
    
    def human_step(self):
        self.map = self.original_map.copy()
        for i in range(len(self.human_state)):
            randaction = 0.8
            randdirect = 0
            if human_mode == 0:
                randaction = 0
                randdirect = 0
            elif human_mode == 1:
                randaction = 0.5
                randdirect = 0
            elif human_mode == 2:
                randaction = random.uniform(1.0, 0.0)
                randdirect = random.uniform(math.pi, -math.pi)
            elif human_mode == 3:
                if self.inwall[i]:
                    #randaction *= -1
                    randdirect = math.pi/self.dt
            elif human_mode == 4:
                if self.inwall[i] and self.human_state[i][1] > 30:
                    self.human_state[i][1] = 2.5
                if self.inwall[i] and self.human_state[i][1] < 2.5:
                    self.human_state[i][1] = 45

            self.human_state[i][2] += randdirect * self.dt
            self.human_state[i][0] += randaction * math.cos(self.human_state[i][2]) * self.dt
            self.human_state[i][1] += randaction * math.sin(self.human_state[i][2]) * self.dt
            if self.human_state[i][2]<0.0:
                self.human_state[i][2] += math.pi * 2.0
            elif math.pi * 2.0 < self.human_state[i][2]:
                self.human_state[i][2] -= math.pi * 2.0

            self.xxx[i] = int(self.human_state[i][0]/self.xyreso)
            self.yyy[i] = (self.map_height-1)-int(self.human_state[i][1]/self.xyreso)
            """
            if self.xxx[i] < 1 or self.yyy[i] < 1:
                self.xxx[i] = 1
                self.yyy[i] = 1
            elif self.xxx[i]>self.map_width-2 or self.yyy[i]>self.map_height-2:
                self.xxx[i] = self.map_width-2
                self.yyy[i] = self.map_width-2
            """
            if self.inwall[i]:
                self.inwall[i] = False
            else:
                try:
                    self.inwall[i] = self.original_map[self.yyy[i],self.xxx[i]] == 1
                except:
                    self.inwall[i] = True
            for h in range(3):
                for w in range(3):
                    try:
                        self.map[self.yyy[i]-1+h,self.xxx[i]-1+w] = 2
                    except:
                        pass

    # レンダリング
    def render(self, mode='human', close=False):
        screen_width  = self.map_width
        screen_height = self.map_height
        scale_width = screen_width / float(self.map_width) 
        scale_height = screen_height / float(self.map_height)

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # wall
            if wall_switch: 
                for i in range(screen_height):
                    for j in range(screen_width):
    
                        if self.linemap[i][j] == 1:
                            wall = rendering.make_capsule(1, 1)
                            self.walltrans = rendering.Transform()
                            wall.add_attr(self.walltrans)
                            wall.set_color(0.2, 0.4, 1.0)
                            self.walltrans.set_rotation(0)
                            self.viewer.add_geom(wall)
                            self.walltrans.set_translation(j, screen_height-i)
                            self.walltrans.set_rotation(0)
                            self.viewer.add_geom(wall)

            # waypoint
            for point in self.waypoints:
                waypoint = rendering.make_circle(self.robot_radius/self.xyreso*scale_width)
                self.waypointtrans = rendering.Transform()
                waypoint.add_attr(self.waypointtrans)
                waypoint.set_color(0.8, 0.8, 0.8)
                self.waypointtrans.set_translation(point[0]/self.xyreso*scale_width, 
                        point[1]/self.xyreso*scale_height)
                self.viewer.add_geom(waypoint)

            # robot pose
            robot = rendering.make_circle(self.human_radius/self.xyreso*scale_width)
            self.robottrans = rendering.Transform()
            robot.add_attr(self.robottrans)
            robot.set_color(0.8, 0.8, 0.8)
            self.viewer.add_geom(robot)
            # robot yawrate
            orientation = rendering.make_capsule(self.human_radius/self.xyreso*scale_width, 2.0)
            self.orientationtrans = rendering.Transform()
            orientation.add_attr(self.orientationtrans)
            orientation.set_color(1.0, 1.0, 1.0)
            self.viewer.add_geom(orientation)

            """
            # human pose
            human = rendering.make_circle(self.human_radius/self.xyreso*scale_width)
            self.humantrans = rendering.Transform()
            human.add_attr(self.humantrans)
            human.set_color(0.6, 0.8, 0.8)
            self.viewer.add_onetime(human)
            # human yawrate
            human_orientation = rendering.make_capsule(self.human_radius/self.xyreso*scale_width, 2.0)
            self.human_orientationtrans = rendering.Transform()
            human_orientation.add_attr(self.human_orientationtrans)
            human_orientation.set_color(0.6, 0.8, 0.8)
            self.viewer.add_onetime(human_orientation)
            """

            # start
            start = rendering.make_circle(self.robot_radius*3/self.xyreso*scale_width)
            self.starttrans = rendering.Transform()
            start.add_attr(self.starttrans)
            start.set_color(0.0, 0.0, 1.0)
            #start.set_color(1.0, 0.0, 0.0)
            self.viewer.add_geom(start)
            # goal
            goal = rendering.make_circle(self.robot_radius*3/self.xyreso*scale_width)
            self.goaltrans = rendering.Transform()
            goal.add_attr(self.goaltrans)
            goal.set_color(1.0, 0.0, 0.0)
            self.viewer.add_geom(goal)



        self.starttrans.set_translation(self.start_p[0]/self.xyreso*scale_width, 
                self.start_p[1]/self.xyreso*scale_height)
        self.goaltrans.set_translation(self.goal[0]/self.xyreso*scale_width, 
                                       self.goal[1]/self.xyreso*scale_height)

        #robot
        robot_x = self.state[0]/self.xyreso * scale_width
        robot_y = self.state[1]/self.xyreso * scale_height
        self.robottrans.set_translation(robot_x, robot_y)
        self.orientationtrans.set_translation(robot_x, robot_y)
        self.orientationtrans.set_rotation(self.state[2])
        """
        #human
        human_x = self.human_state[0]/self.xyreso * scale_width
        human_y = self.human_state[1]/self.xyreso * scale_height
        self.humantrans.set_translation(human_x, human_y)
        self.human_orientationtrans.set_translation(human_x, human_y)
        #self.human_orientationtrans.set_rotation(self.human_state[2])
        """

        for i in range(len(self.human_state)):
            human = rendering.make_circle(self.human_radius/self.xyreso*scale_width)
            self.humantrans = rendering.Transform()
            human.add_attr(self.humantrans)
            human.set_color(0.2, 0.8, 0.2)
            self.humantrans.set_translation(self.human_state[i][0]/self.xyreso*scale_width, self.human_state[i][1]/self.xyreso*scale_height)
            self.viewer.add_onetime(human)

        if self.vis_lidar:
            for lidar in self.lidar:
              # print(lidar)
               if False:#lidar[4]%2==0: # 全部可視化したら見づらいので間引く
                   continue
               scan = rendering.make_capsule(np.sqrt(lidar[0]**2+lidar[1]**2)/self.xyreso*scale_width, 2.0)
               self.scantrans= rendering.Transform()
               scan.add_attr(self.scantrans)
               if lidar[5]:
                   scan.set_color(1.0, 0.5, 0.5)#緑
               elif lidar[2]<0.1:
                   scan.set_color(0.5, 1.0, 0.5)#緑
                   #scan.set_color(0.0, 1.0, 1.0)
               else:
                   scan.set_color(0.0, 1.0, 1.0)
               self.scantrans.set_translation(robot_x, robot_y)
               self.scantrans.set_rotation(self.state[2]+lidar[2])
               self.viewer.add_onetime(scan)
            
        # robot
        robot = rendering.make_circle(self.human_radius/self.xyreso*scale_width)
        self.robottrans = rendering.Transform()
        robot.add_attr(self.robottrans)
        robot.set_color(0.0, 0.0, 1.0)
        self.robottrans.set_translation(robot_x, robot_y)
        self.viewer.add_onetime(robot)
        """
        #human move point 
        human = rendering.make_circle(self.human_radius/self.xyreso*scale_width)
        self.humantrans = rendering.Transform()
        human.add_attr(self.humantrans)
        human.set_color(0.2, 0.8, 0.2)
        self.humantrans.set_translation(human_x, human_y)
        self.viewer.add_onetime(human)
        """
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
