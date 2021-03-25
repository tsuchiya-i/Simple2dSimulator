#coding:utf-8

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import gym
import Simple2dSimulator
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time


np.set_printoptions(suppress=True)

env = gym.make('Simple2dSimulator-v0')
observation = env.reset()
#print(observation)
#print(env.observation_space)

print(env.action_space.high)
print(env.action_space.low)

start=env.world_time

#print(env.action_space.shape[0])
#print(type(env.observation_space.shape))

#print(observation['map'])
print("start : "+str(start))

t_rwd = 0

for i in range(10000):
    stime = time.time()
    if i%1==0:
        env.render()
    action = env.action_space.sample()
    action = [0.5,0.0]
    observation, reward, done,  _ = env.step(action)
    ctime = time.time()-stime
    #print("check_time="+str(ctime))

    #print("=========================")
    t_rwd += reward
    print(observation)
    #print(now_state)
    #print(observation)
    #print(world_time)
    #print(observation['lidar'])
    ltime = time.time()-stime
    #print("loop_time="+str(ltime))
    #print("=========================")
    #time.sleep(0.1)

    if done:
        print("finish : "+str(env.world_time))
        env.reset()
        #break
