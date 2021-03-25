#coding:utf-8

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import gym
import Simple2dSimulator
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import time

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
import matplotlib.pyplot as plt
import tensorflow as tf



np.set_printoptions(suppress=True)

ENV_NAME = 'Simple2dSimulator-v0'

env = gym.make(ENV_NAME)
observation = env.reset()
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(512))
actor.add(Activation('relu'))
actor.add(Dense(256))
actor.add(Activation('relu'))
actor.add(Dense(128))
actor.add(Activation('relu'))
actor.add(Dense(64))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('tanh'))
#print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(512)(x)
x = Activation('relu')(x)
x = Dense(256)(x)
x = Activation('relu')(x)
x = Dense(128)(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
#print(critic.summary())

memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)

agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

try:
    #agent.load_weights(os.path.dirname(__file__) + '/weights/trained_weight/ddpg_{}_weights.h5f'.format(ENV_NAME))
    agent.load_weights('./weights/using/ddpg_{}_weights.h5f'.format(ENV_NAME))
    #agent.load_weights('./ddpg_{}_weights.h5f'.format(ENV_NAME))
    print("find weights-file")
except:
    print("not found weights-file")



t_rwd = 0

success_rate = 0
ep_count = 0
step_count = 0

fig, ax = plt.subplots(1, 1)
ax.set_ylim(math.radians(-50), math.radians(50))
x = []
y = []

#for i in range(100):
while(True):
    env.render()
    #action = env.action_space.sample()
    action = agent.forward(observation)
    #action = [1.0,-0.0]
    observation, reward, done,  GoalOrNot = env.step(action)
    t_rwd += reward
    #print(observation)
    #print(reward)
    step_count += 1
    x.append(step_count)
    y.append(action[1])
    if(len(x)>50):
        x.pop(0)
        y.pop(0)
    plt.xlim(x[0],x[0]+50)
    line, = ax.plot(x, y, color='blue')
    #plt.pause(0.01)
    # グラフをクリア
    line.remove()

    if done or (step_count>1000):
        x.clear()
        y.clear()
        #print("finish : "+str(env.world_time))
        #success_rate += GoalOrNot
        ep_count += 1
        print("{}%".format(success_rate/ep_count *100))
        print("{}回目".format(ep_count))
        env.reset()
        step_count = 0
        #break
