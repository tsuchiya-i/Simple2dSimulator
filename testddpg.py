import numpy as np
import gym

import Simple2dSimulator

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
import matplotlib.pyplot as plt

import tensorflow as tf

ENV_NAME = 'Simple2dSimulator-v0'


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
env.reset()
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
print(actor.summary())

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
print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)

agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

try:
    agent.load_weights('ddpg_{}_weights.h5f'.format(ENV_NAME))
    print("find weights-file")
except:
    print("not found weights-file")

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=1000)
