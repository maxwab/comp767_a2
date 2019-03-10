import numpy as np
import gym

env = gym.make('Pendulum-v0')

# Printing information about the environment
print('Size of the environment action space: {}'.format(env.action_space))
print('Size of the environment observation space: {}'.format(env.observation_space))

# Action space is 1 dimensional: this is the torque (joint effort). It can be either positive or negative.
# Observation space is 3 dimensional: cos(theta), sin(theta) and velocity (theta dot). See (https://github.com/openai/gym/wiki/Pendulum-v0) for min and max values
#
# Note that the Pendulum environment has no episode termination.

# Testing the environment for 200 frames with random actions
env.reset()
for i in range(200):
    env.step(env.action_space.sample())
    env.render()
env.close() # We close the environment when we're done

# Now we observe some values for the state space and action space.
