# In this file we will evaluate the TD lambda algorithm

import numpy as np
import tiles
import angles
import value_function
import algorithms

import gym

env = gym.make('Pendulum-v0')

# Initialisation of the hyperparameters
LEN_EPISODE = 200 # The task is non-episodic so we define the len of an episode as 200 (like openAI gym) 
GAMMA = 0.95
N_TILES = 5
N_BINS = 10

# Initialize the tiles
tiles_intervals = tiles.get_intervals(N_TILES, N_BINS)
n_params = len(tiles_intervals[0][0])

print('{} parameters'.format(n_params))

z = np.zeros(n_params, 1)
obs = env.reset()


