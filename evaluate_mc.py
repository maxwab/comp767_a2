# In this file we will evaluate the TD lambda algorithm
import os
from pathlib import Path
import numpy as np
import random
import tiles
import angles
import value_function
import algorithms
import policy
from env import ResetableEnv
from functools import partial
from tqdm import tqdm
import gym
from time import time
import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument('--alpha', type=float, required=True)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--save_dir', type=str, required=True)

args = parser.parse_args()

# Creating the environment. We wrap it so that we always reset to (0.0, 0.0)
base_env = gym.make('Pendulum-v0')
env = ResetableEnv(base_env)

# Initialisation of the hyperparameters
LEN_EPISODE = 200 # The task is non-episodic so we define the len of an episode as 200 (like openAI gym)
N_EPISODES = 200
GAMMA = .95
N_TILES = 5
N_BINS = 10

# Initialize the tiles
tiles_intervals = tiles.get_intervals(N_TILES, N_BINS)
n_params = N_TILES * N_BINS**2

# --------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------

exp_path = Path('log/{}'.format(args.save_dir))

ALPHA = args.alpha
SEED = args.seed

# Creating the directory if it does not exist
p_exp_mc = exp_path / 'mc'
if not Path.exists(p_exp_mc):
    try:
        os.makedirs(p_exp_mc)
        print('Experiment {} created'.format(p_exp_mc))
    except FileExistsError:
        pass

np.random.seed(SEED)
random.seed(SEED)
w = (2*np.random.rand(n_params)-1)/1000
z = np.zeros((n_params, ))
alpha = ALPHA / N_TILES

print('{} parameters'.format(n_params))

# Definition of the function approximator and gradient approximator
v = value_function.linear_approx
g = value_function.linear_approx_grad

# We define the observation we want to check the value of
phi_0 = tiles.encode(np.array([[1.0, 0.0, 0.0]]), tiles_intervals) # cos(0) = 1, sin(0) = 0

# --------------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------------

# We are going to store in memory the trained weights, at the end of each episode.
lw = []
lphi0 = []
for _ in tqdm(range(N_EPISODES)):
    obs = env.reset().reshape(1, -1)
    lphi = []
    lrew = []
    G = 0
    
    # During the episode we store the states we visit and the rewards we get.
    for i in range(LEN_EPISODE):
        action = policy.policy(obs)
        phi = tiles.encode(obs, tiles_intervals)
        lphi.append(phi)
        next_obs, rew, done, _ = env.renv.step(action)
        lrew.append(rew)
        next_obs = next_obs.reshape(1, -1)
        obs = next_obs

    # Now we update the weights for each transition
    for t in reversed(range(LEN_EPISODE)):
        G += GAMMA * G + lrew[t]
        w = algorithms.update_mc(w, lphi[t], G, v, g, ALPHA)

    # Logging
    lw.append(w)
    lphi0.append(v(phi_0, w))

import pdb; pdb.set_trace()
# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------

W = np.stack(lw, axis=1)
# Save all the weights at the end of the run 
filename_W = 'mc_weights_a:{}_s:{}.npy'.format(ALPHA, SEED)
filename_phi = 'phi_a:{}_s:{}.npy'.format(ALPHA, SEED)
with open(p_exp_mc / filename_W, 'wb') as fd:
    np.save(fd, W) 
    print('Saved weights for alpha={}, seed={} at location {}'.format(ALPHA, SEED, p_exp_mc / filename_W))

with open(p_exp_mc / filename_phi, 'wb') as fd:
    np.save(fd, np.array(lphi0)) 
    print('Saved phi values for alpha={}, seed={} at location {}'.format(ALPHA, SEED, p_exp_mc / filename_phi))
