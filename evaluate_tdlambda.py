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
parser.add_argument('--lambda_', type=float, required=True)
parser.add_argument('--alpha', type=float, required=True)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--torque_value', type=float, default=1.0)
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

LAMBDA = args.lambda_
ALPHA = args.alpha
SEED = args.seed

# Creating the directory if it does not exist
p_exp_l = exp_path / 'l:{}'.format(LAMBDA)
if not Path.exists(p_exp_l):
    try:
        os.makedirs(p_exp_l)
        print('Experiment {} created'.format(p_exp_l))
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
lphi = []
for _ in tqdm(range(N_EPISODES)):
    obs = env.reset().reshape(1, -1)
    for i in range(LEN_EPISODE):
        action = policy.policy(obs, torque_value=args.torque_value)
        phi = tiles.encode(obs, tiles_intervals)
        next_obs, rew, done, _ = env.renv.step(action)
        next_obs = next_obs.reshape(1, -1)
        next_phi = tiles.encode(next_obs, tiles_intervals)
        w, z = algorithms.update_tdlambda(w, z, phi, rew, next_phi, v, g, alpha_ = alpha, lambda_ = LAMBDA, gamma_ = GAMMA)
        obs = next_obs
    lw.append(w)
    lphi.append(v(phi_0, w))

# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------

W = np.stack(lw, axis=1)
# Save all the weights at the end of the run 
filename_W = 'weights_l:{}_a:{}_s:{}.npy'.format(LAMBDA, ALPHA, SEED)
filename_phi = 'phi_l:{}_a:{}_s:{}.npy'.format(LAMBDA, ALPHA, SEED)
with open(p_exp_l / filename_W, 'wb') as fd:
    np.save(fd, W) 
    print('Saved weights for lambda={}, alpha={}, seed={} at location {}'.format(LAMBDA, ALPHA, SEED, p_exp_l / filename_W))

with open(p_exp_l / filename_phi, 'wb') as fd:
    np.save(fd, np.array(lphi)) 
    print('Saved phi values for lambda={}, alpha={}, seed={} at location {}'.format(LAMBDA, ALPHA, SEED, p_exp_l / filename_phi))
