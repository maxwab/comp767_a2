import numpy as np
import angles

# In this file we code the functions required for tile coding.


def get_intervals(n, b):
    r'''
    Creates the tiles for a 2d observation space where each dimension is of dimension 1.
    env: (environment) Gym 2d environment
    n: (int) number of tiles
    b: (int) number of bins per dimension per tile
    '''
    # We process first the angular position and then the angular velocity
    len_per_dim = 1 
    w = len_per_dim / (b - 1 + 1/n)
    u = w / n
    base_interval = np.asarray([i*w for i in range(b)])
    l_intervals = [(base_interval, base_interval)]

    # We now create the intervals for the remaining tilings
    for i in range(1, n):
        l_intervals.append((base_interval - i * u, base_interval - i * u)) 
    
    return l_intervals


def encode(obs, tiles_intervals):
    r'''
    Encodes an observation (output from the environment) into a tile code.
    obs: (np.ndarray) unnormalized obs
    '''
    
    pos_, vel_ = np.split(angles.prep(obs), 2, axis=1) # Normalizing observation(s)

    feat = np.empty((obs.shape[0], 0))
    for inter_pos, inter_vel in tiles_intervals:
        idx_pos = np.digitize(pos_, inter_pos) - 1 # Get the bin
        idx_vel = np.digitize(vel_, inter_vel) - 1
        idx = np.ravel_multi_index((idx_pos, idx_vel), (len(inter_pos), len(inter_vel))).squeeze().reshape(-1, 1)
        feat = np.concatenate((feat, idx), axis=1)

    return feat
