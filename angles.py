# In this file we define functions used to play with angles
import numpy as np
import math

def to_angle(cosa, sina):
    r'''
    cosa: (float), cosinus of angle
    sina: (float), sinus of angle
    return: theta (float), angle, between -pi and pi.
    '''
    
    # Transform the cos and sin of the angle to a real angle that we can discretize
    theta_ = np.arccos(cosa)
    theta = (2*(sina >= 0).astype(int)-1)*theta_ # returns theta if sina >= 0 else, -theta.

    return theta

def prep(obs):
    r'''
    Normalizes an observation so that angular position and velocity are between -1 and 1
    The returned element has 2 dimensions (ang pos and vel) while the input had 3 (cosa, sina, vel)
    obs: (np.ndarray), observation(s).
    '''
    assert type(obs) == np.ndarray
    assert len(obs.shape) == 2

    pos = to_angle(obs[:,0], obs[:,1])
    pos = (1 + pos / np.pi) / 2
    vel = (1 + obs[:, 2] / 8) / 2

    return np.stack((pos, vel), axis=1)
