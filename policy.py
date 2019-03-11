# In this file we define the policy to evaluate
import numpy as np

def policy(obs, torque_value=0.1):
    r'''
    Computes the action to follow given the observation _obs_.
    obs: (np.ndarray), (set of) observations
    '''
    n = obs.shape[0]
    vel = obs[:, -1] # We use the _last_ index so that this code work with an unnormalized and normalized observation
    u = 2*np.random.binomial(1, p=0.9, size=(n, )) - 1 # 1 = same direction as velocity, -1 = opposite direction as velocity
    sgn = np.sign(vel)
    a = np.abs(sgn) * torque_value * u * sgn + (1 - np.abs(sgn)) * torque_value * (2 * np.random.binomial(1, p=0.5, size=(n,)) - 1) 
    return a
