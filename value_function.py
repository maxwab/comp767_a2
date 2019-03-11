# In this file we define the value function approximator we want to use as well as their gradient w.r.t. the parameters.

import numpy as np

def linear_approx(phi, w):
    r'''
    Implementation of a linear approximator
    '''
    return w[phi.astype(bool)].sum()

def linear_approx_grad(phi, w):
    r'''
    Gradient of a linear approximator w.r.t. w
    '''
    return phi
