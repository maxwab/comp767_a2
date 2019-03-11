# In this file we define the TD(lambda) algorithm

def update_tdlambda(w, z, phi, rew, next_phi, v, grad_v_w, alpha_, lambda_, gamma_):
    r'''
    Update step of the td(lambda) algorithm
    w: (np.ndarray) current parameters
    z: (np.ndarray) eligibility trace
    phi: (np.ndarray) representation of the current observation
    rew: (float), reward obtained
    next_phi: (np.ndarray) representation of the next observation
    v: (func) Approximate value function
    grad_v_w: (func) Gradient of the approximate value function w.r.t. the parameters
    alpha_: (float) learning rate
    lambda_: (float) hyperparameter of the algorithm
    gamma_: (float) hyperparameter of the problem
    '''
    grad_ = grad_v_w(phi, w) 
    z = gamma_ * lambda_ * z + grad_ 
    bellman_error = rew + gamma_ * v(next_phi, w) - v(phi, w)
    w = w + alpha_ * bellman_error * z
    return w, z
