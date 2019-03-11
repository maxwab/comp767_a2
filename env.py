# In this file we define another wrapper that lets us reset the state to (0.0, 0.0)
import gym

class ResetableEnv(gym.wrappers.TimeLimit):
    def __init__(self, env):
        self.renv = env
    def reset(self, **kwargs):
        self.renv.reset(**kwargs)
        self.renv.env.last_u = None
        self.renv.env.state = [0.0, 0.0]
        return self.renv.env._get_obs()
