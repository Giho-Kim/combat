"""Gymnasium declaration for the cooperative strike action space."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .env import World

class TeamStrikeEnv(gym.Env):
    def __init__(self, config=None):
        self.world=World(config);c=self.world.c
        self.observation_space=spaces.Box(-1,1,shape=(c.n_agents,self.world.obs_dim),dtype=np.float32)
        self.action_space=spaces.Dict({
            "target":spaces.MultiDiscrete(np.full(c.n_agents,c.n_targets,dtype=np.int64))})
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        return self.world.reset(seed),self.world.metrics()
    def step(self,action):
        obs,reward,terminated,truncated,info=self.world.step(action)
        info=dict(info,agent_rewards=reward)
        return obs,float(reward.sum()),terminated,truncated,info
