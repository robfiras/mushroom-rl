import numpy as np
import collections
from mushroom_rl.environments import Gym


class NoisyDelayedGym(Gym):

    def __init__(self, name, delay_steps=1, noise_std=0.0, **kwargs):
        super(NoisyDelayedGym, self).__init__(name, **kwargs)
        self._delay_steps = delay_steps
        if self._delay_steps > 0:
            self._reset_obs_buffer()
        else:
            self._obs_buffer = None
        self._noise_std = noise_std

    def reset(self, obs=None):
        obs = super(NoisyDelayedGym, self).reset(obs)
        noise = np.random.randn(obs.shape[0]) * self._noise_std
        if self._obs_buffer:
            self._reset_obs_buffer()
            self._obs_buffer.appendleft(obs)
            return self._obs_buffer[-1] + noise
        else:
            return obs + noise

    def _reset_obs_buffer(self):
        self._obs_buffer = collections.deque([np.zeros(self.info.observation_space.shape[0])
                                              for i in range(self._delay_steps+1)], self._delay_steps+1)

    def step(self, action):
        obs, r, a, i = super(NoisyDelayedGym, self).step(action)
        noise = np.random.randn(obs.shape[0]) * self._noise_std
        if self._obs_buffer:
            self._obs_buffer.appendleft(obs)
            return self._obs_buffer[-1] + noise, r, a, i
        else:
            return obs + noise, r, a, i
