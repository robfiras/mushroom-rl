import numpy as np
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv


class Walker2dEnvPOMPD(Walker2dEnv):

    def __init__(self, obs_to_hide=("velocities",), **kwargs):

        self._hidable_obs = ("positions", "velocities")
        if type(obs_to_hide) == str:
            obs_to_hide = (obs_to_hide,)
        assert not all(x in obs_to_hide for x in self._hidable_obs), "You are not allowed to hide all observations!"
        assert all(x in self._hidable_obs for x in obs_to_hide), "Some of the observations you want to hide are not" \
                                                                 "supported. Valid observations to hide are %s."\
                                                                 % (self._hidable_obs,)
        self._obs_to_hide = obs_to_hide
        super().__init__(**kwargs)

    def _get_obs(self):
        observations = []
        if "positions" not in self._obs_to_hide:
            position = self.sim.data.qpos.flat.copy()
            if self._exclude_current_positions_from_observation:
                position = position[1:]
            observations += [position]

        if "velocities" not in self._obs_to_hide:
            velocity = np.clip(self.sim.data.qvel.flat.copy(), -10, 10)
            observations += [velocity]

        return np.concatenate(observations).ravel()

