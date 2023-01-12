import numpy as np
from gym.envs.mujoco.ant_v3 import AntEnv


class AntEnvPOMPD(AntEnv):

    def __init__(self, obs_to_hide=("velocities",), random_force_com=False, max_force_strength=0.5,
                 forward_reward_weight=1.0, **kwargs):

        self._hidable_obs = ("positions", "velocities", "contact_forces")
        if type(obs_to_hide) == str:
            obs_to_hide = (obs_to_hide,)
        assert not all(x in obs_to_hide for x in self._hidable_obs), "You are not allowed to hide all observations!"
        assert all(x in self._hidable_obs for x in obs_to_hide), "Some of the observations you want to hide are not" \
                                                                 "supported. Valid observations to hide are %s."\
                                                                 % (self._hidable_obs,)
        self._obs_to_hide = obs_to_hide
        self._random_force_com = random_force_com
        self._max_force_strength = max_force_strength
        self._force_strength = 0.0
        self._forward_reward_weight = forward_reward_weight
        super().__init__(**kwargs)

    def reset_model(self):
        if self._random_force_com:
            self._force_strength = np.random.choice([-self._max_force_strength, self._max_force_strength])
            sign = -1.0 if self._force_strength < 0 else 1.0
            self._forward_reward_weight = np.abs(self._forward_reward_weight) * sign
        return super().reset_model()

    def _get_obs(self):
        observations = []
        if "positions" not in self._obs_to_hide:
            position = self.sim.data.qpos.flat.copy()
            if self._exclude_current_positions_from_observation:
                position = position[2:]
            observations += [position]

        if "velocities" not in self._obs_to_hide:
            velocity = self.sim.data.qvel.flat.copy()
            observations += [velocity]

        if "contact_forces" not in self._obs_to_hide:
            contact_force = self.contact_forces.flat.copy()
            observations += [contact_force]

        return np.concatenate(observations).ravel()

    def step(self, action):

        torso_index = self.model.body_names.index('torso')
        self.data.xfrc_applied[torso_index, 0] = self._force_strength

        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        done = self.done
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        return observation, reward, done, info

