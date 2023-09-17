import os 
from os import path

import mujoco_py.generated.const
import numpy as np
from gym.envs.mujoco.ant_v3 import AntEnv
from dm_control import mjcf
from tempfile import mkdtemp
import glfw


class AntEnvMazePOMDP(AntEnv):

    def __init__(self, obs_to_hide=("velocities",), forward_reward_weight=10.0, ctrl_cost_weight=0.0,
                 contact_cost_weight=0.0, include_body_vel=False, sequences=None, **kwargs):
        xml_file = path.join(path.dirname(__file__), "assets", "ant_mem.xml")

        if sequences is None:
            # zick-zack from left once starting from right and once from left
            sequences = [[2*((j+i) % 2)-1 for j in range(20)] for i in range(2)]

        self._env_types = [0, 1]
        self._current_env_type = 0

        self._xml_files = []
        for sequence in sequences:
            # modify xml to add segments
            xml_handle = mjcf.from_path(xml_file)
            xml_handle = self._add_segments(xml_handle, sequence)
            self._xml_files.append(xml_handle.to_xml_string())

        self._hidable_obs = ("env_type")
        if type(obs_to_hide) == str:
            obs_to_hide = (obs_to_hide,)
        assert not all(x in obs_to_hide for x in self._hidable_obs), "You are not allowed to hide all observations!"
        assert all(x in self._hidable_obs for x in obs_to_hide), "Some of the observations you want to hide are not" \
                                                                 "supported. Valid observations to hide are %s."\
                                                                 % (self._hidable_obs,)
        self._obs_to_hide = obs_to_hide
        self._forward_reward_weight = forward_reward_weight
        self._include_body_vel = include_body_vel
        super().__init__(xml_file=xml_file, healthy_z_range=(5.3, 6.0), ctrl_cost_weight=ctrl_cost_weight,
                         contact_cost_weight=contact_cost_weight,
                         exclude_current_positions_from_observation=False, **kwargs)
        self.reset_model()

    def reset_model(self):

        xml_file_ind = np.random.randint(0, len(self._xml_files))
        assert xml_file_ind in self._env_types
        self._current_env_type = xml_file_ind
        xml_file = self._xml_files[xml_file_ind]
        self.model = self._mujoco_bindings.load_model_from_xml(xml_file)
        self.sim = self._mujoco_bindings.MjSim(self.model)
        self.data = self.sim.data

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        self._set_action_space()

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)

        # destroy viewer
        if self.viewer is not None:
            glfw.destroy_window(self.viewer.window)
        self.viewer = None
        self._viewers = {}

        return super().reset_model()

    def _get_obs(self):
        obs = super()._get_obs()
        # env_type for policy
        env_type_policy = self._current_env_type if self.get_body_com("torso")[0] < 3 else -1

        # env_type for critic
        env_type_critic = self._current_env_type
        return np.concatenate([obs, [env_type_policy, env_type_critic]]).ravel()

    def get_mask(self, obs_to_hide):
        """ This function returns a boolean mask to hide observations from a fully observable state. """

        if type(obs_to_hide) == str:
            obs_to_hide = (obs_to_hide,)
        assert all(x in self._hidable_obs for x in obs_to_hide), "Some of the observations you want to hide are not" \
                                                                 "supported. Valid observations to hide are %s." \
                                                                 % (self._hidable_obs,)
        mask = []
        position = self.sim.data.qpos.flat.copy()
        if self._exclude_current_positions_from_observation:
            position = position[2:]
        velocity = self.sim.data.qvel.flat.copy()
        contact_force = self.contact_forces.flat.copy()

        # positions
        mask += [np.ones_like(position, dtype=bool)]

        # velocities
        mask += [np.ones_like(velocity, dtype=bool)]

        # contact forces
        mask += [np.ones_like(contact_force, dtype=bool)]

        if "env_type" not in obs_to_hide:
            mask += [np.ones_like(self._current_env_type, dtype=bool)]
            mask += [np.ones_like(self._current_env_type, dtype=bool)]
        else:
            mask += [np.atleast_1d(np.ones_like(self._current_env_type, dtype=bool))]     # show policy env_type
            mask += [np.atleast_1d(np.zeros_like(self._current_env_type, dtype=bool))]      # hide critic env_type

        return np.concatenate(mask).ravel()

    def step(self, action):

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

    def _set_observation_space(self, observation):
        return super(AntEnvMazePOMDP, self)._set_observation_space(observation[:-1])

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy if self._terminate_when_unhealthy else False
        return done

    @staticmethod
    def _add_segments(xml_handle, sequence):

        def _add_segment(segments_handle, n, s):
            segments_handle.add("geom", dclass="floor", size=[3.0,  6.0, 0.0125], pos=[9.0 + n*10, 0.0, 5.0])
            segments_handle.add("geom", dclass="floor", size=[2.0, 3.0, 0.0125], pos=[14.0 + n*10, s*3.0, 5.0])
            segments_handle.add("geom", dclass="wall", size=[5.0, 0.5, 2.0], pos=[11.0 + n*10, 5.5, 7.0])
            segments_handle.add("geom", dclass="wall", size=[5.0, 0.5, 2], pos=[11.0 + n*10, -5.5, 7.0])
            segments_handle.add("geom", dclass="wall", size=[2.0, 0.5, 2.0], pos=[14.0 + n*10, 0.0, 7.0])
            segments_handle.add("geom", type="cylinder", size=[0.2, 3.0], euler=[90.0, 0.0, 0.0], pos=[12.0 + n*10, -s*3.0, 4.8125], rgba=[0.2, 0.3, 0.4, 1.0])
            segments_handle.add("geom", type="cylinder", size=[0.2, 3.0], euler=[90.0, 0.0, 0.0], pos=[16.0 + n*10, -s*3.0, 4.8125], rgba=[0.2, 0.3, 0.4, 1.0])

        handle = xml_handle.find("body", "segments")
        for n, s in enumerate(sequence):
            _add_segment(handle, n, s)

        return xml_handle

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position and so forth.
        """

        cam = self.viewer.cam
        cam.azimuth = 360.0
        cam.distance = 175.0
        cam.elevation = -10.0

        cam.fixedcamid = -1
        cam.type = mujoco_py.generated.const.CAMERA_TRACKING
        cam.trackbodyid = 2

    @staticmethod
    def _save_xml_handle(xml_handle, tmp_dir_name, file_name="tmp_model.xml"):
        """
        Save the Mujoco XML handle to a file at tmp_dir_name. If tmp_dir_name is None,
        a temporary directory is created at /tmp.

        Args:
            xml_handle: Mujoco XML handle.
            tmp_dir_name (str): Path to temporary directory. If None, a
            temporary directory is created at /tmp.

        Returns:
            String of the save path.

        """

        if tmp_dir_name is not None:
            assert os.path.exists(tmp_dir_name), "specified directory (\"%s\") does not exist." % tmp_dir_name

        dir = mkdtemp(dir=tmp_dir_name)
        file_path = os.path.join(dir, file_name)

        # dump data
        mjcf.export_with_assets(xml_handle, dir, file_name)

        return file_path