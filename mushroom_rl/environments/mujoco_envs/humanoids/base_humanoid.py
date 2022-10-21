import time
from abc import abstractmethod
import mujoco

from mushroom_rl.environments.mujoco import MuJoCo, ObservationType
from pathlib import Path

from mushroom_rl.utils import spaces
from mushroom_rl.utils.angles import quat_to_euler
from mushroom_rl.utils.running_stats import *
from mushroom_rl.utils.mujoco import *
from mushroom_rl.environments.mujoco_envs.humanoids.trajectory import Trajectory

from mushroom_rl.environments.mujoco_envs.humanoids.reward import NoGoalReward, CustomReward

# optional imports
try:
    mujoco_viewer_available = True
    import mujoco_viewer
except ModuleNotFoundError:
    mujoco_viewer_available = False


class BaseHumanoid(MuJoCo):
    """
    Base humanoid class for all kinds of humanoid environemnts.

    """
    def __init__(self, xml_path, action_spec, observation_spec, collision_groups=[], gamma=0.99, horizon=1000, n_substeps=10,  goal_reward=None,
                 goal_reward_params=None, traj_params=None, timestep=0.001):
        """
        Constructor.

        """

        super().__init__(xml_path, action_spec, observation_spec, gamma=gamma, horizon=horizon,
                         n_substeps=n_substeps, timestep=timestep, collision_groups=collision_groups)

        # specify the reward
        #if goal_reward == "changing_vel":
        #    self.goal_reward = ChangingVelocityTargetReward(self._sim, **goal_reward_params)
        #elif goal_reward == "no_goal_rand_init":
        #    self.goal_reward = NoGoalRewardRandInit(self._sim, **goal_reward_params)
        # todo: update all rewards to new mujoco interface and not rely on sim anymore
        if goal_reward == "custom":
            self.goal_reward = CustomReward(model=self._model, data=self._data, keys=self.get_all_observation_keys(),
                                            **goal_reward_params)
        elif goal_reward is None:
            self.goal_reward = NoGoalReward()
        else:
            raise NotImplementedError("The specified goal reward has not been"
                                      "implemented: ", goal_reward)

        self.info.observation_space = spaces.Box(*self._get_observation_space())

        # we want the action space to be between -1 and 1
        low, high = self.info.action_space.low.copy(),\
                    self.info.action_space.high.copy()
        self.norm_act_mean = (high + low) / 2.0
        self.norm_act_delta = (high - low) / 2.0
        self.info.action_space.low[:] = -1.0
        self.info.action_space.high[:] = 1.0

        if traj_params:
            self.trajectory = Trajectory(keys=self.get_all_observation_keys(), **traj_params)
        else:
            self.trajectory = None

    def _get_observation_space(self):
        sim_low, sim_high = (self.info.observation_space.low[2:],
                             self.info.observation_space.high[2:])

        # TODO: add ground forces here once collision groups are added
        #grf_low, grf_high = (-np.ones((12,)) * np.inf,
        #                     np.ones((12,)) * np.inf)

        r_low, r_high = self.goal_reward.get_observation_space()

        return (np.concatenate([sim_low, r_low]),   # TODO: add grf here as well
                np.concatenate([sim_high, r_high]))

    def _create_observation(self, obs):
        """
        Creates full vector of observations:
        """
        obs = np.concatenate([obs[2:],
                              # TODO: add grf forces here
                              #self._get_collision_force("ground", "right_foot_back")[:3]/10000.,
                              #self._get_collision_force("ground", "right_foot_front")[:3]/10000.,
                              #self._get_collision_force("ground", "left_foot_back")[:3]/10000.,
                              #self._get_collision_force("ground", "left_foot_front")[:3]/10000.,
                              self.goal_reward.get_observation(),
                              ]).flatten()

        return obs

    def reward(self, state, action, next_state, absorbing):
        goal_reward = self.goal_reward(state, action, next_state)
        return goal_reward

    def setup(self):
        self.goal_reward.reset_state()
        if self.trajectory is not None:
            len_qpos, len_qvel = self.len_qpos_qvel()
            qpos, qvel = self.trajectory.reset_trajectory(len_qpos, len_qvel)
            self._data.qpos = qpos
            self._data.qvel = qvel

    def _preprocess_action(self, action):
        unnormalized_action = ((action.copy() * self.norm_act_delta) + self.norm_act_mean)
        return unnormalized_action

    def is_absorbing(self, obs):
        return self.has_fallen(obs)

    def render(self):

        if self._viewer is None:
            if mujoco_viewer_available:
                self._viewer = mujoco_viewer.MujocoViewer(self._model, self._data)
            else:
                self._viewer = MujocoGlfwViewer(self._model, self.dt, **self._viewer_params)

        if mujoco_viewer_available:
            self._viewer.render()
            time.sleep(self.dt)
        else:
            self._viewer.render(self._data)

    def play_trajectory_demo(self, freq=200, view_from_other_side=False):
        """
        Plays a demo of the loaded trajectory by forcing the model
        positions to the ones in the reference trajectory at every step

        """

        assert self.trajectory is not None

        # Todo: different camera view not working
        # cam = mujoco.MjvCamera()
        # mujoco.mjv_defaultCamera(cam)
        # viewer._render_every_frame = False
        # if view_from_other_side:
        #     #self._model.cam_pos = [3., 2., 0.0]
        #     cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        #     cam.trackbodyid = 0
        #     cam.distance *= 0.3
        #     cam.elevation = -0  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        #     cam.azimuth = 270

        len_qpos, len_qvel = self.len_qpos_qvel()
        qpos, qvel = self.trajectory.reset_trajectory(len_qpos, len_qvel, substep_no=1)
        self._data.qpos = qpos
        self._data.qvel = qvel
        while True:
            sample = self.trajectory.get_next_sample()

            self._data.qpos = sample[0:len_qpos]
            self._data.qvel = sample[len_qpos:len_qpos+len_qvel]

            mujoco.mj_forward(self._model, self._data)

            obs = self._create_observation(sample)
            if self.has_fallen(obs):
                print("Has Fallen!")

            self.render()

    def play_trajectory_demo_from_velocity(self, freq=200, view_from_other_side=False):
        """
        Plays a demo of the loaded trajectory by forcing the model
        positions to the ones in the reference trajectory at every steps
        """

        assert self.trajectory is not None

        len_qpos, len_qvel = self.len_qpos_qvel()
        qpos, qvel = self.trajectory.reset_trajectory(len_qpos, len_qvel, substep_no=1)
        self._data.qpos = qpos
        self._data.qvel = qvel
        curr_qpos = qpos
        while True:

            sample = self.trajectory.get_next_sample()
            qvel = sample[len_qpos:len_qpos + len_qvel]

            self._data.qpos = curr_qpos + self.dt * qvel

            mujoco.mj_forward(self._model, self._data)

            # save current qpos
            curr_qpos = self._data.qpos

            self.render()


    def len_qpos_qvel(self):
        keys = self.get_all_observation_keys()
        len_qpos = len([key for key in keys if key.startswith("q_")])
        len_qvel = len([key for key in keys if key.startswith("dq_")])
        return len_qpos, len_qvel

    @staticmethod
    def has_fallen(obs):
        raise NotImplementedError