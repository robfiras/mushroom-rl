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

from mushroom_rl.environments.mujoco_envs.humanoids.reward import NoGoalReward, CustomReward, TargetVelocityReward

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
                 goal_reward_params=None, traj_params=None, random_start=True, init_step_no=None, init_traj_no=None, timestep=0.001, use_foot_forces=True):
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
            self.goal_reward = CustomReward(**goal_reward_params)
        elif goal_reward == "target_velocity":
            x_vel_idx = self.get_obs_idx("dq_pelvis_tx")
            assert len(x_vel_idx) == 1
            x_vel_idx = x_vel_idx[0]
            self.goal_reward = TargetVelocityReward(x_vel_idx=x_vel_idx, **goal_reward_params)
        elif goal_reward is None:
            self.goal_reward = NoGoalReward()
        else:
            raise NotImplementedError("The specified goal reward has not been"
                                      "implemented: ", goal_reward)

        # optionally use foot forces in the observation space
        self._use_foot_forces = use_foot_forces

        self.info.observation_space = spaces.Box(*self._get_observation_space())

        # we want the action space to be between -1 and 1
        low, high = self.info.action_space.low.copy(),\
                    self.info.action_space.high.copy()
        self.norm_act_mean = (high + low) / 2.0
        self.norm_act_delta = (high - low) / 2.0
        self.info.action_space.low[:] = -1.0
        self.info.action_space.high[:] = 1.0

        # mask to get kinematic observations (-2 for neglecting x and z)
        self._kinematic_obs_mask = np.arange(len(observation_spec) - 2)

        # setup a running average window for the mean ground forces
        self.mean_grf = RunningAveragedWindow(shape=(12,),
                                              window_size=n_substeps)

        if traj_params:
            self.trajectory = Trajectory(keys=self.get_all_observation_keys(),
                                         low=self.info.observation_space.low,
                                         high=self.info.observation_space.high,
                                         joint_pos_idx=self.obs_helper.joint_pos_idx,
                                         observation_spec=self.obs_helper.observation_spec,
                                         **traj_params)
        else:
            self.trajectory = None

        if not traj_params and random_start:
            raise ValueError("Random start not possible without trajectory data.")
        elif not traj_params and init_step_no is not None:
            raise ValueError("Setting an initial step is not possible without trajectory data.")
        elif init_step_no is not None and random_start:
            raise ValueError("Either use a random start or set an initial step, not both.")
        elif traj_params is not None and not (random_start or init_step_no is not None):
            raise ValueError("You have specified a trajectory, you have to use either a random start or "
                             "set an initial step")

        if init_step_no is not None and init_traj_no is None: #for cases before the new implementation: set traj_no to 0
            init_traj_no = 0

        self._random_start = random_start # TODO all occ of init_step_no > init_traj_no
        self._init_step_no = init_step_no
        self._init_traj_no = init_traj_no

    def _get_observation_space(self):
        sim_low, sim_high = (self.info.observation_space.low[2:],
                             self.info.observation_space.high[2:])

        if self._use_foot_forces:
            grf_low, grf_high = (-np.ones((12,)) * np.inf,
                                 np.ones((12,)) * np.inf)
            r_low, r_high = self.goal_reward.get_observation_space()
            return (np.concatenate([sim_low, grf_low, r_low]),
                    np.concatenate([sim_high, grf_high, r_high]))
        else:
            r_low, r_high = self.goal_reward.get_observation_space()
            return (np.concatenate([sim_low, r_low]),
                    np.concatenate([sim_high, r_high]))

    def _create_observation(self, obs):
        """
        Creates full vector of observations:
        """

        if self._use_foot_forces:
            #flatten for multi dim obs spec
            temp = []
            for state in obs[2:]:
                if type(state) == np.ndarray:
                    temp = temp + list(state)
                else:
                    temp.append(state)
            obs = np.concatenate([temp, self._goals,
                                  self.mean_grf.mean / 1000.,
                                  self.goal_reward.get_observation(), # TODO remove the goals of the reward
                                  ]).flatten() # add dtype=object for numpy 1.20
        else:
            # flatten for multi dim obs spec
            temp = []
            for state in obs[2:]:
                if type(state) == np.ndarray:
                    temp = temp + list(state)
                else:
                    temp.append(state)
            obs = np.concatenate([temp, self._goals,
                                  self.goal_reward.get_observation(),
                                  ])
        return obs

    def reward(self, state, action, next_state, absorbing):
        goal_reward = self.goal_reward(state, action, next_state)
        return goal_reward

    def setup(self, substep_no=None):
        self.goal_reward.reset_state()
        if self.trajectory is not None:
            if self._random_start:
                sample = self.trajectory.reset_trajectory()
            else:
                sample = self.trajectory.reset_trajectory(self._init_step_no, self._init_traj_no)

            self.set_qpos_qvel(sample)

    def _preprocess_action(self, action):
        unnormalized_action = ((action.copy() * self.norm_act_delta) + self.norm_act_mean)
        return unnormalized_action

    def _simulation_post_step(self):
        if self._use_foot_forces:
            grf = np.concatenate([self._get_collision_force("floor", "foot_r")[:3],
                                  self._get_collision_force("floor", "front_foot_r")[:3],
                                  self._get_collision_force("floor", "foot_l")[:3],
                                  self._get_collision_force("floor", "front_foot_l")[:3]])

            self.mean_grf.update_stats(grf)

    def is_absorbing(self, obs):
        return self.has_fallen(obs)

    def get_kinematic_obs_mask(self):
        return self._kinematic_obs_mask

    def get_obs_idx(self, name):
        idx = self.obs_helper.obs_idx_map[name]
        return [i-2 for i in idx]

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

    def create_dataset(self, ignore_keys=[], normalizer=None):
        if self.trajectory is not None :
            return self.trajectory.create_dataset(ignore_keys=ignore_keys, normalizer=normalizer)
        else:
            raise ValueError("No trajecory was passed to the environment. To create a dataset,"
                             "pass a trajectory to the dataset first.")

    def play_trajectory_demo(self, freq=200, view_from_other_side=False):
        """
        Plays a demo of the loaded trajectory by forcing the model
        positions to the ones in the reference trajectory at every step

        """
        assert self.trajectory is not None

        ##Todo: different camera view not working
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
        sample = self.trajectory.reset_trajectory(substep_no=1, traj_no=0)
        self.set_qpos_qvel(sample)
        rewards = []
        while True:
            sample = self.trajectory.get_next_sample()
            #sample[-1] = 1
            self.set_qpos_qvel(sample)



            self._simulation_pre_step()

            mujoco.mj_forward(self._model, self._data)

            self._simulation_post_step()



            obs = self._create_observation(sample)
            if self.has_fallen(obs):
                print("Has Fallen!")
            #print(len(rewards))
            rewards.append(self.reward(obs, None, None, False))
            #print(self._goals)
            #if len(rewards) == 200:
             #   time.sleep(1)
            #if len(rewards) == 500:
             #   break


            self.render()
        return rewards





    def play_trajectory_demo_from_velocity(self, freq=200, view_from_other_side=False):
        """
        Plays a demo of the loaded trajectory by forcing the model
        positions to the ones in the reference trajectory at every steps
        """

        assert self.trajectory is not None

        sample = self.trajectory.reset_trajectory(substep_no=1, traj_no=0)
        self.set_qpos_qvel(sample)
        len_qpos, len_qvel = self.len_qpos_qvel()
        #len_qvel+=1
        curr_qpos = sample[0:len_qpos]
        #curr_qpos = np.append(curr_qpos, 0)

        while True:

            sample = np.array(self.trajectory.get_next_sample())
            #if self.
            qvel = sample[len_qpos:len_qpos + len_qvel]
            #qvel = np.append(qvel, 0)
            print("qpos", curr_qpos.shape)
            qpos = curr_qpos + self.dt * qvel
            sample[:len(qpos)] = qpos

            self.set_qpos_qvel(sample)



            self._simulation_pre_step()

            mujoco.mj_forward(self._model, self._data)

            self._simulation_post_step()



            # save current qpos
            curr_qpos = self.get_joint_pos()

            obs = self._create_observation(sample)
            if self.has_fallen(obs):
                print("Has Fallen!")

            self.render()

    def set_qpos_qvel(self, sample):
        obs_spec = self.obs_helper.observation_spec

        #handle goals
        self._goals = np.array(sample[len(obs_spec):], dtype=float)
        sample = sample[:len(obs_spec)]


        assert len(sample) == len(obs_spec)

        for key_name_ot, value in zip(obs_spec, sample):
            key, name, ot = key_name_ot
            if ot == ObservationType.JOINT_POS:
                self._data.joint(name).qpos = value
            elif ot == ObservationType.JOINT_VEL:
                self._data.joint(name).qvel = value
            elif ot == ObservationType.SITE_ROT:
                self._data.site(name).xmat = value
                #TODO tim shouldn't be here works with play_demo still?
                if name == "dir_arrow" and not hasattr(self, '_direction_xmat'):
                    self._direction_xmat = value
                    no_arrrow_transformation = np.dot(value.reshape((3, 3)),
                                                      np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).reshape((9,))
                    self._direction_angle = np.arctan2(no_arrrow_transformation[3], no_arrrow_transformation[0])


    def get_joint_pos(self):
        return self.obs_helper.get_joint_pos_from_obs(self.obs_helper.build_obs(self._data))

    def get_joint_vel(self):
        return self.obs_helper.get_joint_vel_from_obs(self.obs_helper.build_obs(self._data))

    def len_qpos_qvel(self):
        keys = self.get_all_observation_keys()
        len_qpos = len([key for key in keys if key.startswith("q_")])
        len_qvel = len([key for key in keys if key.startswith("dq_")])
        return len_qpos, len_qvel

    @staticmethod
    def has_fallen(obs):
        raise NotImplementedError