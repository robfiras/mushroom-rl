
import time
import sys
from abc import abstractmethod
import mujoco

from pathlib import Path
import os

import numpy as np
from scipy import interpolate


from mushroom_rl.environments.mujoco import MuJoCo, ObservationType
from pathlib import Path

from mushroom_rl.utils import spaces
from mushroom_rl.utils.angles import quat_to_euler
from mushroom_rl.utils.running_stats import *
from mushroom_rl.utils.mujoco import *
from mushroom_rl.environments.mujoco_envs.humanoids.trajectory import Trajectory

from mushroom_rl.environments.mujoco_envs.humanoids.reward import NoGoalReward, CustomReward

import matplotlib.pyplot as plt


# optional imports
try:
    mujoco_viewer_available = True
    import mujoco_viewer
except ModuleNotFoundError:
    mujoco_viewer_available = False

class BaseQuadruped(MuJoCo):
    """
    Mujoco simulation of unitree A1 model
    """
    def __init__(self, xml_path, action_spec, observation_spec, collision_groups=[], gamma=0.99, horizon=1000, n_substeps=10,  goal_reward=None,
                 goal_reward_params=None, traj_params=None, timestep=0.001, use_action_clipping=True):
        """
        Constructor.
        """

        super().__init__(xml_path, action_spec, observation_spec, gamma=gamma, horizon=horizon, n_substeps=n_substeps,
                         timestep=timestep, collision_groups=collision_groups)


        self.use_action_clipping = use_action_clipping
        self.goal_reward = NoGoalReward()

        self.info.observation_space = spaces.Box(*self._get_observation_space())

        if use_action_clipping:
            # clip action space between -1,1
            low, high = self.info.action_space.low.copy(), \
                        self.info.action_space.high.copy()

            self.norm_act_mean = (high + low) / 2.0
            self.norm_act_delta = (high - low) / 2.0
            self.info.action_space.low[:] = -1.0
            self.info.action_space.high[:] = 1.0



        self.mean_grf = RunningAveragedWindow(shape=(12,),
                                              window_size=n_substeps)


        if traj_params:
            self.trajectory = Trajectory(keys=self.get_all_observation_keys(), **traj_params)
            print(self.trajectory)
        else:
            self.trajectory = None


    def _get_observation_space(self):
        sim_low, sim_high = (self.info.observation_space.low[2:],
                             self.info.observation_space.high[2:])

        grf_low, grf_high = (-np.ones((12,)) * np.inf,
                             np.ones((12,)) * np.inf)

        r_low, r_high = self.goal_reward.get_observation_space()

        return (np.concatenate([sim_low, grf_low, r_low]),
                np.concatenate([sim_high, grf_high, r_high]))

    def _create_observation(self, obs):
        """
        Creates full vector of observations:
        needs to be changed for freejoint
        """
        obs = np.concatenate([obs[2:],
                              self.mean_grf.mean / 1000.,
                              self.goal_reward.get_observation(),
                              ]).flatten()

        return obs


    def reward(self, state, action, next_state, absorbing):
        #return -1 if self.has_fallen(self._obs) else 1
        goal_reward = self.goal_reward(state, action, next_state)
        return goal_reward


    def setup(self):
        self.goal_reward.reset_state()
        if self.trajectory is not None:
            len_qpos, len_qvel = self.len_qpos_qvel()
            qpos, qvel = self.trajectory.reset_trajectory(len_qpos, len_qvel)
            self._data.qpos = qpos
            self._data.qvel = qvel

    def _simulation_pre_step(self):
        #self._data.qfrc_applied[self._action_indices] = self._data.qfrc_bias[:12]
        #print(self._data.qfrc_bias[:12])
        #self._data.ctrl[self._action_indices] = self._data.qfrc_bias[:12] + self._data.ctrl[self._action_indices]
        #print(self._data.qfrc_bias[:12])
        #self._data.qfrc_applied[self._action_indices] = self._data.qfrc_bias[:12] + self._data.qfrc_applied[self._action_indices]
        #self._data.qfrc_actuator[self._action_indices] += self._data.qfrc_bias[:12]
        #self._data.ctrl[self._action_indices] += self._data.qfrc_bias[:12]

        pass

    #def _compute_action(self, obs, action):
    #    gravity = self._data.qfrc_bias[self._action_indices]
    #    action = action+gravity
    #    return action


    def _preprocess_action(self, action):
        if self.use_action_clipping:
            unnormalized_action = ((action.copy() * self.norm_act_delta) + self.norm_act_mean)
            return unnormalized_action


        return action

    def _simulation_post_step(self):
        grf = np.concatenate([self._get_collision_force("floor", "foot_FL")[:3],
                              self._get_collision_force("floor", "foot_FR")[:3],
                              self._get_collision_force("floor", "foot_RL")[:3],
                              self._get_collision_force("floor", "foot_RR")[:3]])

        self.mean_grf.update_stats(grf)
        #self._data.qfrc_applied[self._action_indices] = self._data.qfrc_bias[self._action_indices] + self._data.qfrc_applied[self._action_indices]

        #print(self._data.qfrc_bias[:12])


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

    def create_dataset(self, ignore_keys=[], normalizer=None):
        if self.trajectory is not None:
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

        len_qpos, len_qvel = self.len_qpos_qvel()
        qpos, qvel = self.trajectory.reset_trajectory(len_qpos, len_qvel, substep_no=1)
        print(qpos, qvel)

        self._data.qpos = qpos
        self._data.qvel = qvel
        while True:
            sample = self.trajectory.get_next_sample()
            obs_spec = self.obs_helper.observation_spec
            assert len(sample) == len(obs_spec)

            # self._data.qpos = sample[0:len_qpos]
            # self._data.qvel = sample[len_qpos:len_qpos + len_qvel]

            for key_name_ot, value in zip(obs_spec, sample):
                key, name, ot = key_name_ot
                if ot == ObservationType.JOINT_POS:
                    self._data.joint(name).qpos = value
                elif ot == ObservationType.JOINT_VEL:
                    self._data.joint(name).qvel = value

            mujoco.mj_forward(self._model, self._data)

            obs = self._create_observation(sample)
            if self.has_fallen(obs):
                print("Has Fallen!")

            self.render()

        # def play_trajectory_demo(self, freq=200, view_from_other_side=False):
        #     """
        #     Plays a demo of the loaded trajectory by forcing the model
        #     positions to the ones in the reference trajectory at every step
        #
        #     """
        #
        #     assert self.trajectory is not None
        #
        #     # Todo: different camera view not working
        #     # cam = mujoco.MjvCamera()
        #     # mujoco.mjv_defaultCamera(cam)
        #     # viewer._render_every_frame = False
        #     # if view_from_other_side:
        #     #     #self._model.cam_pos = [3., 2., 0.0]
        #     #     cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        #     #     cam.trackbodyid = 0
        #     #     cam.distance *= 0.3
        #     #     cam.elevation = -0  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        #     #     cam.azimuth = 270
        #
        #     len_qpos, len_qvel = self.len_qpos_qvel()
        #     qpos, qvel = self.trajectory.reset_trajectory(len_qpos, len_qvel, substep_no=1)
        #     self._data.qpos = qpos
        #     self._data.qvel = qvel
        #     while True:
        #         sample = self.trajectory.get_next_sample()
        #
        #         self._data.qpos = sample[0:len_qpos]
        #         self._data.qvel = sample[len_qpos:len_qpos+len_qvel]
        #
        #         mujoco.mj_forward(self._model, self._data)
        #
        #         obs = self._create_observation(sample)
        #         if self.has_fallen(obs):
        #             print("Has Fallen!")
        #
        #         self.render()

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
            qpos = curr_qpos + self.dt * qvel
            sample[:len(qpos)] = qpos

            obs_spec = self.obs_helper.observation_spec
            assert len(sample) == len(obs_spec)

            for key_name_ot, value in zip(obs_spec, sample):
                key, name, ot = key_name_ot
                if ot == ObservationType.JOINT_POS:
                    self._data.joint(name).qpos = value
                elif ot == ObservationType.JOINT_VEL:
                    self._data.joint(name).qvel = value

            mujoco.mj_forward(self._model, self._data)

            # save current qpos
            curr_qpos = self._data.qpos

            obs = self._create_observation(sample)
            if self.has_fallen(obs):
                print("Has Fallen!")

            self.render()

    def len_qpos_qvel(self):
        keys = self.get_all_observation_keys()
        len_qpos = len([key for key in keys if key.startswith("q_")])
        len_qvel = len([key for key in keys if key.startswith("dq_")])
        return len_qpos, len_qvel

    def play_action_demo(self, action_path, states_path, control_dt=0.01, demo_dt=0.01, dataset_path=None, ignore_keys=[]):
        """

        Plays a demo of the loaded actions by using the actions in action_path.
        action_path: path to the .npz file. Should be in format (number of samples/steps, action dimension)
        states_path: path to states.npz file, for initial position; should be in format like for play_trajectory_demo
        control_dt: model control frequency
        demo_dt: freqency the data was collected
        dataset_path: if set, method creates a dataset with actions, states, episode_starts, next_states, absorbing, rewards
        ignore_keys if dataset_path is set; index of keys to ignore in dataset
        make sure action clipping is off

        """
        assert (ignore_keys==[]) or bool(dataset_path), "ignore_keys only available if dataset_path set"

        # to get the same init position
        trajectory_files = np.load(states_path, allow_pickle=True)
        trajectory = np.array([trajectory_files[key] for key in trajectory_files.keys()])
        print(type(trajectory[0]))
        print(trajectory.shape)

        obs_spec = self.obs_helper.observation_spec
        #set x and y to 0: be carefull need to be at 0,1
        trajectory[0, :] -= trajectory[0, 0]
        trajectory[1, :] -= trajectory[1, 0]
        for key_name_ot, value in zip(obs_spec, trajectory[:,0]):
            key, name, ot = key_name_ot
            if ot == ObservationType.JOINT_POS:
                self._data.joint(name).qpos = value
            elif ot == ObservationType.JOINT_VEL:
                self._data.joint(name).qvel = value





        #np.set_printoptions(threshold=sys.maxsize)
        action_files = np.load(action_path, allow_pickle=True)

        actions = np.array([action_files[key] for key in action_files.keys()])[0]


        #scale frequencies
        if demo_dt != control_dt:
            new_demo_sampling_factor = demo_dt / control_dt
            x = np.arange(actions.shape[0])
            x_new = np.linspace(0, actions.shape[0]-1, round(actions.shape[0]*new_demo_sampling_factor),
                                endpoint=True)
            actions = interpolate.interp1d(x, actions, kind="cubic", axis=0)(x_new)

            trajectory = interpolate.interp1d(x, trajectory, kind="cubic", axis=1)(x_new)

        true_pos=[]
        set_point=[]

        if (dataset_path): #format shape[1] is nr motors/observable points
            actions_dataset=[]
            states_dataset=[]
            episode_starts_dataset= [False]*actions.shape[0]
            episode_starts_dataset[0]=True
            #next_states_dataset=[]
            absorbing_dataset=[]
            rewards_dataset=[]



        for i in np.arange(actions.shape[0]):#1024
            #time.sleep(.1)
            if(dataset_path and i>1024):
                actions_dataset.append(list(actions[i]))
                states_dataset_temp = list(self._data.qpos[:]) + list(self._data.qvel[:])

                for x in sorted(ignore_keys, reverse=True):
                    del states_dataset_temp[x]
                states_dataset.append(states_dataset_temp)
                absorbing_dataset.append(self.is_absorbing(self._obs))
                temp_obs=self._obs
            action = actions[i]
            true_pos.append(list(self._data.qpos[6:]))
            set_point.append(trajectory[6:18,i])
            nstate, _, absorbing, _ = self.step(action)
            self.render()
            if(dataset_path and i>1024):

                #next_states_dataset_temp = list(self._data.qpos[:]) + list(self._data.qvel[:])
                #for x in sorted(ignore_keys, reverse=True):
                #    del next_states_dataset_temp[x]
                #next_states_dataset.append(next_states_dataset_temp)


                rewards_dataset.append(self.reward(temp_obs, action, self._obs, self.is_absorbing(self._obs)))

        if (dataset_path):
            np.savez(os.path.join(dataset_path, 'dataset_unitreeA1_IRL.npz'),
                     actions=actions_dataset, states=states_dataset, episode_starts=episode_starts_dataset,
                      absorbing=absorbing_dataset, rewards=rewards_dataset)# next_states=next_states_dataset,


        true_pos=np.array(true_pos)
        set_point=np.array(set_point)
        # --------------------------------------------------------------------------------------------------------------
        data= {
            "setpoint": set_point[:,0],
            "actual pos" : true_pos[:,0]
        }

        fig = plt.figure()
        ax = fig.gca()
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for i, v in enumerate(data.items()):
            ax.plot(v[1], color=colors[i], linestyle='-', label=v[0])
        plt.legend(loc=4)
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.savefig("hip.png")

        # --------------------------------------------------------------------------------------------------------------
        data = {
            "setpoint": set_point[:, 1],
            "actual pos": true_pos[:, 1]
        }

        fig = plt.figure()
        ax = fig.gca()
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for i, v in enumerate(data.items()):
            ax.plot(v[1], color=colors[i], linestyle='-', label=v[0])
        plt.legend(loc=4)
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.savefig("thigh.png")

        # --------------------------------------------------------------------------------------------------------------

        data = {
            "setpoint": set_point[:, 2],
            "actual pos": true_pos[:, 2]
        }

        fig = plt.figure()
        ax = fig.gca()
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for i, v in enumerate(data.items()):
            ax.plot(v[1], color=colors[i], linestyle='-', label=v[0])
        plt.legend(loc=4)
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.savefig("calf.png")

        # --------------------------------------------------------------------------------------------------------------

        data = {
            "hip error": set_point[:, 0]-true_pos[:, 0],
            "thigh error": set_point[:, 1]-true_pos[:, 1],
            "calf error": set_point[:, 2]-true_pos[:, 2]
        }

        fig = plt.figure()
        ax = fig.gca()
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for i, v in enumerate(data.items()):
            ax.plot(v[1], color=colors[i], linestyle='-', label=v[0])
        plt.legend(loc=4)
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.savefig("error.png")




        




