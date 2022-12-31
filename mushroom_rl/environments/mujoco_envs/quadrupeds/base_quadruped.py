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
from mushroom_rl.environments.mujoco_envs.humanoids.base_humanoid import BaseHumanoid

from mushroom_rl.environments.mujoco_envs.humanoids.reward import NoGoalReward, CustomReward

import matplotlib.pyplot as plt

# optional imports
try:
    mujoco_viewer_available = True
    import mujoco_viewer
except ModuleNotFoundError:
    mujoco_viewer_available = False


class BaseQuadruped(BaseHumanoid):
    """
    Mujoco simulation of unitree A1 model
    """


    # def _simulation_pre_step(self):
        # self._data.qfrc_applied[self._action_indices] = self._data.qfrc_bias[:12]
        # print(self._data.qfrc_bias[:12])
        # self._data.ctrl[self._action_indices] = self._data.qfrc_bias[:12] + self._data.ctrl[self._action_indices]
        # print(self._data.qfrc_bias[:12])
        # self._data.qfrc_applied[self._action_indices] = self._data.qfrc_bias[:12] + self._data.qfrc_applied[self._action_indices]
        # self._data.qfrc_actuator[self._action_indices] += self._data.qfrc_bias[:12]
        # self._data.ctrl[self._action_indices] += self._data.qfrc_bias[:12]

        #    pass




    #def _compute_action(self, obs, action):
    #    gravity = self._data.qfrc_bias[self._action_indices]
    #    action = action+gravity
    #    return action


    def _simulation_post_step(self):
        grf = np.concatenate([self._get_collision_force("floor", "foot_FL")[:3],
                              self._get_collision_force("floor", "foot_FR")[:3],
                              self._get_collision_force("floor", "foot_RL")[:3],
                              self._get_collision_force("floor", "foot_RR")[:3]])

        self.mean_grf.update_stats(grf)
        # self._data.qfrc_applied[self._action_indices] = self._data.qfrc_bias[self._action_indices] + self._data.qfrc_applied[self._action_indices]

        # print(self._data.qfrc_bias[:12])

    def create_dataset(self, data_path, ignore_keys=[], normalizer=None, only_state=True, use_next_states=True):
        """
        creates dataset.
        If data_path is set only states has to be false -> creates dataset with states, actions (next_states)
        else dataset with only states is created
        scales/interpolates to the correct frequencies
        dataset needs to be in the same order as self.obs_helper.observation_spec
        """
        if only_state and use_next_states:

            trajectory_files = np.load(data_path, allow_pickle=True)
            trajectory_files = {k: d for k, d in trajectory_files.items()}  # convert to dict to be mutable

            keys = trajectory_files.keys()

            trajectory = np.array([trajectory_files[key] for key in keys])

            demo_dt = self.trajectory.traj_dt
            control_dt = self.trajectory.control_dt


            #interpolation
            if demo_dt != control_dt:
                new_traj_sampling_factor = demo_dt / control_dt

                x = np.arange(trajectory.shape[1])
                x_new = np.linspace(0, trajectory.shape[1] - 1, round(trajectory.shape[1] * new_traj_sampling_factor),
                                    endpoint=True)

                trajectory = interpolate.interp1d(x, trajectory, kind="cubic", axis=1)(x_new)

            # create a dict and extract all elements except the ones specified in ignore_keys.
            all_data = dict(zip(keys, list(trajectory)))
            for ikey in ignore_keys:
                del all_data[ikey]
            traj = list(all_data.values())
            states = np.transpose(np.array(traj))

            # normalize if needed
            if normalizer:
                normalizer.set_state(dict(mean=np.mean(states, axis=0),
                                          var=1 * (np.std(states, axis=0) ** 2),
                                          count=1))
                states = np.array([normalizer(st) for st in states])

            # convert to dict with states and next_states
            new_states = states[:-1]
            new_next_states = states[1:]
            absorbing = np.zeros(len(new_states))

            return dict(states=new_states, next_states=new_next_states, absorbing=absorbing)





        elif not only_state:

            # change name in ignore keys into
            obs_keys = list(np.array(self.obs_helper.observation_spec)[:, 0])
            ignore_index = []
            for key in ignore_keys:
                ignore_index.append(obs_keys.index(key))

            dataset = dict()

            # load expert training data
            expert_files = np.load(data_path)
            dataset["states"] = expert_files["states"]
            dataset["actions"] = expert_files["actions"]

            dataset["episode_starts"] = expert_files["episode_starts"]
            assert dataset["episode_starts"][0] and [x for x in dataset["episode_starts"][1:] if
                                                     x == True] == [], "Implementation only for one long trajectory"

            # remove ignore indices
            for i in sorted(ignore_index, reverse=True):
                dataset["states"] = np.delete(dataset["states"], i, 1)

            # scale frequencies
            demo_dt = self.trajectory.traj_dt
            control_dt = self.trajectory.control_dt
            if demo_dt != control_dt:
                new_demo_sampling_factor = demo_dt / control_dt
                x = np.arange(dataset["states"].shape[0])
                x_new = np.linspace(0, dataset["states"].shape[0] - 1,
                                    round(dataset["states"].shape[0] * new_demo_sampling_factor),
                                    endpoint=True)
                dataset["states"] = interpolate.interp1d(x, dataset["states"], kind="cubic", axis=0)(x_new)
                dataset["actions"] = interpolate.interp1d(x, dataset["actions"], kind="cubic", axis=0)(x_new)
                dataset["episode_starts"] = [False] * x_new
                dataset["episode_starts"][0] = True

            # maybe we have next action and next next state
            try:
                dataset["next_actions"] = expert_files["next_actions"]
                dataset["next_next_states"] = expert_files["next_next_states"]
                # remove ignore indices
                for i in sorted(ignore_index, reverse=True):
                    dataset["next_next_states"] = np.delete(dataset["next_next_states"], i, 1)
                # scaling
                if demo_dt != control_dt:
                    dataset["next_actions"] = interpolate.interp1d(x, dataset["next_actions"], kind="cubic", axis=0)(
                        x_new)
                    dataset["next_next_states"] = interpolate.interp1d(x, dataset["next_next_states"], kind="cubic",
                                                                       axis=0)(x_new)

            except KeyError as e:
                print("Did not find next action or next next state.")

            # maybe we have next states and dones in the dataset
            try:
                dataset["next_states"] = expert_files["next_states"]
                dataset["absorbing"] = expert_files["absorbing"]

                # remove ignore indices
                for i in sorted(ignore_index, reverse=True):
                    dataset["next_states"] = np.delete(dataset["next_states"], i, 1)

                # scaling
                if demo_dt != control_dt:
                    dataset["next_states"] = interpolate.interp1d(x, dataset["next_states"], kind="cubic", axis=0)(
                        x_new)
                    # TODO: not sure about this
                    dataset["absorbing"] = interpolate.interp1d(x, dataset["absorbing"], kind="cubic", axis=0)(x_new)

            except KeyError as e:
                print("Warning Dataset: %s" % e)
            return dataset
        else:
            raise ValueError("Wrong input or method doesn't support this type now")




    def play_action_demo(self, action_path, states_path, control_dt=0.01, demo_dt=0.01, dataset_path=None):
        """

        Plays a demo of the loaded actions by using the actions in action_path.
        action_path: path to the .npz file. Should be in format (number of samples/steps, action dimension)
        states_path: path to states.npz file, for initial position; should be in format like for play_trajectory_demo
        control_dt: model control frequency
        demo_dt: freqency the data was collected
        dataset_path: if set, method creates a dataset with actions, states, episode_starts, next_states, absorbing, rewards
        make sure action clipping is off

        """
        assert demo_dt == control_dt, "needs changes for that"
        # to get the same init position
        trajectory_files = np.load(states_path, allow_pickle=True)
        trajectory = np.array([trajectory_files[key] for key in trajectory_files.keys()])


        #to safe the optimal trajectory for states only in IRL repo -> update all datasets in the same step
        if dataset_path:
            traj_start_offset = 1023  # offset where to start logging the trajectory
            print("Shape optimal states: ", trajectory[:,traj_start_offset + 1:].shape)

            np.savez(os.path.join(dataset_path, 'dataset_only_states_unitreeA1_IRL_optimal.npz'),
                     q_trunk_tx=np.array(trajectory[0][traj_start_offset + 1:]),
                     q_trunk_ty=np.array(trajectory[1][traj_start_offset + 1:]),
                     q_trunk_tz=np.array(trajectory[2][traj_start_offset + 1:]),
                     q_trunk_tilt=np.array(trajectory[3][traj_start_offset + 1:]),
                     q_trunk_list=np.array(trajectory[4][traj_start_offset + 1:]),
                     q_trunk_rotation=np.array(trajectory[5][traj_start_offset + 1:]),
                     q_FR_hip_joint=np.array(trajectory[6][traj_start_offset + 1:]),
                     q_FR_thigh_joint=np.array(trajectory[7][traj_start_offset + 1:]),
                     q_FR_calf_joint=np.array(trajectory[8][traj_start_offset + 1:]),
                     q_FL_hip_joint=np.array(trajectory[9][traj_start_offset + 1:]),
                     q_FL_thigh_joint=np.array(trajectory[10][traj_start_offset + 1:]),
                     q_FL_calf_joint=np.array(trajectory[11][traj_start_offset + 1:]),
                     q_RR_hip_joint=np.array(trajectory[12][traj_start_offset + 1:]),
                     q_RR_thigh_joint=np.array(trajectory[13][traj_start_offset + 1:]),
                     q_RR_calf_joint=np.array(trajectory[14][traj_start_offset + 1:]),
                     q_RL_hip_joint=np.array(trajectory[15][traj_start_offset + 1:]),
                     q_RL_thigh_joint=np.array(trajectory[16][traj_start_offset + 1:]),
                     q_RL_calf_joint=np.array(trajectory[17][traj_start_offset + 1:]),
                     dq_trunk_tx=np.array(trajectory[18][traj_start_offset + 1:]),
                     dq_trunk_tz=np.array(trajectory[19][traj_start_offset + 1:]),
                     dq_trunk_ty=np.array(trajectory[20][traj_start_offset + 1:]),
                     dq_trunk_tilt=np.array(trajectory[21][traj_start_offset + 1:]),
                     dq_trunk_list=np.array(trajectory[22][traj_start_offset + 1:]),
                     dq_trunk_rotation=np.array(trajectory[23][traj_start_offset + 1:]),
                     dq_FR_hip_joint=np.array(trajectory[24][traj_start_offset + 1:]),
                     dq_FR_thigh_joint=np.array(trajectory[25][traj_start_offset + 1:]),
                     dq_FR_calf_joint=np.array(trajectory[26][traj_start_offset + 1:]),
                     dq_FL_hip_joint=np.array(trajectory[27][traj_start_offset + 1:]),
                     dq_FL_thigh_joint=np.array(trajectory[28][traj_start_offset + 1:]),
                     dq_FL_calf_joint=np.array(trajectory[29][traj_start_offset + 1:]),
                     dq_RR_hip_joint=np.array(trajectory[30][traj_start_offset + 1:]),
                     dq_RR_thigh_joint=np.array(trajectory[31][traj_start_offset + 1:]),
                     dq_RR_calf_joint=np.array(trajectory[32][traj_start_offset + 1:]),
                     dq_RL_hip_joint=np.array(trajectory[33][traj_start_offset + 1:]),
                     dq_RL_thigh_joint=np.array(trajectory[34][traj_start_offset + 1:]),
                     dq_RL_calf_joint=np.array(trajectory[35][traj_start_offset + 1:]))

        print("Trajectory shape: ", trajectory.shape)
        # set x and y to 0: be carefull need to be at index 0,1
        trajectory[0, :] -= trajectory[0, 0]
        trajectory[1, :] -= trajectory[1, 0]

        obs_spec = self.obs_helper.observation_spec
        for key_name_ot, value in zip(obs_spec, trajectory[:, 0]):
            key, name, ot = key_name_ot
            if ot == ObservationType.JOINT_POS:
                self._data.joint(name).qpos = value
            elif ot == ObservationType.JOINT_VEL:
                self._data.joint(name).qvel = value

        # np.set_printoptions(threshold=sys.maxsize)
        action_files = np.load(action_path, allow_pickle=True)
        actions = np.array([action_files[key] for key in action_files.keys()])[0]

        # scale frequencies
        if demo_dt != control_dt:
            new_demo_sampling_factor = demo_dt / control_dt
            x = np.arange(actions.shape[0])
            x_new = np.linspace(0, actions.shape[0] - 1, round(actions.shape[0] * new_demo_sampling_factor),
                                endpoint=True)
            actions = interpolate.interp1d(x, actions, kind="cubic", axis=0)(x_new)
            trajectory = interpolate.interp1d(x, trajectory, kind="cubic", axis=1)(x_new)

        true_pos = []
        set_point = []

        if (dataset_path):
            actions_dataset = []
            states_dataset = []
            episode_starts_dataset = [False] * (actions.shape[0] - traj_start_offset - 1)
            episode_starts_dataset[0] = True
            # next_states_dataset=[]
            # absorbing_dataset=[]
            # rewards_dataset=[]

        for i in np.arange(actions.shape[0]):
            # time.sleep(.1)
            if i > 1023:
                self.has_fallen(self._obs)
            if (dataset_path and i > traj_start_offset):
                actions_dataset.append(list(actions[i]))
                states_dataset.append(list(self._data.qpos[:]) + list(self._data.qvel[:]))
                # absorbing_dataset.append(self.is_absorbing(self._obs))
                temp_obs = self._obs

            action = actions[i]
            true_pos.append(list(self._data.qpos[6:]))
            set_point.append(trajectory[6:18, i])
            #action = 10*(trajectory[6:18, i+1]-self._data.qpos[6:])#+1*(-self._data.qvel[6:])
            nstate, _, absorbing, _ = self.step(action) #clipping in xml torque needeed?
            self.render()


            # if(dataset_path and i>traj_start_offset):

            # if nextstate is used for training; wasn't compatible with action in the moment
            # next_states_dataset.append(list(self._data.qpos[:]) + list(self._data.qvel[:]))

            # rewards_dataset.append(self.reward(temp_obs, action, self._obs, self.is_absorbing(self._obs)))

        if (dataset_path):
            # extra dataset with only states for initial positions
            only_states_dataset = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
                                   [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
            for j in np.arange(len(states_dataset)):
                for i in np.arange(len(only_states_dataset)):
                    only_states_dataset[i].append(states_dataset[j][i])
            print("Shape only_states: ", np.array(only_states_dataset).shape)
            np.savez(os.path.join(dataset_path, 'dataset_only_states_unitreeA1_IRL.npz'),
                     q_trunk_tx=np.array(only_states_dataset[0]),
                     q_trunk_ty=np.array(only_states_dataset[1]),
                     q_trunk_tz=np.array(only_states_dataset[2]),
                     q_trunk_tilt=np.array(only_states_dataset[3]),
                     q_trunk_list=np.array(only_states_dataset[4]),
                     q_trunk_rotation=np.array(only_states_dataset[5]),
                     q_FR_hip_joint=np.array(only_states_dataset[6]),
                     q_FR_thigh_joint=np.array(only_states_dataset[7]),
                     q_FR_calf_joint=np.array(only_states_dataset[8]),
                     q_FL_hip_joint=np.array(only_states_dataset[9]),
                     q_FL_thigh_joint=np.array(only_states_dataset[10]),
                     q_FL_calf_joint=np.array(only_states_dataset[11]),
                     q_RR_hip_joint=np.array(only_states_dataset[12]),
                     q_RR_thigh_joint=np.array(only_states_dataset[13]),
                     q_RR_calf_joint=np.array(only_states_dataset[14]),
                     q_RL_hip_joint=np.array(only_states_dataset[15]),
                     q_RL_thigh_joint=np.array(only_states_dataset[16]),
                     q_RL_calf_joint=np.array(only_states_dataset[17]),
                     dq_trunk_tx=np.array(only_states_dataset[18]),
                     dq_trunk_tz=np.array(only_states_dataset[19]),
                     dq_trunk_ty=np.array(only_states_dataset[20]),
                     dq_trunk_tilt=np.array(only_states_dataset[21]),
                     dq_trunk_list=np.array(only_states_dataset[22]),
                     dq_trunk_rotation=np.array(only_states_dataset[23]),
                     dq_FR_hip_joint=np.array(only_states_dataset[24]),
                     dq_FR_thigh_joint=np.array(only_states_dataset[25]),
                     dq_FR_calf_joint=np.array(only_states_dataset[26]),
                     dq_FL_hip_joint=np.array(only_states_dataset[27]),
                     dq_FL_thigh_joint=np.array(only_states_dataset[28]),
                     dq_FL_calf_joint=np.array(only_states_dataset[29]),
                     dq_RR_hip_joint=np.array(only_states_dataset[30]),
                     dq_RR_thigh_joint=np.array(only_states_dataset[31]),
                     dq_RR_calf_joint=np.array(only_states_dataset[32]),
                     dq_RL_hip_joint=np.array(only_states_dataset[33]),
                     dq_RL_thigh_joint=np.array(only_states_dataset[34]),
                     dq_RL_calf_joint=np.array(only_states_dataset[35]))

            # safe dataset with actions, absorbing etc -> used for learning in gail
            print("Shape actions, states: ", np.array(actions_dataset).shape, np.array(states_dataset).shape)
            np.savez(os.path.join(dataset_path, 'dataset_unitreeA1_IRL.npz'),
                     actions=actions_dataset, states=list(states_dataset), episode_starts=episode_starts_dataset)
            # absorbing=absorbing_dataset, rewards=rewards_dataset)# next_states=next_states_dataset,

        # plotting of error and comparison of setpoint and actual position
        true_pos = np.array(true_pos)
        set_point = np.array(set_point)
        # --------------------------------------------------------------------------------------------------------------
        data = {
            "setpoint": set_point[:, 0],
            "actual pos": true_pos[:, 0]
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
            "hip error": set_point[:, 0] - true_pos[:, 0],
            "thigh error": set_point[:, 1] - true_pos[:, 1],
            "calf error": set_point[:, 2] - true_pos[:, 2]
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

# changed force inertia mass, ranges, kp, removed limp



    def create_dataset(self, data_path, ignore_keys=[], normalizer=None, only_state=True, use_next_states=True):
        """
        creates dataset.
        If data_path is set only states has to be false -> creates dataset with states, actions (next_states)
        else dataset with only states is created
        scales/interpolates to the correct frequencies
        dataset needs to be in the same order as self.obs_helper.observation_spec
        """
        if only_state and use_next_states:

            trajectory_files = np.load(data_path, allow_pickle=True)
            trajectory_files = {k: d for k, d in trajectory_files.items()}  # convert to dict to be mutable

            keys = trajectory_files.keys()

            trajectory = np.array([trajectory_files[key] for key in keys])

            demo_dt = self.trajectory.traj_dt
            control_dt = self.trajectory.control_dt


            #interpolation
            if demo_dt != control_dt:
                new_traj_sampling_factor = demo_dt / control_dt

                x = np.arange(trajectory.shape[1])
                x_new = np.linspace(0, trajectory.shape[1] - 1, round(trajectory.shape[1] * new_traj_sampling_factor),
                                    endpoint=True)

                trajectory = interpolate.interp1d(x, trajectory, kind="cubic", axis=1)(x_new)

            # create a dict and extract all elements except the ones specified in ignore_keys.
            all_data = dict(zip(keys, list(trajectory)))
            for ikey in ignore_keys:
                del all_data[ikey]
            traj = list(all_data.values())
            states = np.transpose(np.array(traj))

            # normalize if needed
            if normalizer:
                normalizer.set_state(dict(mean=np.mean(states, axis=0),
                                          var=1 * (np.std(states, axis=0) ** 2),
                                          count=1))
                states = np.array([normalizer(st) for st in states])

            # convert to dict with states and next_states
            new_states = states[:-1]
            new_next_states = states[1:]
            absorbing = np.zeros(len(new_states))

            return dict(states=new_states, next_states=new_next_states, absorbing=absorbing)





        elif not only_state:

            # change name in ignore keys into
            obs_keys = list(np.array(self.obs_helper.observation_spec)[:, 0])
            ignore_index = []
            for key in ignore_keys:
                ignore_index.append(obs_keys.index(key))

            dataset = dict()

            # load expert training data
            expert_files = np.load(data_path)
            dataset["states"] = expert_files["states"]
            dataset["actions"] = expert_files["actions"]

            dataset["episode_starts"] = expert_files["episode_starts"]
            assert dataset["episode_starts"][0] and [x for x in dataset["episode_starts"][1:] if
                                                     x == True] == [], "Implementation only for one long trajectory"

            # remove ignore indices
            for i in sorted(ignore_index, reverse=True):
                dataset["states"] = np.delete(dataset["states"], i, 1)

            # scale frequencies
            demo_dt = self.trajectory.traj_dt
            control_dt = self.trajectory.control_dt
            if demo_dt != control_dt:
                new_demo_sampling_factor = demo_dt / control_dt
                x = np.arange(dataset["states"].shape[0])
                x_new = np.linspace(0, dataset["states"].shape[0] - 1,
                                    round(dataset["states"].shape[0] * new_demo_sampling_factor),
                                    endpoint=True)
                dataset["states"] = interpolate.interp1d(x, dataset["states"], kind="cubic", axis=0)(x_new)
                dataset["actions"] = interpolate.interp1d(x, dataset["actions"], kind="cubic", axis=0)(x_new)
                dataset["episode_starts"] = [False] * x_new
                dataset["episode_starts"][0] = True

            # maybe we have next action and next next state
            try:
                dataset["next_actions"] = expert_files["next_actions"]
                dataset["next_next_states"] = expert_files["next_next_states"]
                # remove ignore indices
                for i in sorted(ignore_index, reverse=True):
                    dataset["next_next_states"] = np.delete(dataset["next_next_states"], i, 1)
                # scaling
                if demo_dt != control_dt:
                    dataset["next_actions"] = interpolate.interp1d(x, dataset["next_actions"], kind="cubic", axis=0)(
                        x_new)
                    dataset["next_next_states"] = interpolate.interp1d(x, dataset["next_next_states"], kind="cubic",
                                                                       axis=0)(x_new)

            except KeyError as e:
                print("Did not find next action or next next state.")

            # maybe we have next states and dones in the dataset
            try:
                dataset["next_states"] = expert_files["next_states"]
                dataset["absorbing"] = expert_files["absorbing"]

                # remove ignore indices
                for i in sorted(ignore_index, reverse=True):
                    dataset["next_states"] = np.delete(dataset["next_states"], i, 1)

                # scaling
                if demo_dt != control_dt:
                    dataset["next_states"] = interpolate.interp1d(x, dataset["next_states"], kind="cubic", axis=0)(
                        x_new)
                    # TODO: not sure about this
                    dataset["absorbing"] = interpolate.interp1d(x, dataset["absorbing"], kind="cubic", axis=0)(x_new)

            except KeyError as e:
                print("Warning Dataset: %s" % e)
            return dataset
        else:
            raise ValueError("Wrong input or method doesn't support this type now")


    def preprocess_expert_data(self, dataset_path, state_type, action_type, states_path, actions_path=None,
                               control_dt=0.01, demo_dt=0.01, use_rendering=False, use_plotting=False):

        assert state_type == "mujoco_data" or state_type == "optimal", "state type not supported"
        assert action_type is None or action_type == "optimal" or action_type == "p-controller", "action type not supported"
        assert control_dt == demo_dt, "Doesn't support scaling yet; shouldn't be needed -> scaling in create_dataset"



        # TODO: nötig mujoco simulation bei action type optimal?
        if (state_type == "mujoco_data" and (action_type is None or action_type == "optimal")):
            assert actions_path is not None

            states_dataset, actions_dataset = self.play_action_demo2(actions_path=actions_path, states_path=states_path,
                                                                      control_dt=control_dt, demo_dt=demo_dt,
                                                                      use_rendering=use_rendering, use_plotting=use_plotting,
                                                                      use_pd_controller=False
                                                                      )
        elif action_type == "p-controller":
            assert actions_path is not None

            states_dataset, actions_dataset = self.play_action_demo2(actions_path=actions_path, states_path=states_path,
                                                                      control_dt=control_dt, demo_dt=demo_dt,
                                                                      use_rendering=use_rendering,
                                                                      use_plotting=use_plotting,
                                                                      use_pd_controller=True
                                                                      )

        if action_type == "optimal":
            # load optimal states from datamodel (for init_position/states dataset)
            trajectory_files = np.load(actions_path, allow_pickle=True)
            opt_actions = np.array([trajectory_files[key] for key in trajectory_files.keys()])
            actions_dataset = opt_actions[0][:-1,:]


        if state_type == "optimal":
            # load optimal states from datamodel (for init_position/states dataset)
            trajectory_files = np.load(states_path, allow_pickle=True)
            opt_states = np.array([trajectory_files[key] for key in trajectory_files.keys()])
            states_dataset = opt_states[:,:-1]


        # Annahme: alle states_dataset im Format 36,51024

        """
        Fälle
        only states:
            eigtl immer mit optimalen states
        mit actions
            states optimal und actions optimal
            states optimal und actions berechnet/pd-controller (---Frage an Firas - muss dafür mujoco simulieren-> geht dabei nicht was kaputt?)
            states mit mujoco erzeugt und actions optimal
            
        
            
        """

        traj_start_offset = 1023  # offset where to start logging the trajectory

        # store the states
        print("Shape states: ", states_dataset[:, traj_start_offset + 1:].shape)
        np.savez(os.path.join(dataset_path, 'dataset_only_states_unitreeA1_IRL.npz'),
                 q_trunk_tx=np.array(states_dataset[0][traj_start_offset + 1:]),
                 q_trunk_ty=np.array(states_dataset[1][traj_start_offset + 1:]),
                 q_trunk_tz=np.array(states_dataset[2][traj_start_offset + 1:]),
                 q_trunk_tilt=np.array(states_dataset[3][traj_start_offset + 1:]),
                 q_trunk_list=np.array(states_dataset[4][traj_start_offset + 1:]),
                 q_trunk_rotation=np.array(states_dataset[5][traj_start_offset + 1:]),
                 q_FR_hip_joint=np.array(states_dataset[6][traj_start_offset + 1:]),
                 q_FR_thigh_joint=np.array(states_dataset[7][traj_start_offset + 1:]),
                 q_FR_calf_joint=np.array(states_dataset[8][traj_start_offset + 1:]),
                 q_FL_hip_joint=np.array(states_dataset[9][traj_start_offset + 1:]),
                 q_FL_thigh_joint=np.array(states_dataset[10][traj_start_offset + 1:]),
                 q_FL_calf_joint=np.array(states_dataset[11][traj_start_offset + 1:]),
                 q_RR_hip_joint=np.array(states_dataset[12][traj_start_offset + 1:]),
                 q_RR_thigh_joint=np.array(states_dataset[13][traj_start_offset + 1:]),
                 q_RR_calf_joint=np.array(states_dataset[14][traj_start_offset + 1:]),
                 q_RL_hip_joint=np.array(states_dataset[15][traj_start_offset + 1:]),
                 q_RL_thigh_joint=np.array(states_dataset[16][traj_start_offset + 1:]),
                 q_RL_calf_joint=np.array(states_dataset[17][traj_start_offset + 1:]),
                 dq_trunk_tx=np.array(states_dataset[18][traj_start_offset + 1:]),
                 dq_trunk_tz=np.array(states_dataset[19][traj_start_offset + 1:]),
                 dq_trunk_ty=np.array(states_dataset[20][traj_start_offset + 1:]),
                 dq_trunk_tilt=np.array(states_dataset[21][traj_start_offset + 1:]),
                 dq_trunk_list=np.array(states_dataset[22][traj_start_offset + 1:]),
                 dq_trunk_rotation=np.array(states_dataset[23][traj_start_offset + 1:]),
                 dq_FR_hip_joint=np.array(states_dataset[24][traj_start_offset + 1:]),
                 dq_FR_thigh_joint=np.array(states_dataset[25][traj_start_offset + 1:]),
                 dq_FR_calf_joint=np.array(states_dataset[26][traj_start_offset + 1:]),
                 dq_FL_hip_joint=np.array(states_dataset[27][traj_start_offset + 1:]),
                 dq_FL_thigh_joint=np.array(states_dataset[28][traj_start_offset + 1:]),
                 dq_FL_calf_joint=np.array(states_dataset[29][traj_start_offset + 1:]),
                 dq_RR_hip_joint=np.array(states_dataset[30][traj_start_offset + 1:]),
                 dq_RR_thigh_joint=np.array(states_dataset[31][traj_start_offset + 1:]),
                 dq_RR_calf_joint=np.array(states_dataset[32][traj_start_offset + 1:]),
                 dq_RL_hip_joint=np.array(states_dataset[33][traj_start_offset + 1:]),
                 dq_RL_thigh_joint=np.array(states_dataset[34][traj_start_offset + 1:]),
                 dq_RL_calf_joint=np.array(states_dataset[35][traj_start_offset + 1:]))




        if action_type is not None:
            action_states_dataset = []#[[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
            for i in range(states_dataset.shape[1]):
                action_states_dataset.append(states_dataset[:, i])
            action_states_dataset = np.array(action_states_dataset)

            print("Shape actions, states: ", actions_dataset[traj_start_offset+1:].shape, ", ", action_states_dataset[traj_start_offset+1:].shape)
            episode_starts_dataset = [False] * actions_dataset[traj_start_offset+1:].shape[0]
            episode_starts_dataset[0]=True
            np.savez(os.path.join(dataset_path, 'dataset_unitreeA1_IRL.npz'),
                     actions=actions_dataset[traj_start_offset+1:], states=action_states_dataset[traj_start_offset+1:], episode_starts=episode_starts_dataset)
        else:
            print("Only states dataset/without actions")







    #states action dataset not in all cases needed (onlystates)


    def play_action_demo2(self, actions_path, states_path, control_dt=0.01, demo_dt=0.01,
                          use_rendering=True, use_plotting=False, use_pd_controller=False):
        """

        Plays a demo of the loaded actions by using the actions in actions_path.
        actions_path: path to the .npz file. Should be in format (number of samples/steps, action dimension)
        states_path: path to states.npz file, for initial position; should be in format like for play_trajectory_demo
        control_dt: model control frequency
        demo_dt: freqency the data was collected
        use_rendering: if the mujoco simulation should be rendered
        use_plotting: if the setpoint and the actual position should be plotted

        """
        assert demo_dt == control_dt, "needs changes for that"
        # to get the same init position
        trajectory_files = np.load(states_path, allow_pickle=True)
        trajectory = np.array([trajectory_files[key] for key in trajectory_files.keys()])

        print("Trajectory shape: ", trajectory.shape)
        # set x and y to 0: be carefull need to be at index 0,1
        trajectory[0, :] -= trajectory[0, 0]
        trajectory[1, :] -= trajectory[1, 0]

        # set initial position
        obs_spec = self.obs_helper.observation_spec
        for key_name_ot, value in zip(obs_spec, trajectory[:, 0]):
            key, name, ot = key_name_ot
            if ot == ObservationType.JOINT_POS:
                self._data.joint(name).qpos = value
            elif ot == ObservationType.JOINT_VEL:
                self._data.joint(name).qvel = value

        # np.set_printoptions(threshold=sys.maxsize)

        # load actions
        action_files = np.load(actions_path, allow_pickle=True)
        actions = np.array([action_files[key] for key in action_files.keys()])[0]

        # TODO: needs changes? -----------------------------------------------------------------------------------------
        # scale frequencies
        if demo_dt != control_dt:
            new_demo_sampling_factor = demo_dt / control_dt
            x = np.arange(actions.shape[0])
            x_new = np.linspace(0, actions.shape[0] - 1, round(actions.shape[0] * new_demo_sampling_factor),
                                endpoint=True)
            actions = interpolate.interp1d(x, actions, kind="cubic", axis=0)(x_new)
            trajectory = interpolate.interp1d(x, trajectory, kind="cubic", axis=1)(x_new)

        true_pos = []
        set_point = []



        actions_dataset = []
        states_dataset = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
        assert len(states_dataset) == len(self.obs_helper.observation_spec)
        # next_states_dataset=[]
        # absorbing_dataset=[]
        # rewards_dataset=[]
        e_old = 0
        for i in np.arange(actions.shape[0]-1):
            #time.sleep(.1)

            # for plotting
            true_pos.append(list(self._data.qpos[6:]))
            set_point.append(trajectory[6:18, i])

            #choose actions of dataset or pd-controller
            if not use_pd_controller:
                action = actions[i]
            else:
                e = trajectory[6:18, i+1]-self._data.qpos[6:]
                de = e-e_old
                kp = 10
                kd = .15
                action = kp*e+(kd/control_dt)*de
                e_old = e

            # store actions and states for datasets
            actions_dataset.append(list(action))
            q_pos_vel = list(self._data.qpos[:]) + list(self._data.qvel[:])
            for i in range(len(states_dataset)):
                states_dataset[i].append(q_pos_vel[i])
            # absorbing_dataset.append(self.is_absorbing(self._obs))
            # temp_obs = self._obs

            nstate, _, absorbing, _ = self.step(action)
            if use_rendering:
                self.render()




            # rewards_dataset.append(self.reward(temp_obs, action, self._obs, self.is_absorbing(self._obs)))

        if use_plotting:
        # plotting of error and comparison of setpoint and actual position
            plot_set_actual_position(true_pos=true_pos, set_point=set_point)

        return np.array(states_dataset), np.array(actions_dataset)


    def plot_set_actual_position(self, true_pos, set_point):
        true_pos = np.array(true_pos)
        set_point = np.array(set_point)
        # --------------------------------------------------------------------------------------------------------------
        data = {
            "setpoint": set_point[:, 0],
            "actual pos": true_pos[:, 0]
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
            "hip error": set_point[:, 0] - true_pos[:, 0],
            "thigh error": set_point[:, 1] - true_pos[:, 1],
            "calf error": set_point[:, 2] - true_pos[:, 2]
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

# changed force inertia mass, ranges, kp, removed limp

