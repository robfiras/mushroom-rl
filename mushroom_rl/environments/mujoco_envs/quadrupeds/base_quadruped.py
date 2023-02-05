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
        if self.use_2d_ctrl:
            #TODO: from view of robot angle correct like this for traj. replay?; where else adapt this: sim_post_step; sim_post_step and setup only for one constant angle; modify obs also for trunk tilt change?; noise be e-1 laufbar aber driftet ab; numpy 1.20 possible?
            #TODO correct?
            trunk_tilt = self._data.joint("trunk_tilt").qpos[0]

            R = np.array(
                [[np.cos(trunk_tilt), -np.sin(trunk_tilt), 0], [np.sin(trunk_tilt), np.cos(trunk_tilt), 0], [0, 0, 1]])


            self._data.site("dir_arrow").xmat = np.dot(R, self._direction_xmat.reshape((3,3))).reshape((9,))
            self._data.site("dir_arrow_ball").xpos = self._data.body("dir_arrow").xpos + [-0.1 * np.cos(self._direction_angle+trunk_tilt), -0.1 * np.sin(self._direction_angle+trunk_tilt), 0]
        # self._data.qfrc_applied[self._action_indices] = self._data.qfrc_bias[self._action_indices] + self._data.qfrc_applied[self._action_indices]

        # print(self._data.qfrc_bias[:12])



    def traj_post_step(self): # for play_trajectory_demo
        if self.use_2d_ctrl:
            trunk_tilt = self._data.joint("trunk_tilt").qpos[0]
            R = np.array(
                [[np.cos(trunk_tilt), -np.sin(trunk_tilt), 0], [np.sin(trunk_tilt), np.cos(trunk_tilt), 0], [0, 0, 1]])

            self._data.site("dir_arrow").xmat = np.dot(R, self._direction_xmat.reshape((3,3))).reshape((9,))
            self._data.site("dir_arrow_ball").xpos = self._direction_xpos + [
                -0.1 * np.cos(self._direction_angle+trunk_tilt), -0.1 * np.sin(self._direction_angle+trunk_tilt), 0]




    def create_dataset(self, data_path, ignore_keys=[], normalizer=None, only_state=True, use_next_states=True, interpolate_map=None, interpolate_remap=None):
        """
        creates dataset.
        If data_path is set only states has to be false -> creates dataset with states, actions (next_states)
        else dataset with only states is created
        scales/interpolates to the correct frequencies
        dataset needs to be in the same order as self.obs_helper.observation_spec
        """

        assert interpolate_map is not None, "needed for numpy unwrap etc"
        if only_state and use_next_states:

            trajectory_files = np.load(data_path, allow_pickle=True)
            trajectory_files = {k: d for k, d in trajectory_files.items()}  # convert to dict to be mutable

            keys = list(trajectory_files.keys())

            if "split_points" in trajectory_files.keys():
                split_points = trajectory_files["split_points"]
                keys.remove("split_points")
            else:
                split_points = np.array([0, len(list(trajectory_files.values())[0])])

            trajectories = np.array([[list(trajectory_files[key])[
                                    split_points[i]:split_points[i + 1]] for i in
                                    range(len(split_points) - 1)] for key in keys], dtype=object)




            demo_dt = self.trajectory.traj_dt
            control_dt = self.trajectory.control_dt


            #interpolation
            if demo_dt != control_dt:
                new_traj_sampling_factor = demo_dt / control_dt

                trajectories = Trajectory._interpolate_trajectory(
                    trajectories, factor=new_traj_sampling_factor,
                    map_funct=interpolate_map, re_map_funct=interpolate_remap, axis=1
                )

            #check for has_fallen violations
            try:
                # TODO
                for i in range(len(trajectories[0])):
                    transposed = np.transpose(trajectories[2:, i])
                    has_fallen_violation = next(x for x in transposed if self.has_fallen(x))
                    np.set_printoptions(threshold=sys.maxsize)
                    raise RuntimeError("has_fallen violation occured: ", has_fallen_violation)
            except StopIteration:
                print("No has_fallen violation found")
                # opt_states[:,:-1]
            print("   Traj minimal height:", min([min(trajectories[2][i]) for i in range(len(trajectories[0]))]))
            print("   Traj max x-rotation:",
                  max([max(trajectories[4][i], key=abs) for i in range(len(trajectories[0]))], key=abs))
            print("   Traj max y-rotation:",
                  max([max(trajectories[5][i], key=abs) for i in range(len(trajectories[0]))], key=abs))


            # remove ignoreindex here (after has_fallen but before modify observation)
            for ikey in ignore_keys:
                trajectories = np.delete(trajectories, keys.index(ikey), 0)
                keys.remove(ikey)


            #turns rotation matrix into angle
            if self.use_2d_ctrl:  # TODO will need changes if changing modify_obs

                dim1 = len(self._modify_observation(np.hstack(trajectories[:,0,0]))) #dim1 of trajectories after applied modify_obs
                #apply modify obs on each state to be sure to have the same observation as the robot for learning
                new_trajectories = np.empty((dim1, trajectories.shape[1], trajectories.shape[2]))
                for i in range(len(trajectories[0])):
                    transposed_traj = np.transpose(trajectories[:,i])
                    for j in range(len(transposed_traj)):
                        flat_sample = np.hstack(transposed_traj[j])
                        new_trajectories[:, i, j] = self._modify_observation(flat_sample)
                trajectories = new_trajectories


                # for mat in traj[36].reshape((len(traj[0]), 9)):
                #    arrow = np.dot(mat.reshape((3, 3)), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).reshape((9,))
                #   temp.append(np.arctan2(arrow[3], arrow[0]))
                # traj_list[36] = temp



            new_states = []
            new_next_states = []
            # for each trajectory in trajectories append to the result vars
            for i in range(len(trajectories[0])):
                trajectory = trajectories[:,i]
                states = np.transpose(trajectory)

                # normalize if needed
                if normalizer:
                    normalizer.set_state(dict(mean=np.mean(states, axis=0),
                                              var=1 * (np.std(states, axis=0) ** 2),
                                              count=1))
                    states = np.array([normalizer(st) for st in states])

                # convert to dict with states and next_states
                new_states += list(states[:-1])
                new_next_states += list(states[1:])

            absorbing = np.zeros(len(new_states))

            return dict(states=np.array(new_states), next_states=np.array(new_next_states), absorbing=absorbing)





        elif not only_state:
            raise NotImplementedError("needs adapts for multiple traj per paths and interpolatemap/remap and 2d control")

            try:
                # TODO
                for i in range(len(states_dataset[0])):
                    traj = states_dataset[2:, i]
                    transposed2 = np.transpose(traj)
                    has_fallen_violation = next(x for x in transposed if self.has_fallen(x))
                    np.set_printoptions(threshold=sys.maxsize)
                    raise RuntimeError("has_fallen violation occured: ", has_fallen_violation)
            except StopIteration:
                print("No has_fallen violation found")
                # opt_states[:,:-1]

            print(dataset_name, " minimal height:", [min(states_dataset[2][i]) for i in range(len(states_dataset[0]))])
            print(dataset_name, " max x-rotation:",
                  [max(states_dataset[4][i], key=abs) for i in range(len(states_dataset[0]))])
            print(dataset_name, " max y-rotation:",
                  [max(states_dataset[5][i], key=abs) for i in range(len(states_dataset[0]))])





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

            # tranform rot mat into rot angle
            if self.use_2d_ctrl:
                traj_list = [list() for j in range(len(dataset["states"]))]

                for i in range(len(traj_list)):
                    traj_list[i] = list(dataset["states"][i])
                traj_list[36] = [
                    np.arctan2(
                        np.dot(mat.reshape((3, 3)), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).reshape((9,))[3],
                        np.dot(mat.reshape((3, 3)), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).reshape((9,))[0])
                    for mat in dataset["states"][36].reshape((len(dataset["states"][0]), 9))]
                # for mat in traj[36].reshape((len(traj[0]), 9)):
                #    arrow = np.dot(mat.reshape((3, 3)), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).reshape((9,))
                #   temp.append(np.arctan2(arrow[3], arrow[0]))
                # traj_list[36] = temp
                dataset["states"] = np.array(traj_list)

            # scale frequencies
            demo_dt = self.trajectory.traj_dt
            control_dt = self.trajectory.control_dt
            if demo_dt != control_dt:
                new_demo_sampling_factor = demo_dt / control_dt
                x = np.arange(dataset["actions"].shape[0])
                x_new = np.linspace(0, dataset["actions"].shape[0] - 1,
                                    round(dataset["actions"].shape[0] * new_demo_sampling_factor),
                                    endpoint=True)
                dataset["states"] = interpolate.interp1d(x, dataset["states"], kind="cubic", axis=0)(x_new)
                dataset["actions"] = interpolate.interp1d(x, dataset["actions"], kind="cubic", axis=0)(x_new)
                dataset["episode_starts"] = [False] * x_new
                dataset["episode_starts"][0] = True
                dataset["states"] = self._interpolate_trajectory(
                    dataset["states"], factor=new_demo_sampling_factor,
                    map_funct=interpolate_map, re_map_funct=interpolate_remap, axis=0
                )

            # maybe we have next action and next next state
            try:
                dataset["next_actions"] = expert_files["next_actions"]
                dataset["next_next_states"] = expert_files["next_next_states"]
                # remove ignore indices
                for i in sorted(ignore_index, reverse=True):
                    dataset["next_next_states"] = np.delete(dataset["next_next_states"], i, 1)

                # tranform rot mat into rot angle
                if self.use_2d_ctrl:
                    traj_list = [list() for j in range(len(dataset["next_next_states"]))]

                    for i in range(len(traj_list)):
                        traj_list[i] = list(dataset["next_next_states"][i])
                    traj_list[36] = [
                        np.arctan2(
                            np.dot(mat.reshape((3, 3)), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).reshape((9,))[3],
                            np.dot(mat.reshape((3, 3)), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).reshape((9,))[0])
                        for mat in dataset["states"][36].reshape((len(dataset["next_next_states"][0]), 9))]
                    dataset["next_next_states"] = np.array(traj_list)
                # scaling
                if demo_dt != control_dt:
                    dataset["next_actions"] = interpolate.interp1d(x, dataset["next_actions"], kind="cubic", axis=0)(
                        x_new)
                    dataset["next_next_states"] = self._interpolate_trajectory(
                    dataset["next_next_states"], factor=new_demo_sampling_factor,
                    map_funct=interpolate_map, re_map_funct=interpolate_remap, axis=0
                )

            except KeyError as e:
                print("Did not find next action or next next state.")

            # maybe we have next states and dones in the dataset
            try:
                dataset["next_states"] = expert_files["next_states"]
                dataset["absorbing"] = expert_files["absorbing"]

                # remove ignore indices
                for i in sorted(ignore_index, reverse=True):
                    dataset["next_states"] = np.delete(dataset["next_states"], i, 1)

                # tranform rot mat into rot angle
                if self.use_2d_ctrl:
                    traj_list = [list() for j in range(len(dataset["next_states"]))]

                    for i in range(len(traj_list)):
                        traj_list[i] = list(dataset["next_states"][i])
                    traj_list[36] = [
                        np.arctan2(
                            np.dot(mat.reshape((3, 3)), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).reshape((9,))[3],
                            np.dot(mat.reshape((3, 3)), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).reshape((9,))[0])
                        for mat in dataset["states"][36].reshape((len(dataset["next_states"][0]), 9))]
                    dataset["next_states"] = np.array(traj_list)

                # scaling
                if demo_dt != control_dt:
                    dataset["next_states"] = self._interpolate_trajectory(
                        dataset["next_states"], factor=new_demo_sampling_factor,
                        map_funct=interpolate_map, re_map_funct=interpolate_remap, axis=0
                    )
                    # TODO: not sure about this
                    dataset["absorbing"] = interpolate.interp1d(x, dataset["absorbing"], kind="cubic", axis=0)(x_new)

            except KeyError as e:
                print("Warning Dataset: %s" % e)
            return dataset
        else:
            raise ValueError("Wrong input or method doesn't support this type now")


    def preprocess_expert_data(self, dataset_path, state_type, action_type, states_path, dataset_name='', actions_path=None,
                               control_dt=0.01, demo_dt=0.01, use_rendering=False, use_plotting=False, interpolate_map=None,
                               interpolate_remap=None):

        raise NotImplementedError(" needs adaptions for goal_speed (state[37])")
        assert state_type == "mujoco_data" or state_type == "optimal", "state type not supported"
        assert action_type is None or action_type == "optimal" or action_type == "p-controller", "action type not supported"
        assert control_dt == demo_dt, "Doesn't support scaling yet; shouldn't be needed -> scaling in create_dataset"

        if not os.path.exists((dataset_path)):
            os.makedirs(dataset_path)
            print('Created Directory ', dataset_path)

        appendix_only_states = state_type
        appendix_states_actions = ''


        if(action_type == 'optimal'):
            appendix_states_actions = state_type[:3]+'_opt'
        elif(action_type == 'p-controller'):
            appendix_states_actions = state_type[:3] + '_pd'


        # load states for split_points
        trajectory_files = np.load(states_path, allow_pickle=True)
        trajectory_files = {k:d for k, d in trajectory_files.items()} # convert to dict to be mutable
        keys = list(trajectory_files.keys())
        if "split_points" in trajectory_files.keys():
            split_points = trajectory_files["split_points"]
            keys.remove("split_points")
        else:
            split_points = np.array([0, len(list(trajectory_files.values())[0])])


        opt_states = np.array([list(trajectory_files[key]) for key in keys], dtype=object)

        #opt_states = np.array([[list(trajectory_files[key][j]) for j in range(len(trajectory_files[key]))] for key in keys], dtype=object)



        states_dataset = [list() for i in range(len(opt_states))]
        actions_dataset = []

        # TODO: nötig mujoco simulation bei action type optimal?
        if (state_type == "mujoco_data" or action_type == "p-controller"):
            assert actions_path is not None
            for k in range(len(split_points)-1):
                states_dataset_temp, actions_dataset_temp = self.play_action_demo(actions_path=actions_path, states_path=states_path,
                                                                          control_dt=control_dt, demo_dt=demo_dt,
                                                                          use_rendering=use_rendering, use_plotting=use_plotting,
                                                                          use_pd_controller=action_type == "p-controller", interpolate_map=interpolate_map,
                                                                        interpolate_remap=interpolate_remap, traj_no=k
                                                                          )
                for j in range(len(states_dataset_temp)):
                    states_dataset[j] += list(states_dataset_temp[j])
                actions_dataset += list(actions_dataset_temp)
            states_dataset = np.array(states_dataset, dtype=object)
            actions_dataset = np.array(actions_dataset)






        if state_type == "optimal":
            # load optimal states from datamodel (for init_position/states dataset)
            #remove last entry for each trajectory (because p-controller cant use last element)
            states_dataset = opt_states
            for point in sorted(split_points[1:], reverse=True):
                states_dataset = np.delete(states_dataset, point-1, 1)



        if action_type == "optimal":
            # load optimal states from datamodel (for init_position/states dataset)
            trajectory_files = np.load(actions_path, allow_pickle=True)

            actions_dataset = np.array([trajectory_files[key] for key in trajectory_files.keys()])[0]
            # delete last of each traj
            for point in sorted(split_points[1:], reverse=True):
                actions_dataset = np.delete(actions_dataset, point-1, 0)


        # check if states dataset has any fallen states
        try:

            traj = states_dataset[2:]
            transposed = np.transpose(traj)
            has_fallen_violation = next(x for x in transposed if self.has_fallen(x))
            np.set_printoptions(threshold=sys.maxsize)
            raise RuntimeError("has_fallen violation occured: ", has_fallen_violation)
        except StopIteration:
            print("No has_fallen violation found")
            # opt_states[:,:-1]


        print(dataset_name, " minimal height:", min(states_dataset[2]))
        print(dataset_name, " max x-rotation:", max(states_dataset[4], key=abs))
        print(dataset_name, " max y-rotation:", max(states_dataset[5], key=abs))




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
        for i in range(len(split_points)):
            split_points[i] -=i
        #remove offset from every traj:
        for point in sorted(split_points[:-1], reverse=True):
            states_dataset = np.delete(states_dataset, slice(point, point+traj_start_offset), 1)
            if action_type is not None:
                actions_dataset = np.delete(actions_dataset, slice(point, point+traj_start_offset), 0)
        for i in range(len(split_points)):
            split_points[i] -=i*traj_start_offset


        # store the states
        if not self.use_2d_ctrl:
            #print("Shape states: ", states_dataset[:, traj_start_offset + 1:].shape)
            np.savez(os.path.join(dataset_path, 'dataset_only_states_unitreeA1_IRL'+dataset_name+'_'+appendix_only_states+'.npz'),
                     q_trunk_tx=np.array(states_dataset[0]),
                     q_trunk_ty=np.array(states_dataset[1]),
                     q_trunk_tz=np.array(states_dataset[2]),
                     q_trunk_tilt=np.array(states_dataset[3]),
                     q_trunk_list=np.array(states_dataset[4]),
                     q_trunk_rotation=np.array(states_dataset[5]),
                     q_FR_hip_joint=np.array(states_dataset[6]),
                     q_FR_thigh_joint=np.array(states_dataset[7]),
                     q_FR_calf_joint=np.array(states_dataset[8]),
                     q_FL_hip_joint=np.array(states_dataset[9]),
                     q_FL_thigh_joint=np.array(states_dataset[10]),
                     q_FL_calf_joint=np.array(states_dataset[11]),
                     q_RR_hip_joint=np.array(states_dataset[12]),
                     q_RR_thigh_joint=np.array(states_dataset[13]),
                     q_RR_calf_joint=np.array(states_dataset[14]),
                     q_RL_hip_joint=np.array(states_dataset[15]),
                     q_RL_thigh_joint=np.array(states_dataset[16]),
                     q_RL_calf_joint=np.array(states_dataset[17]),
                     dq_trunk_tx=np.array(states_dataset[18]),
                     dq_trunk_tz=np.array(states_dataset[19]),
                     dq_trunk_ty=np.array(states_dataset[20]),
                     dq_trunk_tilt=np.array(states_dataset[21]),
                     dq_trunk_list=np.array(states_dataset[22]),
                     dq_trunk_rotation=np.array(states_dataset[23]),
                     dq_FR_hip_joint=np.array(states_dataset[24]),
                     dq_FR_thigh_joint=np.array(states_dataset[25]),
                     dq_FR_calf_joint=np.array(states_dataset[26]),
                     dq_FL_hip_joint=np.array(states_dataset[27]),
                     dq_FL_thigh_joint=np.array(states_dataset[28]),
                     dq_FL_calf_joint=np.array(states_dataset[29]),
                     dq_RR_hip_joint=np.array(states_dataset[30]),
                     dq_RR_thigh_joint=np.array(states_dataset[31]),
                     dq_RR_calf_joint=np.array(states_dataset[32]),
                     dq_RL_hip_joint=np.array(states_dataset[33]),
                     dq_RL_thigh_joint=np.array(states_dataset[34]),
                     dq_RL_calf_joint=np.array(states_dataset[35]))
        else:
            np.savez(os.path.join(dataset_path,
                                  'dataset_only_states_unitreeA1_IRL' + dataset_name + '_' + appendix_only_states + '.npz'),
                     q_trunk_tx=np.array(states_dataset[0]),
                     q_trunk_ty=np.array(states_dataset[1]),
                     q_trunk_tz=np.array(states_dataset[2]),
                     q_trunk_tilt=np.array(states_dataset[3]),
                     q_trunk_list=np.array(states_dataset[4]),
                     q_trunk_rotation=np.array(states_dataset[5]),
                     q_FR_hip_joint=np.array(states_dataset[6]),
                     q_FR_thigh_joint=np.array(states_dataset[7]),
                     q_FR_calf_joint=np.array(states_dataset[8]),
                     q_FL_hip_joint=np.array(states_dataset[9]),
                     q_FL_thigh_joint=np.array(states_dataset[10]),
                     q_FL_calf_joint=np.array(states_dataset[11]),
                     q_RR_hip_joint=np.array(states_dataset[12]),
                     q_RR_thigh_joint=np.array(states_dataset[13]),
                     q_RR_calf_joint=np.array(states_dataset[14]),
                     q_RL_hip_joint=np.array(states_dataset[15]),
                     q_RL_thigh_joint=np.array(states_dataset[16]),
                     q_RL_calf_joint=np.array(states_dataset[17]),
                     dq_trunk_tx=np.array(states_dataset[18]),
                     dq_trunk_tz=np.array(states_dataset[19]),
                     dq_trunk_ty=np.array(states_dataset[20]),
                     dq_trunk_tilt=np.array(states_dataset[21]),
                     dq_trunk_list=np.array(states_dataset[22]),
                     dq_trunk_rotation=np.array(states_dataset[23]),
                     dq_FR_hip_joint=np.array(states_dataset[24]),
                     dq_FR_thigh_joint=np.array(states_dataset[25]),
                     dq_FR_calf_joint=np.array(states_dataset[26]),
                     dq_FL_hip_joint=np.array(states_dataset[27]),
                     dq_FL_thigh_joint=np.array(states_dataset[28]),
                     dq_FL_calf_joint=np.array(states_dataset[29]),
                     dq_RR_hip_joint=np.array(states_dataset[30]),
                     dq_RR_thigh_joint=np.array(states_dataset[31]),
                     dq_RR_calf_joint=np.array(states_dataset[32]),
                     dq_RL_hip_joint=np.array(states_dataset[33]),
                     dq_RL_thigh_joint=np.array(states_dataset[34]),
                     dq_RL_calf_joint=np.array(states_dataset[35]),
                     dir_arrow=np.array(states_dataset[36]),
                     split_points=split_points)



        if action_type is not None:
            action_states_dataset = []
            for i in range(states_dataset.shape[1]):
                action_states_dataset.append(states_dataset[:, i])
            action_states_dataset = np.array(action_states_dataset)

            print("Shape actions, states: ", actions_dataset.shape, ", ", action_states_dataset.shape)
            episode_starts_dataset = [False] * actions_dataset.shape[0]
            episode_starts_dataset[0]=True
            np.savez(os.path.join(dataset_path, 'dataset_unitreeA1_IRL'+dataset_name+'_'+appendix_states_actions+'.npz'),
                     actions=actions_dataset, states=[action_states_dataset] , episode_starts=episode_starts_dataset, split_points=split_points) #action_states_dataset[traj_start_offset+1:]
        else:
            print("Only states dataset/without actions")







    #states action dataset not in all cases needed (onlystates)










    def play_action_demo(self, actions_path, states_path, control_dt=0.01, demo_dt=0.01, traj_no=0,
                          use_rendering=True, use_plotting=False, use_pd_controller=False, interpolate_map=None, interpolate_remap=None):
        """

        Plays a demo of the loaded actions by using the actions in actions_path.
        actions_path: path to the .npz file. Should be in format (number of samples/steps, action dimension)
        states_path: path to states.npz file, for initial position; should be in format like for play_trajectory_demo
        control_dt: model control frequency
        demo_dt: freqency the data was collected
        traj_no: number of trajectory in path to use
        use_rendering: if the mujoco simulation should be rendered
        use_plotting: if the setpoint and the actual position should be plotted

        """
        #assert demo_dt == control_dt, "needs changes for that"
        # to get the same init position
        trajectory_files = np.load(states_path, allow_pickle=True)
        trajectory_files = {k:d for k, d in trajectory_files.items()} # convert to dict to be mutable


        keys = list(trajectory_files.keys())

        if "split_points" in trajectory_files.keys():
            split_points = trajectory_files["split_points"]
            keys.remove("split_points")
        else:
            split_points = np.array([0, len(list(trajectory_files.values())[0])])


        trajectory = np.array([[list(trajectory_files[key])[split_points[traj_no]:split_points[traj_no+1]]] for key in keys], dtype=object)








        # np.set_printoptions(threshold=sys.maxsize)

        # load actions
        action_files = np.load(actions_path, allow_pickle=True)
        actions = np.array([list(action_files[key])[split_points[traj_no]:split_points[traj_no+1]] for key in action_files.keys()], dtype=object)[0]

        # TODO: needs changes? -----------------------------------------------------------------------------------------
        # scale frequencies
        if demo_dt != control_dt:
            new_demo_sampling_factor = demo_dt / control_dt
            x = np.arange(actions.shape[0])
            x_new = np.linspace(0, actions.shape[0] - 1, round(actions.shape[0] * new_demo_sampling_factor),
                                endpoint=True)
            actions = interpolate.interp1d(x, actions, kind="cubic", axis=0)(x_new)
            trajectory = Trajectory._interpolate_trajectory(
                trajectory, factor=new_demo_sampling_factor,
                map_funct=interpolate_map, re_map_funct=interpolate_remap, axis=1
            )[:,0]
        else:
            trajectory = trajectory[:,0] # remove inner list needed for interpolation

        true_pos = []
        set_point = []



        # set x and y to 0: be carefull need to be at index 0,1
        trajectory[0, :] -= trajectory[0, 0]
        trajectory[1, :] -= trajectory[1, 0]

        # set initial position
        self.set_qpos_qvel(trajectory[:,0])


        actions_dataset = []
        states_dataset = [list() for j in range(len(self.obs_helper.observation_spec))]
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
                self._data.qpos = trajectory[:18, i]
                self._data.qvel = trajectory[18:36, i]
                e = trajectory[6:18, i+1]-self._data.qpos[6:]
                de = e-e_old
                # TODO wenn jedes mal zurück setzten kann auch ohne mujoco actions berechnen
                """
                kp = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
                kd = np.array([1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2])
                """
                #maybe try pos with actions but with optimal states
                kp = 10 #np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
                hip = 0.2 #0.62
                rest = .4 #1.24
                kd = np.array([hip, rest, rest, hip, rest, rest, hip, rest, rest, hip, rest, rest])
                action = kp*e+(kd/control_dt)*de
                e_old = e

            # store actions and states for datasets
            actions_dataset.append(list(action))
            q_pos_vel = list(self._data.qpos[:]) + list(self._data.qvel[:])
            if self.use_2d_ctrl:
                q_pos_vel.append(list(trajectory[36,i]))#
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
            self.plot_set_actual_position(true_pos=true_pos, set_point=set_point)

        return np.array(states_dataset, dtype=object), np.array(actions_dataset)


    def plot_set_actual_position(self, true_pos, set_point):
        true_pos = np.array(true_pos)
        set_point = np.array(set_point, dtype=object)
        # --------------------------------------------------------------------------------------------------------------
        data = {
            "setpoint": set_point[:, 6],
            "actual pos": true_pos[:, 6]
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
            "setpoint": set_point[:, 7],
            "actual pos": true_pos[:, 7]
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
            "setpoint": set_point[:, 8],
            "actual pos": true_pos[:, 8]
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
            "hip error": set_point[:, 6] - true_pos[:, 6],
            "thigh error": set_point[:, 7] - true_pos[:, 7],
            "calf error": set_point[:, 8] - true_pos[:, 8]
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
