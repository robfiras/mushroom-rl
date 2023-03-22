import time
import warnings
from copy import deepcopy
from time import perf_counter
from contextlib import contextmanager

from mushroom_rl.utils.angles import euler_to_quat, quat_to_euler
from mushroom_rl.environments.mujoco import ObservationType

import matplotlib.pyplot as plt

import numpy as np
from scipy import signal, interpolate

# Todo: do we want to include foot position in future?
# FOOT_KEYS = ["rel_feet_xpos_r", "rel_feet_ypos_r", "rel_feet_zpos_r",
#              "rel_feet_xpos_l", "rel_feet_ypos_l", "rel_feet_zpos_l",
#              "feet_q1_r", "feet_q2_r", "feet_q3_r", "feet_q4_r",
#              "feet_q1_l", "feet_q2_l", "feet_q3_l", "feet_q4_l",
#              "feet_xvelp_r", "feet_yvelp_r", "feet_zvelp_r",
#              "feet_xvelp_l", "feet_yvelp_l", "feet_zvelp_l",
#              "feet_xvelr_r", "feet_yvelr_r", "feet_zvelr_r",
#              "feet_xvelr_l", "feet_yvelr_l", "feet_zvelr_l"]




class Trajectory(object):
    """
    Builds a general trajectory from a numpy bin file(.npy), and automatically
    synchronizes the trajectory timestep to the desired control timestep while
    also allowing to change it's speed by the desired amount. When using
    periodic trajectories it is also possible to pass split points which signal
    the points where the trajectory repeats, and provides an utility to select
    the desired cycle.

    """
    def __init__(self, keys, traj_path, low, high, joint_pos_idx, observation_spec=None,
                 traj_dt=0.002, control_dt=0.01, ignore_keys=[], interpolate_map=None, interpolate_remap=None):
        """
        Constructor.

        Args:
            model: mujoco model.
            data: mujoco data structure.
            keys (list): list of keys to extract data from the trajectories.
            traj_path (string): path with the trajectory for the
                model to follow. Should be a numpy zipped file (.npz)
                with a 'trajectory_data' array and possibly a
                'split_points' array inside. The 'trajectory_data'
                should be in the shape (joints x observations);
            traj_dt (float, 0.01): time step of the trajectory file;
            control_dt (float, 0.01): model control frequency (used to
                synchronize trajectory with the control step);

        """
        self._trajectory_files = np.load(traj_path, allow_pickle=True)
        self._trajectory_files = {k:d for k, d in self._trajectory_files.items()} # convert to dict to be mutable
        self.check_if_trajectory_is_in_range(low, high, keys, joint_pos_idx)

        # add all goal states to keys (goal states have to start with 'goal') #todo tim goals
        # #if not in keys
        keys += [key for key in self._trajectory_files.keys() if key.startswith('goal')]


        # needed for deep mimic
        #if "rel_feet_xpos_r" in self._trajectory_files.keys():
        #    keys += FOOT_KEYS

        # remove unwanted keys
        for ik in ignore_keys:
            keys.remove(ik)

        #splitpoints mark the beginning of the next trajectory. the last split_point points to the index behind the last element of the trajectories -> len(traj)
        if "split_points" in self._trajectory_files.keys():
            self.split_points = self._trajectory_files["split_points"]
        else:
            self.split_points = np.array([0, len(list(self._trajectory_files.values())[0])])


        # 3d matrix: (len state space, no trajectories, no of samples)
        self.trajectory = np.array([[list(self._trajectory_files[key])[self.split_points[i]:self.split_points[i+1]]
                                     for i in range(len(self.split_points)-1)] for key in keys], dtype=object)

        self.keys = keys
        print("Trajectory shape: ", self.trajectory.shape)

        # TODO from Tim: in case I forget that, I needed it nowhere - do you?
        #self.n_repeating_steps = len(self.split_points) - 1

        self.traj_dt = traj_dt
        self.control_dt = control_dt
        self.traj_speed_multiplier = 1.0    # todo: delete the trajecotry speed multiplier stuff

        # interpolation of the trajectories
        if self.traj_dt != control_dt:
            self.trajectory = self._interpolate_trajectory(
                self.trajectory, map_funct=interpolate_map, re_map_funct=interpolate_remap
            )
            # interpolation of split_points
            self.split_points=[0]
            for k in range(len(self.trajectory[0])):
                self.split_points.append(self.split_points[-1]+len(self.trajectory[0][k]))

        self.subtraj_step_no = 0
        self.traj_no = 0
        self.x_dist = 0
        self.subtraj = self.trajectory[:,0].copy()

    @property
    def traj_length(self):
        """
        returns array of the length for each trajectory
        """
        return [len(self.trajectory[0][i]) for i in range(len(self.trajectory[0]))]

    #TODO from Tim: not adapted to multi dim obs_spec and multiple trajs/implement my own function in unitree
    def create_dataset(self, ignore_keys=[], normalizer=None):

        # create a dict and extract all elements except the ones specified in ignore_keys.
        all_data = dict(zip(self.keys, deepcopy(list(self.trajectory))))
        for ikey in ignore_keys:
            del all_data[ikey]
        traj = list(all_data.values())
        states = np.transpose(deepcopy(np.array(traj)))

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

    #TODO from Tim: not adapted to multi dim obs_spec and multiple trajs/implement my own function in unitree
    def create_datase_with_triplet_states(self, normalizer=None):

        # get relevant data
        states = np.transpose(deepcopy(self.trajectory))

        # normalize if needed
        if normalizer:
            normalizer.set_state(dict(mean=np.mean(states, axis=0),
                                      var=1 * (np.std(states, axis=0) ** 2),
                                      count=1))
            norm_states = np.array([normalizer(st) for st in states])

        # convert to dict with states and next_states
        states = norm_states[:-2]
        next_states = norm_states[1:-1]
        next_next_states = norm_states[2:]

        return dict(states=states, next_states=next_states, next_next_states=next_next_states)

    def _interpolate_trajectory(self, trajs, map_funct=None, re_map_funct=None):
        """
        interpolates each trajectory
        Args:
            trajs: multiple trajectories, shape: (len state space, no trajectories, no of samples)
            map_funct: how to preprocess the trajectory before interpolation
            re_map_funct: how to postprocess the trajectory after interpolation
        """
        assert (map_funct is None) == (re_map_funct is None)
        new_trajs = [list() for i in range(len(trajs))]
        # interpolate over each trajectory
        for j in range(len(trajs[0])):
            # the trajectory we interpolate
            traj = trajs[:,j]
            x = np.arange(traj.shape[1])
            new_traj_sampling_factor = (1 / self.traj_speed_multiplier) * (
                    self.traj_dt / self.control_dt)
            x_new = np.linspace(0, traj.shape[1] - 1, round(traj.shape[1] * new_traj_sampling_factor),
                                endpoint=True)

            # proprocess trajectory
            if map_funct is not None:
                traj = map_funct(traj)

            new_traj = interpolate.interp1d(x, traj, kind="cubic", axis=1)(x_new)

            # postprocess trajectory
            if re_map_funct is not None:
                new_traj = re_map_funct(new_traj)

            # append results
            for k in range(len(trajs)):
                new_trajs[k].append(new_traj[k])
        return np.array(new_trajs)


    def get_next_sub_trajectory(self):
        """
        Get the next trajectory once the current one reaches it's end.

        """
        self.x_dist += self.subtraj[0][-1]
        self.reset_trajectory()

    #TODO from Tim: not adapted to multi dim obs_spec and multiple trajs
    def _get_traj_gait_sub_steps(self, initial_walking_step,
                                 number_of_walking_steps=1):
        start_sim_step = self.split_points[initial_walking_step]
        end_sim_step = self.split_points[
            initial_walking_step + number_of_walking_steps
        ]

        sub_traj = self.trajectory[:, start_sim_step:end_sim_step].copy()
        initial_x_pos = self.trajectory[0][start_sim_step]
        sub_traj[0, :] -= initial_x_pos
        return sub_traj

    def reset_trajectory(self, substep_no=None, traj_no=None):
        """
        Resets the trajectory and the model. The trajectory can be forced
        to start on the 'substep_no' if desired, else it starts at
        a random one.

        Args:
            substep_no (int, None): starting point of the trajectory.
                If None, the trajectory starts from a random point.
            traj_no (int, None): number of the trajectory to start from.
                If None, it starts from a random trajectory
        """
        self.x_dist = 0
        if substep_no is None:
            self.traj_no = int(np.random.rand() * len(self.trajectory[0]))
            self.subtraj_step_no = int(np.random.rand() * (
                    self.traj_length[self.traj_no] * 0.45))
        else:
            self.traj_no = traj_no
            self.subtraj_step_no = substep_no
        self.subtraj = self.trajectory[:,self.traj_no].copy()

        # reset x and y to middle position
        self.subtraj[0] -= self.subtraj[0, self.subtraj_step_no]
        self.subtraj[1] -= self.subtraj[1, self.subtraj_step_no]

        return np.array([self.subtraj[i][self.subtraj_step_no] for i in range(len(self.subtraj))], dtype=object)

    def get_next_sample(self):
        if self.subtraj_step_no >= self.traj_length[self.traj_no]:
            self.get_next_sub_trajectory()
        sample = deepcopy([self.subtraj[i][self.subtraj_step_no] for i in range(len(self.subtraj))])
        self.subtraj_step_no += 1
        return np.array(sample, dtype=object)

    def check_if_trajectory_is_in_range(self, low, high, keys, j_idx):

        # get q_pos indices
        j_idx = j_idx[2:]   # exclude x and y

        # check if they are in range
        for i, item in enumerate(self._trajectory_files.items()):
            k, d = item
            if i in j_idx:
                high_i = high[i-2]
                low_i = low[i-2]
                if np.max(d) > high_i:
                    warnings.warn("Trajectory violates joint range in %s. Maximum in trajecotry is %f "
                                  "and maximum range is %f. Clipping the trajecotry into range!"
                                  % (keys[i], np.max(d), high_i), RuntimeWarning)
                elif np.min(d) < low_i:
                    warnings.warn("Trajectory violates joint range in %s. Minimum in trajecotry is %f "
                                  "and minimum range is %f. Clipping the trajecotry into range!"
                                  % (keys[i], np.min(d), low_i), RuntimeWarning)

                # clip trajectory to min & max
                self._trajectory_files[k] = np.clip(self._trajectory_files[k], low_i, high_i)