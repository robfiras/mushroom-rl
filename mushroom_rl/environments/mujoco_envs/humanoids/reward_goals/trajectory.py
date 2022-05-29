import time
from copy import deepcopy
from time import perf_counter
from contextlib import contextmanager

from mushroom_rl.utils.angles import euler_to_quat, quat_to_euler
from mushroom_rl.environments.mujoco import ObservationType

import matplotlib.pyplot as plt
import mujoco_py
import numpy as np
from scipy import signal, interpolate

# KEYS
PELVIS_POS_KEYS = ["pelvis_tz", "pelvis_tx",  "pelvis_ty"]
PELVIS_EULER_KEYS = ["pelvis_tilt", "pelvis_list", "pelvis_rotation",]
PELVIS_QUAT_KEYS = ["pelvis_q1", "pelvis_q2", "pelvis_q3", "pelvis_q4"]
JOINT_KEYS = ["hip_adduction_r", "hip_flexion_r", "hip_rotation_r", "knee_angle_r",
              "ankle_angle_r", "hip_adduction_l", "hip_flexion_l", "hip_rotation_l", "knee_angle_l", "ankle_angle_l"]

# KEYS of euler dataset
EKEYS = PELVIS_POS_KEYS + PELVIS_EULER_KEYS + JOINT_KEYS

# KEYS of quaternion dataset
QKEYS = PELVIS_POS_KEYS + PELVIS_QUAT_KEYS + JOINT_KEYS

class Trajectory(object):
    """
    Builds a general trajectory from a numpy bin file(.npy), and automatically
    synchronizes the trajectory timestep to the desired control timestep while
    also allowing to change it's speed by the desired amount. When using
    periodic trajectories it is also possible to pass split points which signal
    the points where the trajectory repeats, and provides an utility to select
    the desired cycle.

    """
    def __init__(self, traj_path, traj_dt=0.01, control_dt=0.01,
                 traj_speed_mult=1.0, ignore_keys=[]):
        """
        Constructor.

        Args:
            traj_path (string): path with the trajectory for the
                model to follow. Should be a numpy zipped file (.npz)
                with a 'trajectory_data' array and possibly a
                'split_points' array inside. The 'trajectory_data'
                should be in the shape (joints x observations);
            traj_dt (float, 0.01): time step of the trajectory file;
            control_dt (float, 0.01): model control frequency (used to
                synchronize trajectory with the control step);
            traj_speed_mult (float, 1.0): factor to speed up or slowdown the
                trajectory velocity.

        """
        self._trajectory_files = np.load(traj_path, allow_pickle=True)

        # make new keys, one for joint position and one for joint velocity
        keys = ["q_"+k for k in QKEYS] + ["dq_"+k for k in EKEYS]

        if "goal" in self._trajectory_files.keys():
            keys += ["goal"]

        # remove unwanted keys
        for ik in ignore_keys:
            keys.remove(ik)

        self.trajectory = np.array([self._trajectory_files[key] for key in keys])

        if "split_points" in self._trajectory_files.files:
            self.split_points = trajectory_files["split_points"]
        else:
            self.split_points = np.array([0, self.trajectory.shape[1]])

        self.n_repeating_steps = len(self.split_points) - 1

        self.traj_dt = traj_dt
        self.control_dt = control_dt
        self.traj_speed_multiplier = traj_speed_mult

        if self.traj_dt != control_dt or traj_speed_mult != 1.0:
            new_traj_sampling_factor = (1 / traj_speed_mult) * (
                    self.traj_dt / control_dt)

            self.trajectory = self._interpolate_trajectory(
                self.trajectory, factor=new_traj_sampling_factor
            )

            self.split_points = np.round(
                self.split_points * new_traj_sampling_factor).astype(np.int32)

    def create_dataset(self, normalizer=None):

        # get relevant data
        states = np.transpose(deepcopy(self.trajectory))

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


    def _interpolate_trajectory(self, traj, factor):
        x = np.arange(traj.shape[1])
        x_new = np.linspace(0, traj.shape[1] - 1, round(traj.shape[1] * factor),
                            endpoint=True)
        new_traj = interpolate.interp1d(x, traj, kind="cubic", axis=1)(x_new)
        return new_traj

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
