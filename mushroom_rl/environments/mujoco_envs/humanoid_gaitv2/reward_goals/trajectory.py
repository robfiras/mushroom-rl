import time
from copy import deepcopy
from mushroom_rl.utils.angles import euler_to_quat, quat_to_euler

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
        trajectory_files = np.load(traj_path, allow_pickle=True)

        # make new keys, one for joint position and one for joint velocity
        keys = ["q_"+k for k in QKEYS] + ["dq_"+k for k in EKEYS]

        # remove unwanted keys
        for ik in ignore_keys:
            keys.remove(ik)

        self.trajectory = np.array([trajectory_files[key] for key in keys])

        if "split_points" in trajectory_files.files:
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
        numy = round(traj.shape[1] * factor)
        no_round = traj.shape[1] * factor
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


class HumanoidTrajectory(Trajectory):
    """
    Loads a trajectory to be used by the humanoid environment. The trajectory
    file should be structured as:
    trajectory[0:15] -> model's qpos;
    trajectory[15:29] -> model's qvel;
    trajectory[29:34] -> model's foot vector position
    trajectory[34:36] -> model's ground force reaction over z

    """
    def __init__(self, sim, traj_path, traj_dt=0.005,
                 control_dt=0.005, traj_speed_mult=1.0,
                 velocity_smooth_window=1001):
        """
        Constructor.

        Args:
            sim (MjSim): Mujoco simulation object which is passed to
                the Humanoid Trajectory as is used to set model to
                trajectory corresponding initial state;
            traj_path (string): path with the trajectory for the
                model to follow. Should be a numpy zipped file (.npz)
                with a 'trajectory_data' array and possibly a
                'split_points' array inside. The 'trajectory_data'
                should be in the shape (joints x observations);
            traj_dt (float, 0.0025): time step of the trajectory file;
            control_dt (float, 0.005): Model control frequency(used to
                synchronize trajectory with the control step)
            traj_speed_mult (float, 1.0): factor to speed up or slowdown the
                trajectory velocity;
            velocity_smooth_window (int, 1001): size of window used to average
                the torso velocity. It is used in order to get the average
                travelling velocity(as walking velocity from humanoids
                are sinusoidal).

        """
        super().__init__(traj_path, traj_dt, control_dt, traj_speed_mult)

        self.sim = sim
        self.trajectory[15:29] *= traj_speed_mult

        self.complete_velocity_profile = self._smooth_vel_profile(
                self.trajectory[15:18],  window_size=velocity_smooth_window)

        self.subtraj_step_no = 0
        self.x_dist = 0

        self.subtraj = self.trajectory.copy()
        self.velocity_profile = self.complete_velocity_profile.copy()
        self.reset_trajectory()

    @property
    def traj_length(self):
        return self.subtraj.shape[1]

    def _get_traj_gait_sub_steps(self, initial_walking_step,
                                 number_of_walking_steps=1):
        start_sim_step = self.split_points[initial_walking_step]
        end_sim_step = self.split_points[
            initial_walking_step + number_of_walking_steps
        ]

        sub_traj = self.trajectory[:, start_sim_step:end_sim_step].copy()
        initial_x_pos = self.trajectory[0][start_sim_step]
        sub_traj[0, :] -= initial_x_pos

        sub_vel_profile = self.complete_velocity_profile[
                          :, start_sim_step:end_sim_step].copy()

        return sub_traj, sub_vel_profile

    def _smooth_vel_profile(self, vel, use_simple_mean=False, window_size=1001,
                            polyorder=2):
        if use_simple_mean:
            filtered = np.tile(np.mean(vel, axis=1),
                               reps=(self.trajectory.shape[1], 1)).T
        else:
            filtered = signal.savgol_filter(vel, window_length=window_size,
                                            polyorder=polyorder, axis=1)
        return filtered

    def reset_trajectory(self, substep_no=None):
        """
        Resets the trajectory and the model. The trajectory can be forced
        to start on the 'substep_no' if desired, else it starts at
        a random one.

        Args:
            substep_no (int, None): starting point of the trajectory.
                If None, the trajectory starts from a random point.
        """
        self.x_dist = 0
        if substep_no is None:
            self.subtraj_step_no = int(np.random.rand() * (
                    self.traj_length * 0.45))
        else:
            self.subtraj_step_no = substep_no

        self.subtraj = self.trajectory.copy()
        self.subtraj[0, :] -= self.subtraj[0, self.subtraj_step_no]

        self.sim.data.qpos[0:17] = self.subtraj[0:17, self.subtraj_step_no]
        self.sim.data.qvel[0:16] = self.subtraj[17:33, self.subtraj_step_no]

    def get_next_sub_trajectory(self):
        """
        Get the next trajectory once the current one reaches it's end.

        """
        self.x_dist += self.subtraj[0][-1]
        self.reset_trajectory()

    def play_trajectory_demo(self, freq=200):
        """
        Plays a demo of the loaded trajectory by forcing the model
        positions to the ones in the reference trajectory at every step

        """
        viewer = mujoco_py.MjViewer(self.sim)
        viewer._render_every_frame = False
        self.reset_trajectory(substep_no=1)
        curr_qpos = self.subtraj[0:17, self.subtraj_step_no]
        while True:
            if self.subtraj_step_no >= self.traj_length:
                self.get_next_sub_trajectory()

            # self.sim.data.qpos[0:17] = np.r_[
            #     self.x_dist + self.subtraj[0, self.subtraj_step_no],
            #     self.subtraj[1:17, self.subtraj_step_no]
            # ]

            self.sim.data.qpos[0:17] = self.subtraj[0:17, self.subtraj_step_no]
            self.sim.data.qvel[0:17] = self.subtraj[17:33, self.subtraj_step_no]
            self.sim.forward()

            curr_qpos = self.sim.data.qpos[:]

            self.subtraj_step_no += 1
            time.sleep(1 / freq)
            viewer.render()

    def play_trajectory_demo_from_velocity(self, freq=200):
        """
        Plays a demo of the loaded trajectory by forcing the model
        positions to the ones in the reference trajectory at every steps
        """
        viewer = mujoco_py.MjViewer(self.sim)
        viewer._render_every_frame = False
        self.reset_trajectory(substep_no=1)
        curr_qpos = self.subtraj[0:17, self.subtraj_step_no]
        while True:
            if self.subtraj_step_no >= self.traj_length:
                self.get_next_sub_trajectory()

            # get velocities (we omit pelvis rotations here)
            pelvis_v = self.subtraj[17:20, self.subtraj_step_no]
            pelvis_v_rot = self.subtraj[20:23, self.subtraj_step_no]
            dq = self.subtraj[23:33, self.subtraj_step_no]

            # update positions in simulation
            self.sim.data.qpos[0:3] = curr_qpos[0:3] + self.control_dt * pelvis_v
            new_orientation_euler = quat_to_euler(curr_qpos[3:7], 'XYZ') + self.control_dt * pelvis_v_rot
            self.sim.data.qpos[3:7] = euler_to_quat(new_orientation_euler, 'XYZ')
            self.sim.data.qpos[7:17] = curr_qpos[7:17] + self.control_dt * dq




            self.sim.forward()

            # save current qpos
            curr_qpos = self.sim.data.qpos[:]

            self.subtraj_step_no += 1
            time.sleep(1 / freq)
            viewer.render()

    def _plot_joint_trajectories(self, n_points=2000):
        """
        Plots the joint trajectories(qpos / qvel) in case the user wishes
            to consult them.

        """
        fig, ax = plt.subplots(2, 8, figsize=(15 * 8, 15))
        fig.suptitle("Complete Trajectories Sample", size=25)

        for j in range(8):
            ax[0, j].plot(self.subtraj[7 + j, 0:n_points])
            ax[0, j].legend(["Joint {} pos".format(j)])

            ax[1, j].plot(self.subtraj[7 + j + 14, 0:n_points])
            ax[1, j].plot(np.diff(
                self.subtraj[7 + j, 0:n_points]) / self.control_dt)
            ax[1, j].legend(["Joint {} vel".format(j), "derivate of pos"])
        plt.show()
