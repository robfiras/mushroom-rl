import time
from copy import deepcopy
from time import perf_counter
from contextlib import contextmanager

from mushroom_rl.utils.angles import euler_to_quat, quat_to_euler
from mushroom_rl.utils.running_stats import RunningAveragedWindow
from mushroom_rl.environments.mujoco import ObservationType
from mushroom_rl.environments.mujoco_envs.humanoids.reward_goals.trajectory import Trajectory

import matplotlib.pyplot as plt
import mujoco_py
import numpy as np
from scipy import signal, interpolate


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
        # reset x and y to middle position
        self.subtraj[0, :] -= self.subtraj[0, self.subtraj_step_no]
        self.subtraj[1, :] -= self.subtraj[1, self.subtraj_step_no]

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
        running_mean = RunningAveragedWindow(shape=(1,), window_size=500)
        viewer = mujoco_py.MjViewer(self.sim)
        viewer._render_every_frame = False
        self.reset_trajectory(substep_no=1)
        while True:
            with catchtime() as t:
                if self.subtraj_step_no >= self.traj_length:
                    self.get_next_sub_trajectory()

                self.sim.data.qpos[0:17] = self.subtraj[0:17, self.subtraj_step_no]
                self.sim.data.qvel[0:17] = self.subtraj[17:33, self.subtraj_step_no]
                running_mean.update_stats(self.subtraj[17:18, self.subtraj_step_no])
                print("Running mean: ", running_mean.mean)
                self.sim.forward()

                self.subtraj_step_no += 1
                sleep_time = np.maximum(1/freq - t(), 0.0)
                time.sleep(sleep_time)
                viewer.render()

                # check if the humanoid has fallen
                torso_euler = quat_to_euler(self.sim.data.qpos[3:7])
                z_pos = self.sim.data.qpos[2]
                has_fallen = ((z_pos < 0.90) or (z_pos > 1.20) or abs(torso_euler[0]) > np.pi / 12 or (torso_euler[1] < -np.pi / 12) or (torso_euler[1] > np.pi / 8))
                              #or (torso_euler[2] < -np.pi / 4) or (torso_euler[2] > np.pi / 4))

                if has_fallen:
                    print("HAS FALLEN!")
                    return


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
            with catchtime() as t:
                if self.subtraj_step_no >= self.traj_length:
                    self.get_next_sub_trajectory()

                # get velocities
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
                sleep_time = np.maximum(1 / freq - t(), 0.0)
                time.sleep(sleep_time)
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

@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start
