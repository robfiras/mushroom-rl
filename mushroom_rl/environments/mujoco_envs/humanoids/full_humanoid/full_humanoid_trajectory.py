import time
from copy import deepcopy
from time import perf_counter
from contextlib import contextmanager

from mushroom_rl.utils.angles import euler_to_quat, quat_to_euler
from mushroom_rl.utils.running_stats import RunningAveragedWindow
from mushroom_rl.environments.mujoco import ObservationType
from mushroom_rl.environments.mujoco_envs.humanoids.reward_goals.trajectory import Trajectory
from mushroom_rl.environments.mujoco_envs.humanoids.full_humanoid.full_humanoid import FullHumanoid

import matplotlib.pyplot as plt
import mujoco_py
import numpy as np
from scipy import signal, interpolate

# EKEYS = [ "pelvis_tx", "pelvis_tz", "pelvis_ty", "pelvis_tilt", "pelvis_list", "pelvis_rotation", "hip_flexion_r",
#          "hip_adduction_r", "hip_rotation_r", "knee_angle_r_translation2", "knee_angle_r_translation1", "knee_angle_r",
#          "knee_angle_r_rotation2", "knee_angle_r_rotation3", "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r",
#          "knee_angle_r_beta_translation2", "knee_angle_r_beta_translation1", "knee_angle_r_beta_rotation1",
#          "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l_translation2", "knee_angle_l_translation1",
#          "knee_angle_l", "knee_angle_l_rotation2", "knee_angle_l_rotation3", "ankle_angle_l", "subtalar_angle_l",
#          "mtp_angle_l", "knee_angle_l_beta_translation2", "knee_angle_l_beta_translation1", "knee_angle_l_beta_rotation1",
#          "lumbar_extension", "lumbar_bending", "lumbar_rotation", "arm_flex_r", "arm_add_r", "arm_rot_r",
#          "elbow_flex_r", "pro_sup_r", "wrist_flex_r", "wrist_dev_r", "arm_flex_l", "arm_add_l", "arm_rot_l",
#          "elbow_flex_l", "pro_sup_l", "wrist_flex_l", "wrist_dev_l"]


class FullHumanoidTrajectory():
    """
    Loads a trajectory to be used by the full humanoid environment. The trajectory
    file should be structured as:
    trajectory[0:51] -> model's qpos;
    trajectory[51:102] -> model's qvel;
    """
    def __init__(self, sim, traj_path, traj_dt=0.002, control_dt=0.01, ignore_keys=[]):
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
        traj_speed_mult = 1.0

        self._trajectory_files = np.load(traj_path, allow_pickle=True)

        # make new keys, one for joint position and one for joint velocity
        keys = ["q_"+k for k in EKEYS] + ["dq_"+k for k in EKEYS]

        if "goal" in self._trajectory_files.keys():
            keys += ["goal"]

        # needed for deep mimic
        if "rel_feet_xpos_r" in self._trajectory_files.keys():
            keys += FOOT_KEYS

        # remove unwanted keys
        for ik in ignore_keys:
            keys.remove(ik)

        self.trajectory = np.array([self._trajectory_files[key] for key in keys])
        self.keys = keys

        if "split_points" in self._trajectory_files.keys():
            self.split_points = self._trajectory_files["split_points"]
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

        self.sim = sim

        self.complete_velocity_profile = self.trajectory[51:102]

        self.subtraj_step_no = 0
        self.x_dist = 0

        self.subtraj = self.trajectory.copy()
        self.velocity_profile = self.complete_velocity_profile.copy()
        #self.reset_trajectory()

    @property
    def traj_length(self):
        return self.subtraj.shape[1]

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

        self.sim.data.qpos[0:51] = self.subtraj[0:51, self.subtraj_step_no]
        self.sim.data.qvel[0:51] = self.subtraj[51:102, self.subtraj_step_no]

    def get_next_sub_trajectory(self):
        """
        Get the next trajectory once the current one reaches it's end.

        """
        self.x_dist += self.subtraj[0][-1]
        self.reset_trajectory()

    def play_trajectory_demo(self, freq=200, view_from_other_side=False):
        """
        Plays a demo of the loaded trajectory by forcing the model
        positions to the ones in the reference trajectory at every step

        """
        running_mean = RunningAveragedWindow(shape=(1,), window_size=500)
        viewer = mujoco_py.MjViewer(self.sim)
        viewer._render_every_frame = False
        if view_from_other_side:
            from mujoco_py.generated import const
            viewer.cam.type = const.CAMERA_TRACKING
            viewer.cam.trackbodyid = 0
            viewer.cam.distance *= 0.3
            viewer.cam.elevation = -0  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
            viewer.cam.azimuth = 270
        self.reset_trajectory(substep_no=1)
        while True:
            with catchtime() as t:
                if self.subtraj_step_no >= self.traj_length:
                    self.get_next_sub_trajectory()

                self.sim.data.qpos[0:51] = self.subtraj[0:51, self.subtraj_step_no]
                self.sim.data.qvel[0:51] = self.subtraj[51:102, self.subtraj_step_no]
                running_mean.update_stats(self.subtraj[51:52, self.subtraj_step_no])
                print("Running mean: ", running_mean.mean)
                self.sim.forward()

                self.subtraj_step_no += 1
                sleep_time = np.maximum(1 / freq - t(), 0.0)
                time.sleep(sleep_time)
                viewer.render()

                # check if the humanoid has fallen
                torso_euler = quat_to_euler(self.sim.data.qpos[3:7])
                z_pos = self.sim.data.qpos[2]

                pelvis_euler = self.sim.data.qpos[3:6]
                pelvis_condition = ((self.sim.data.qpos[2] < -0.35) or (self.sim.data.qpos[2] > 0.10)
                                    or (pelvis_euler[0] < (-np.pi / 4.5)) or (pelvis_euler[0] > (np.pi / 12))
                                    or (pelvis_euler[1] < -np.pi / 12) or (pelvis_euler[1] > np.pi / 8)
                                    or (pelvis_euler[2] < (-np.pi / 10)) or (pelvis_euler[2] > (np.pi / 10))
                                    )
                lumbar_euler = self.sim.data.qpos[37:40]
                lumbar_condition = ((lumbar_euler[0] < (-np.pi / 4.5)) or (lumbar_euler[0] > (np.pi / 12))
                                    or (lumbar_euler[1] < -np.pi / 5) or (lumbar_euler[1] > np.pi / 5)
                                    or (lumbar_euler[2] < (-np.pi / 4.5)) or (lumbar_euler[2] > (np.pi / 4.5))
                                    )
                has_fallen = pelvis_condition or lumbar_condition
                if has_fallen:
                    print("HAS FALLEN!")
                    print("Lumbar_condition:", lumbar_condition)
                    print("Pelvis_condition:", pelvis_condition)


    def play_trajectory_demo_from_velocity(self, freq=200, view_from_other_side=False):
        """
        Plays a demo of the loaded trajectory by forcing the model
        positions to the ones in the reference trajectory at every steps
        """
        viewer = mujoco_py.MjViewer(self.sim)
        viewer._render_every_frame = False
        if view_from_other_side:
            from mujoco_py.generated import const
            viewer.cam.type = const.CAMERA_TRACKING
            viewer.cam.trackbodyid = 0
            viewer.cam.distance *= 0.3
            viewer.cam.elevation = -0  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
            viewer.cam.azimuth = 270
        self.reset_trajectory(substep_no=1)
        curr_qpos = self.subtraj[0:51, self.subtraj_step_no]

        while True:
            with catchtime() as t:
                if self.subtraj_step_no >= self.traj_length:
                    self.get_next_sub_trajectory()

                # get velocities
                pelvis_v = self.subtraj[51:54, self.subtraj_step_no]
                pelvis_v_rot = self.subtraj[54:57, self.subtraj_step_no]
                dq = self.subtraj[57:102, self.subtraj_step_no]

                # update positions in simulation
                self.sim.data.qpos[0:3] = curr_qpos[0:3] + self.control_dt * pelvis_v
                self.sim.data.qpos[3:6] = curr_qpos[3:6] + self.control_dt * pelvis_v_rot
                self.sim.data.qpos[6:51] = curr_qpos[6:51] + self.control_dt * dq

                self.sim.forward()

                # save current qpos
                curr_qpos = self.sim.data.qpos[:]

                self.subtraj_step_no += 1
                sleep_time = np.maximum(1 / freq - t(), 0.0)
                time.sleep(sleep_time)
                viewer.render()

@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start
