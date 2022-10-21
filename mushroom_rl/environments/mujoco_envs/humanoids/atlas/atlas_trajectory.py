import time
import numpy as np

from time import perf_counter
from contextlib import contextmanager

from mushroom_rl.environments.mujoco import ObservationType
from mushroom_rl.environments.mujoco_envs.humanoids.full_humanoid.full_humanoid_trajectory import FullHumanoidTrajectory

from mushroom_rl.environments.mujoco_envs.humanoids.utils import quat_to_euler, euler_to_quat


@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start


class AtlasTrajectory(FullHumanoidTrajectory):

    def play_trajectory_demo(self, mdp, freq=200, view_from_other_side=False):
        """
        Plays a demo of the loaded trajectory by forcing the model
        positions to the ones in the reference trajectory at every step

        """
        viewer = mujoco_py.MjViewer(self.sim)
        viewer._render_every_frame = False
        self.reset_trajectory(substep_no=1)

        if view_from_other_side:
            from mujoco_py.generated import const
            viewer.cam.type = const.CAMERA_TRACKING
            viewer.cam.trackbodyid = 0
            viewer.cam.distance *= 0.3
            viewer.cam.elevation = -0  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
            viewer.cam.azimuth = 270

        while True:
            with catchtime() as t:
                if self.subtraj_step_no >= self.traj_length:
                    self.get_next_sub_trajectory()

                # get data
                root_pose = self.subtraj[0:7, self.subtraj_step_no]
                qpos = self.subtraj[7:17, self.subtraj_step_no]
                root_vel = self.subtraj[17:23, self.subtraj_step_no]
                q_vel = self.subtraj[23:33, self.subtraj_step_no]
                state = [root_pose, *qpos, root_vel, *q_vel]

                for name_ot, value in zip(mdp._observation_map, state):
                    name, ot = name_ot
                    if ot == ObservationType.JOINT_POS:
                        mdp._sim.data.set_joint_qpos(name, value)
                    elif ot == ObservationType.JOINT_VEL:
                        mdp._sim.data.set_joint_qvel(name, value)

                self.sim.forward()

                self.subtraj_step_no += 1
                sleep_time = np.maximum(1/freq - t(), 0.0)
                time.sleep(sleep_time)
                viewer.render()

                # check if the humanoid has fallen
                torso_euler = quat_to_euler(self.sim.data.qpos[3:7])
                z_pos = self.sim.data.qpos[2]
                has_fallen = ((z_pos < 0.78) or (z_pos > 1.20) or abs(torso_euler[0]) > np.pi / 12 or (
                            torso_euler[1] < -np.pi / 12) or (torso_euler[1] > np.pi / 8))
                # or (torso_euler[2] < -np.pi / 4) or (torso_euler[2] > np.pi / 4))

                if has_fallen:
                    print("HAS FALLEN!")
                    #return

    def play_trajectory_demo_from_velocity(self, mdp, freq=200, view_from_other_side=False):
        """
        Plays a demo of the loaded trajectory by forcing the model
        positions to the ones in the reference trajectory at every steps
        """
        viewer = mujoco_py.MjViewer(self.sim)
        viewer._render_every_frame = False
        self.reset_trajectory(substep_no=1)
        curr_root_pose = self.subtraj[0:7, self.subtraj_step_no]
        curr_qpos = self.subtraj[7:17, self.subtraj_step_no]

        if view_from_other_side:
            from mujoco_py.generated import const
            viewer.cam.type = const.CAMERA_TRACKING
            viewer.cam.trackbodyid = 0
            viewer.cam.distance *= 0.3
            viewer.cam.elevation = -0  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
            viewer.cam.azimuth = 270

        # order of keys for atlas from xml
        ATLAS_JOINT_KEYS = ["hip_rotation_l", "hip_adduction_l", "hip_flexion_l", "knee_angle_l",
                            "ankle_angle_l", "hip_rotation_r", "hip_adduction_r", "hip_flexion_r", "knee_angle_r",
                            "ankle_angle_r", ]

        joint_indices = dict(zip(JOINT_KEYS, [i for i in range(len(JOINT_KEYS))]))
        reordered_keys = [joint_indices[k] for k in ATLAS_JOINT_KEYS]

        while True:
            with catchtime() as t:
                if self.subtraj_step_no >= self.traj_length:
                    self.get_next_sub_trajectory()

                # get data
                root_vel = self.subtraj[17:23, self.subtraj_step_no]
                q_vel = self.subtraj[23:33, self.subtraj_step_no]

                # update position based on velocities
                new_root_pos = curr_root_pose[0:3] + self.control_dt * root_vel[0:3]
                new_orientation_euler = quat_to_euler(curr_root_pose[3:7], 'XYZ') + self.control_dt * root_vel[3:]
                new_root_orientation = euler_to_quat(new_orientation_euler, 'XYZ')
                root_pose = np.concatenate([new_root_pos, new_root_orientation])
                qpos = curr_qpos + self.control_dt * q_vel

                state = [root_pose, *qpos, root_vel, *q_vel]

                for name_ot, value in zip(mdp._observation_map, state):
                    name, ot = name_ot
                    if ot == ObservationType.JOINT_POS:
                        mdp._sim.data.set_joint_qpos(name, value)
                    elif ot == ObservationType.JOINT_VEL:
                        mdp._sim.data.set_joint_qvel(name, value)

                self.sim.forward()

                # save current qpos
                curr_root_pose = self.sim.data.qpos[0:7]
                curr_qpos[reordered_keys] = self.sim.data.qpos[7:17]

                self.subtraj_step_no += 1
                sleep_time = np.maximum(1 / freq - t(), 0.0)
                time.sleep(sleep_time)
                viewer.render()

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

        # order of keys for atlas from xml
        ATLAS_JOINT_KEYS = ["hip_rotation_l", "hip_adduction_l", "hip_flexion_l", "knee_angle_l",
                            "ankle_angle_l", "hip_rotation_r", "hip_adduction_r", "hip_flexion_r", "knee_angle_r",
                            "ankle_angle_r", ]

        joint_indices = dict(zip(JOINT_KEYS, [i for i in range(len(JOINT_KEYS))]))
        reordered_keys = [joint_indices[k] for k in ATLAS_JOINT_KEYS]

        self.sim.data.qpos[0:7] = self.subtraj[0:7, self.subtraj_step_no]
        self.sim.data.qpos[7:17] = self.subtraj[7:17, self.subtraj_step_no][reordered_keys]
        self.sim.data.qvel[0:6] = self.subtraj[17:23, self.subtraj_step_no]
        self.sim.data.qvel[6:16] = self.subtraj[23:33, self.subtraj_step_no][reordered_keys]
