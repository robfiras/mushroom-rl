
import os
import time
from abc import abstractmethod
import mujoco
from dm_control import mjcf


from pathlib import Path

import numpy as np
from time import perf_counter
from contextlib import contextmanager

from mushroom_rl.environments.mujoco import MuJoCo, ObservationType
from pathlib import Path

from mushroom_rl.utils import spaces
from mushroom_rl.utils.angles import quat_to_euler
from mushroom_rl.utils.running_stats import *
from mushroom_rl.utils.mujoco import *
from mushroom_rl.environments.mujoco_envs.humanoids.trajectory import Trajectory
from mushroom_rl.environments.mujoco_envs.quadrupeds.base_quadruped import BaseQuadruped

from mushroom_rl.environments.mujoco_envs.humanoids.reward import NoGoalReward, CustomReward

# optional imports
try:
    mujoco_viewer_available = True
    import mujoco_viewer
except ModuleNotFoundError:
    mujoco_viewer_available = False


# TODO needs changes:


# TODO --------------------------------------------------------------------------------------------------------------------------------------------------------------
DIM_OF_OBS_TYPE = {
    ObservationType.BODY_POS: 3,
    ObservationType.BODY_ROT: 4,
    ObservationType.BODY_VEL: 6,
    ObservationType.JOINT_POS: 1,  # can be 7
    ObservationType.JOINT_VEL: 1,  # can be 6
    ObservationType.SITE_POS: 3,
    ObservationType.SITE_ROT: 9
}

class UnitreeA1(BaseQuadruped):
    """
    Mujoco simulation of unitree A1 model
    to switch between torque and position control: adjust xml file (and if needed action_position.npz/action_position.npz)
    to switch between freejoint and mul_joint: adapt obs space and xml path
    """
    def __init__(self, gamma=0.99, horizon=1000, n_substeps=10, random_start=False, init_step_no=None,
                 traj_params=None, timestep=0.001, goal_reward=None, goal_reward_params=None, use_torque_ctrl=False,
                 use_2d_ctrl=False, tmp_dir_name=None):
        """
        Constructor.
        for clipping in torques need to adjust xml gear 34 and ctrllimited
        """
        if use_torque_ctrl:
            xml_path = (Path(__file__).resolve().parent.parent / "data" / "quadrupeds" /
                    "unitree_a1_torque_mul_joint.xml").as_posix()
            print("Using torque control for unitreeA1")
        else:
            xml_path = (Path(__file__).resolve().parent.parent / "data" / "quadrupeds" /
                        "unitree_a1_position_mul_joint.xml").as_posix()
            print("Using position control for unitreeA1")

        action_spec = [# motors
            "FR_hip", "FR_thigh", "FR_calf",
            "FL_hip", "FL_thigh", "FL_calf",
            "RR_hip", "RR_thigh", "RR_calf",
            "RL_hip", "RL_thigh", "RL_calf"]
        observation_spec = [
            # ------------------- JOINT POS -------------------
            # --- Trunk ---
            #("body_freejoint", "body", ObservationType.JOINT_POS),
            ("q_trunk_tx", "trunk_tx", ObservationType.JOINT_POS),
            ("q_trunk_ty", "trunk_ty", ObservationType.JOINT_POS),
            ("q_trunk_tz", "trunk_tz", ObservationType.JOINT_POS),
            ("q_trunk_tilt", "trunk_tilt", ObservationType.JOINT_POS),
            ("q_trunk_list", "trunk_list", ObservationType.JOINT_POS),
            ("q_trunk_rotation", "trunk_rotation", ObservationType.JOINT_POS),
            # --- Front ---
            ("q_FR_hip_joint", "FR_hip_joint", ObservationType.JOINT_POS),
            ("q_FR_thigh_joint", "FR_thigh_joint", ObservationType.JOINT_POS),
            ("q_FR_calf_joint", "FR_calf_joint", ObservationType.JOINT_POS),
            ("q_FL_hip_joint", "FL_hip_joint", ObservationType.JOINT_POS),
            ("q_FL_thigh_joint", "FL_thigh_joint", ObservationType.JOINT_POS),
            ("q_FL_calf_joint", "FL_calf_joint", ObservationType.JOINT_POS),
            # --- Rear ---
            ("q_RR_hip_joint", "RR_hip_joint", ObservationType.JOINT_POS),
            ("q_RR_thigh_joint", "RR_thigh_joint", ObservationType.JOINT_POS),
            ("q_RR_calf_joint", "RR_calf_joint", ObservationType.JOINT_POS),
            ("q_RL_hip_joint", "RL_hip_joint", ObservationType.JOINT_POS),
            ("q_RL_thigh_joint", "RL_thigh_joint", ObservationType.JOINT_POS),
            ("q_RL_calf_joint", "RL_calf_joint", ObservationType.JOINT_POS),
            # ------------------- JOINT VEL -------------------
            # --- Trunk ---
            ("dq_trunk_tx", "trunk_tx", ObservationType.JOINT_VEL),
            ("dq_trunk_tz", "trunk_tz", ObservationType.JOINT_VEL),
            ("dq_trunk_ty", "trunk_ty", ObservationType.JOINT_VEL),
            ("dq_trunk_tilt", "trunk_tilt", ObservationType.JOINT_VEL),
            ("dq_trunk_list", "trunk_list", ObservationType.JOINT_VEL),
            ("dq_trunk_rotation", "trunk_rotation", ObservationType.JOINT_VEL),
            # --- Front ---
            ("dq_FR_hip_joint", "FR_hip_joint", ObservationType.JOINT_VEL),
            ("dq_FR_thigh_joint", "FR_thigh_joint", ObservationType.JOINT_VEL),
            ("dq_FR_calf_joint", "FR_calf_joint", ObservationType.JOINT_VEL),
            ("dq_FL_hip_joint", "FL_hip_joint", ObservationType.JOINT_VEL),
            ("dq_FL_thigh_joint", "FL_thigh_joint", ObservationType.JOINT_VEL),
            ("dq_FL_calf_joint", "FL_calf_joint", ObservationType.JOINT_VEL),
            # --- Rear ---
            ("dq_RR_hip_joint", "RR_hip_joint", ObservationType.JOINT_VEL),
            ("dq_RR_thigh_joint", "RR_thigh_joint", ObservationType.JOINT_VEL),
            ("dq_RR_calf_joint", "RR_calf_joint", ObservationType.JOINT_VEL),
            ("dq_RL_hip_joint", "RL_hip_joint", ObservationType.JOINT_VEL),
            ("dq_RL_thigh_joint", "RL_thigh_joint", ObservationType.JOINT_VEL),
            ("dq_RL_calf_joint", "RL_calf_joint", ObservationType.JOINT_VEL)]

        collision_groups = [("floor", ["floor"]),
                            ("foot_FR", ["FR_foot"]),
                            ("foot_FL", ["FL_foot"]),
                            ("foot_RR", ["RR_foot"]),
                            ("foot_RL", ["RL_foot"])]

        if use_2d_ctrl:
            observation_spec.append(("dir_arrow", "dir_arrow", ObservationType.SITE_ROT))
            assert tmp_dir_name is not None, "If you want to use 2d_ctrl, you have to specify a" \
                                             "directory name for the xml-files to be saved."
            xml_handle = self.add_dir_vector_to_xml_handle(mjcf.from_path(xml_path))
            xml_path = self.save_xml_handle(xml_handle, tmp_dir_name)
            print("Using 2D Control with direction arrow")
            self.use_2d_ctrl = True
        else:
            self.use_2d_ctrl = False

        i = 0
        self._keys_dim = {}
        for key, name, ot in observation_spec:
            self._keys_dim[i] = DIM_OF_OBS_TYPE[ot]
            i+=1

        super().__init__(xml_path, action_spec, observation_spec, gamma=gamma, horizon=horizon, n_substeps=n_substeps,
                         timestep=timestep, collision_groups=collision_groups, traj_params=traj_params, init_step_no=init_step_no,
                         goal_reward=goal_reward, goal_reward_params=goal_reward_params, random_start=random_start)



    def _modify_observation(self, obs):
        """
        transform direction arrow matrix into one rotation angle

        Args:
            obs (np.ndarray): the generated observation

        Returns:
            The environment observation.

        """
        if self.use_2d_ctrl:
            new_obs = obs[:34]
            # transform rotation matrix into rotation angle
            temp = np.dot(obs[34:43].reshape((3, 3)), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).reshape((9,))
            new_obs = np.append(new_obs, np.arctan2(temp[3], temp[0]))
            new_obs = np.append(new_obs, obs[43:])
            return new_obs
        return obs

    def add_dir_vector_to_xml_handle(self, xml_handle):

        # find trunk and attach direction arrow
        trunk = xml_handle.find("body", "trunk")
        trunk.add("body", name="dir_arrow", pos="0 0 0.15")
        dir_vec = xml_handle.find("body", "dir_arrow")
        dir_vec.add("site", name="dir_arrow_ball", type="sphere", size=".03", pos="-.1 0 0")
        dir_vec.add("site", name="dir_arrow", type="cylinder", size=".01", fromto="-.1 0 0 .1 0 0")

        return xml_handle

    # TODO: copied/modified from reudced_humanoid -> inherit?
    def save_xml_handle(self, xml_handle, tmp_dir_name):

        # save new model and return new xml path
        new_model_dir_name = 'new_unitree_a1_with_dir_vec_model/' + tmp_dir_name + "/"
        cwd = Path.cwd()
        new_model_dir_path = Path.joinpath(cwd, new_model_dir_name)
        xml_file_name = "modified_unitree_a1.xml"
        mjcf.export_with_assets(xml_handle, new_model_dir_path, xml_file_name)
        new_xml_path = Path.joinpath(new_model_dir_path, xml_file_name)

        return new_xml_path.as_posix()

    def setup(self, substep_no=None):
        self.goal_reward.reset_state()
        if self.trajectory is not None:
            if self._random_start:
                sample = self.trajectory.reset_trajectory()
            else:
                sample = self.trajectory.reset_trajectory(self._init_step_no)
            if self.use_2d_ctrl:
                self._direction_xmat = sample[36] #self._trajectory_files["dir_arrow"][1500]
                temp = np.dot(self._direction_xmat.reshape((3,3)), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).reshape((9,))
                self._direction_angle = np.arctan2(temp[3], temp[0])
            self.set_qpos_qvel(sample)
        else: # TODO: add this fuctionality in base_humanoid for all env
            self._data.qpos = [0, 0, -0.16, 0, 0, 0, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8]
            self._data.qvel = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            if self.use_2d_ctrl:

                self._direction_xmat = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0])
                #matrixmult with inverse of default rotation to rewind offset
                temp = np.dot(self._direction_xmat.reshape((3,3)), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).reshape((9,))
                self._direction_angle = np.arctan2(temp[3], temp[0])

    @staticmethod
    def has_fallen(state):
        """
        # with freejoint
        trunk_euler = quat_to_euler(state[3:7])
        trunk_condition = ((trunk_euler[0] < -np.pi * 40 / 180) or (trunk_euler[0] > np.pi * 40 / 180)
                           or (trunk_euler[1] < (-np.pi * 40 / 180)) or (trunk_euler[1] > (np.pi * 40 / 180))
                           )
        """

        # without freejoint


        trunk_euler = state[1:4]
        """
        # old/first strict has_fallen; only for forward walking
        trunk_condition = ((trunk_euler[0] < -0.5) or (trunk_euler[0] > 0.02)
                            or (trunk_euler[1] < -0.095) or (trunk_euler[1] > 0.095)
                            or (trunk_euler[2] < -0.075) or (trunk_euler[2] > 0.075)
                            or state[0] < -.22 #.25
                            ca 30 degree
                            remove z rot
                            )"""
        """
        #less strict has_fallen (old)
        for cluster datasets
        trunk_condition = ((trunk_euler[1] < -0.6981) or (trunk_euler[1] > 0.6981)
                           or (trunk_euler[2] < -0.6981) or (trunk_euler[2] > 0.6981)
                           or state[0] < -.25
                           )
        """

        #new stricter has_fallen, adapted to 8 walking dir
        # minimal height : -0.19103749641009019
        # max x-rot: 0.21976069929211345 -> 12.5914 degree
        # max y-rot: -0.1311784030716909 -> -7.516 degree

        trunk_condition = ((trunk_euler[1] < -0.2793) or (trunk_euler[1] > 0.2793) # x-rotation 16 degree
                           or (trunk_euler[2] < -0.192) or (trunk_euler[2] > 0.192) # y-rotation 11 degree
                           or state[0] < -.24
                           )

        #if trunk_condition:
        #    print("con1: ", (trunk_euler[0] < -0.5) or (trunk_euler[0] > 0.02), trunk_euler[0])
        #    print("con2: ", (trunk_euler[1] < -0.095) or (trunk_euler[1] > 0.095), trunk_euler[1])
        #    print("con3: ", (trunk_euler[2] < -0.075) or (trunk_euler[2] > 0.075), trunk_euler[2])
        #    print("con4: ", state[0] < -.22, state[0])
        #    print(state)
        return trunk_condition

def test_rotate_data(traj_path, store_path='./new_unitree_a1_with_dir_vec_model'):

    trajectory_files = np.load(traj_path, allow_pickle=True)
    trajectory_files = {k: d for k, d in trajectory_files.items()}
    trajectory = np.array([trajectory_files[key].flatten() for key in trajectory_files.keys()],
                               dtype=object)
    rot_diff=[0,np.pi*0.5, np.pi]




    for x in rot_diff:
        R1 = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        R2 = np.array([[np.cos(x), 0, np.sin(x)], [0, 1, 0], [-np.sin(x), 0, np.cos(x)]])
        R3 = np.array([[np.cos(x), -np.sin(x), 0], [np.sin(x), np.cos(x), 0], [0, 0, 1]])
        temp = np.dot(np.dot(np.dot(R1, R2), R3), np.array([trajectory[23], trajectory[22], trajectory[21]]))

        R = np.array([[np.cos(x), -np.sin(x), 0], [np.sin(x), np.cos(x), 0], [0, 0, 1]])
        arrow = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0]).reshape((3, 3))
        rotation_matrix_dir = np.dot(arrow, R)

        dir_arrow=[]
        for i in range(int(len(np.array(trajectory[36]))/9)):
            a = trajectory[36][i*9:i*9+9].reshape((3,3))
            b = np.dot(a, rotation_matrix_dir)
            dir_arrow.append(b.reshape((9,)))

        print("jg")
        np.savez(os.path.join(store_path, 'test_rotate_dataset_'+str(x)+'.npz'),
                 q_trunk_tx=np.cos(x) * np.array(trajectory[0]) - np.sin(x) * np.array(trajectory[1]),
                 q_trunk_ty=np.sin(x) * np.array(trajectory[0]) + np.cos(x) * np.array(trajectory[1]),
                 q_trunk_tz=np.array(trajectory[2]),
                 q_trunk_tilt=np.array(trajectory[3])+x,
                 q_trunk_list=np.array(trajectory[4]),
                 q_trunk_rotation=np.array(trajectory[5]),
                 q_FR_hip_joint=np.array(trajectory[6]),
                 q_FR_thigh_joint=np.array(trajectory[7]),
                 q_FR_calf_joint=np.array(trajectory[8]),
                 q_FL_hip_joint=np.array(trajectory[9]),
                 q_FL_thigh_joint=np.array(trajectory[10]),
                 q_FL_calf_joint=np.array(trajectory[11]),
                 q_RR_hip_joint=np.array(trajectory[12]),
                 q_RR_thigh_joint=np.array(trajectory[13]),
                 q_RR_calf_joint=np.array(trajectory[14]),
                 q_RL_hip_joint=np.array(trajectory[15]),
                 q_RL_thigh_joint=np.array(trajectory[16]),
                 q_RL_calf_joint=np.array(trajectory[17]),
                 dq_trunk_tx=np.cos(x) * np.array(trajectory[18]) - np.sin(x) * np.array(trajectory[19]),
                 dq_trunk_tz=np.sin(x) * np.array(trajectory[18]) + np.cos(x) * np.array(trajectory[19]),
                 dq_trunk_ty=np.array(trajectory[20]),
                 dq_trunk_tilt=trajectory[21],
                 dq_trunk_list=trajectory[22],
                 dq_trunk_rotation=trajectory[23],
                 dq_FR_hip_joint=np.array(trajectory[24]),
                 dq_FR_thigh_joint=np.array(trajectory[25]),
                 dq_FR_calf_joint=np.array(trajectory[26]),
                 dq_FL_hip_joint=np.array(trajectory[27]),
                 dq_FL_thigh_joint=np.array(trajectory[28]),
                 dq_FL_calf_joint=np.array(trajectory[29]),
                 dq_RR_hip_joint=np.array(trajectory[30]),
                 dq_RR_thigh_joint=np.array(trajectory[31]),
                 dq_RR_calf_joint=np.array(trajectory[32]),
                 dq_RL_hip_joint=np.array(trajectory[33]),
                 dq_RL_thigh_joint=np.array(trajectory[34]),
                 dq_RL_calf_joint=np.array(trajectory[35]),
                 dir_arrow=np.array(dir_arrow) )

    return os.path.join(store_path, 'test_rotate_dataset_'+str(np.pi*0.5)+'.npz')



@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start


def interpolate_map(traj):
    traj_list = [list() for j in range(len(traj))]
    #traj_list = [list() for i in range()]
    for i in range(len(traj_list)):
        traj_list[i] = list(traj[i])
    temp = []
    traj_list[36] = [
        np.arctan2(np.dot(mat.reshape((3, 3)), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).reshape((9,))[3],
                   np.dot(mat.reshape((3, 3)), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).reshape((9,))[0])
        for mat in traj[36]]
    # for mat in traj[36].reshape((len(traj[0]), 9)):
    #    arrow = np.dot(mat.reshape((3, 3)), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).reshape((9,))
    #   temp.append(np.arctan2(arrow[3], arrow[0]))
    # traj_list[36] = temp
    return np.array(traj_list)


def interpolate_remap(traj):
    traj_list = [list() for j in range(len(traj))]
    for i in range(len(traj_list)):
        traj_list[i] = list(traj[i])
    traj_list[36] = [
        np.dot(np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]),
               np.array([0, 0, 1, 1, 0, 0, 0, 1, 0]).reshape((3, 3))).reshape((9,)) for angle in traj[36]]
    # for angle in traj[36]:
    #   R = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    #  arrow = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0]).reshape((3, 3))
    # temp = temp + list(np.dot(R, arrow).reshape((9,)))
    # traj_list[36] = temp
    return np.array(traj_list, dtype=object)


if __name__ == '__main__':
    # TODO: different behavior, action control completed?, for clipping in torques need to adjust xml gear 34 and ctrllimited
    """
    #trajectory demo:
    np.random.seed(1)
    # define env and data frequencies
    env_freq = 1000  # hz, added here as a reminder
    traj_data_freq = 500  # hz, added here as a reminder
    desired_contr_freq = 500  # hz
    n_substeps = env_freq // desired_contr_freq

    traj_path = '/home/tim/Documents/locomotion_simulation/log/2D_Walking/states_50k_forward.npz'

    #found solution: adjust npc model
    # weird that opposite directions need different rotations
    # should I adjust the backwalking?
    # remaining question: should the arrow in the dataset be corresponding/relative to the robot
    # Should for same seed etc get the same result? stricter has_fallen seems a little bit better

    #traj_path = test_rotate_data(traj_path, store_path='./new_unitree_a1_with_dir_vec_model')



    # prepare trajectory params
    traj_params = dict(traj_path=traj_path,
                       traj_dt=(1 / traj_data_freq),
                       control_dt=(1 / desired_contr_freq),
                       interpolate_map=interpolate_map, #transforms 9dim rot matrix into one rot angle
                       interpolate_remap=interpolate_remap # and back
                       )
    gamma = 0.99
    horizon = 1000

    env = UnitreeA1(timestep=1/env_freq, gamma=gamma, horizon=horizon, n_substeps=n_substeps,use_torque_ctrl=True,
                    traj_params=traj_params, random_start=True, use_2d_ctrl=True, tmp_dir_name=".")


    with catchtime() as t:
        env.play_trajectory_demo(desired_contr_freq)
        print("Time: %fs" % t())

    print("Finished")
    # still problem with different behaviour (if robot rolls to the side - between freejoint and muljoints) action[1] and [7] = -1 (with action clipping)


    #solref="0.004 1000" /damping 500, stiffness from 0,93 to 62,5
    #0.004 1000000
    #0.004-0.005 1000000 kp=1000
    # favorite 0.005 1000000 | solref="-0.000001 -400"
    # final: solref="-0.0000000001 -250"

    """


    # action demo
    env_freq = 1000  # hz, added here as a reminder simulation freq
    traj_data_freq = 500  # hz, added here as a reminder  controll_freq of data model -> sim_freq/n_substeps
    desired_contr_freq = 500  # hz contl freq.
    n_substeps =  env_freq // desired_contr_freq
    # TODO: unstable so that it falls if des_contr_freq!= data_freq
    #to interpolate
    demo_dt = (1 / traj_data_freq)
    control_dt = (1 / desired_contr_freq)

    #dataset settings:
    #action...50k_noise0... has correct acions for torque and one more datapoint

    actions_path = ['/home/tim/Documents/locomotion_simulation/log/2D_Walking/actions_torque_50k_backward.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/actions_torque_50k_backward_noise1.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/actions_torque_50k_backward_noise2.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/actions_torque_50k_BL.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/actions_torque_50k_BL_noise1.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/actions_torque_50k_BL_noise2.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/actions_torque_50k_BR.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/actions_torque_50k_BR_noise1.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/actions_torque_50k_BR_noise2.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/actions_torque_50k_FL.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/actions_torque_50k_FL_noise1.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/actions_torque_50k_FL_noise2.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/actions_torque_50k_forward.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/actions_torque_50k_forward_noise1.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/actions_torque_50k_forward_noise2.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/actions_torque_50k_FR.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/actions_torque_50k_FR_noise1.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/actions_torque_50k_FR_noise2.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/actions_torque_50k_left.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/actions_torque_50k_left_noise1.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/actions_torque_50k_left_noise2.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/actions_torque_50k_right.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/actions_torque_50k_right_noise1.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/actions_torque_50k_right_noise2.npz'] #actions_torque.npz
    states_path = ['/home/tim/Documents/locomotion_simulation/log/2D_Walking/states_50k_backward.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/states_50k_backward_noise1.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/states_50k_backward_noise2.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/states_50k_BL.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/states_50k_BL_noise1.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/states_50k_BL_noise2.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/states_50k_BR.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/states_50k_BR_noise1.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/states_50k_BR_noise2.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/states_50k_FL.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/states_50k_FL_noise1.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/states_50k_FL_noise2.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/states_50k_forward.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/states_50k_forward_noise1.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/states_50k_forward_noise2.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/states_50k_FR.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/states_50k_FR_noise1.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/states_50k_FR_noise2.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/states_50k_left.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/states_50k_left_noise1.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/states_50k_left_noise2.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/states_50k_right.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/states_50k_right_noise1.npz',
                    '/home/tim/Documents/locomotion_simulation/log/2D_Walking/states_50k_right_noise2.npz']
    #actions_path = '/home/tim/Documents/locomotion_simulation/log/actions_torque.npz'
    #states_path = '/home/tim/Documents/locomotion_simulation/log/2D_Walking/states_50k_right_noise1.npz' #'/home/tim/Documents/locomotion_simulation/log/states.npz'
    dataset_path = '/home/tim/Documents/test_datasets/' #'/home/tim/Documents/IRL_unitreeA1/data/2D_Walking' #'/home/tim/Documents/test_datasets/'#None # '/home/tim/Documents/IRL_unitreeA1/data'
    use_rendering = False # both only for mujoco generated states
    use_plotting = False
    state_type = "optimal"
    action_type = None#"optimal"

    use_2d_ctrl = True
    use_torque_ctrl = True





    assert not (action_type == "p-controller" and not use_torque_ctrl)


    gamma = 0.99
    horizon = 1000



    env = UnitreeA1(timestep=1/env_freq, gamma=gamma, horizon=horizon, n_substeps=n_substeps,
                    use_torque_ctrl=use_torque_ctrl, use_2d_ctrl=use_2d_ctrl, tmp_dir_name=".")

    action_dim = env.info.action_space.shape[0]
    print("Dimensionality of Obs-space:", env.info.observation_space.shape[0])
    print("Dimensionality of Act-space:", env.info.action_space.shape[0])

    env.reset()


    """
    env.play_action_demo(states_path=states_path,
                         #dataset_path=dataset_path,

                         action_path=actions_path,

                         control_dt=control_dt, demo_dt=demo_dt
                        )
    exit()

    
    
    env.play_action_demo2(actions_path=actions_path, states_path=states_path, control_dt=control_dt, demo_dt=demo_dt,
                          use_rendering=True, use_plotting=False, use_pd_controller=True)
    exit()
    """
    if type(actions_path) == list and type(states_path) == list:
        assert len(actions_path) == len(states_path)
        for i in range(len(actions_path)):
            env.preprocess_expert_data(dataset_path=dataset_path,
                                       dataset_name=states_path[i][states_path[i].rfind('/')+7:-4],
                                       state_type=state_type,
                                       action_type=action_type,
                                       states_path=states_path[i],
                                       actions_path=actions_path[i],
                                       use_rendering=use_rendering,
                                       use_plotting=use_plotting,
                                       demo_dt=demo_dt,
                                       control_dt=control_dt
                                       )
    else:
        env.preprocess_expert_data(dataset_path=dataset_path,
                                state_type=state_type,
                                action_type=action_type,
                                states_path=states_path,
                                actions_path=actions_path,
                                use_rendering=use_rendering,
                                use_plotting=use_plotting,
                                demo_dt=demo_dt,
                                control_dt=control_dt
                                )






    #reduce noise; find problem with 250k; concatenate trajectories; stricter has_fallen; generate new datasets

    """

    #general experiments - easier with action clipping

    # action demo - need action clipping to be off
    env_freq = 1000  # hz, added here as a reminder simulation freq
    traj_data_freq = 500  # hz, added here as a reminder  controll_freq of data model -> sim_freq/n_substeps
    desired_contr_freq = 500  # hz contl freq.
    n_substeps =  env_freq // desired_contr_freq
    # TODO: unstable so that it falls if des_contr_freq!= data_freq
    #to interpolate
    demo_dt = (1 / traj_data_freq)
    control_dt = (1 / desired_contr_freq)

    traj_path = '/home/tim/Documents/locomotion_simulation/log/states_temp.npz'

    # traj_path = test_rotate_data(traj_path, store_path='./new_unitree_a1_with_dir_vec_model')

    # prepare trajectory params
    traj_params = dict(traj_path=traj_path,
                       traj_dt=(1 / traj_data_freq),
                       control_dt=(1 / desired_contr_freq))


    gamma = 0.99
    horizon = 1000



    env = UnitreeA1(timestep=1/env_freq, gamma=gamma, horizon=horizon, n_substeps=n_substeps, use_torque_ctrl=True,
                    use_2d_ctrl=True, tmp_dir_name='.', traj_params=traj_params, random_start=True)
    action_dim = env.info.action_space.shape[0]
    print("Dimensionality of Obs-space:", env.info.observation_space.shape[0])
    print("Dimensionality of Act-space:", env.info.action_space.shape[0])

    env.reset()
    env.render()

    absorbing = False
    i = 0
    while True:
        #time.sleep(.1)
        if i == 500:
            env._direction_xmat = np.array([-0.00000,0.00000,-1.00000,-1.00000,0.00000,0.00000,0.00000,1.00000,0.00000]) #np.array([0, 0, 1, 1, 0, 0, 0, 1, 0])
            # matrixmult with inverse of default rotation to rewind offset
            temp = np.dot(env._direction_xmat.reshape((3, 3)), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).reshape(
                (9,))
            env._direction_angle = np.arctan2(temp[3], temp[0])
            #print("------ RESET ------")
            #env.reset()
            i = 0
            absorbing = False

        action = np.random.randn(action_dim)
        nstate, _, absorbing, _ = env.step(action)
        print("angle:",nstate[34], len(nstate))
        print("angle2:",env._direction_angle)

        print(len(env._data.qpos))
        print(len(nstate))
        x = nstate[4]
        R1 = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        x = nstate[3]
        R2 = np.array([[np.cos(x), 0, np.sin(x)], [0, 1, 0], [-np.sin(x), 0, np.cos(x)]])
        x = nstate[3]
        R3 = np.array([[np.cos(x), -np.sin(x), 0], [np.sin(x), np.cos(x), 0], [0, 0, 1]])
        a = env._data
        vx=0
        vy=0.4
        x = np.arctan2(vy, vx)
        print("x ",x)


        R4 = np.array([[np.cos(x), -np.sin(x), 0], [np.sin(x), np.cos(x), 0], [0, 0, 1]])
        arr = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0]).reshape((3,3))
        rot = np.dot(R4, arr).reshape((9,))
        #env._data.site("dir_arrow").xmat = rot #env._data.body("trunk").xmat #np.dot(R3, np.array([0, 0, 1, 1, 0, 0, 0, 1, 0]).reshape((3,3))).reshape((9,))#np.dot(R1, np.dot(R2, np.array([0, 0, 1, 1, 0, 0, 0, 1, 0]).reshape((3,3)))).reshape((9,)) #R3.reshape((9,)) #np.dot(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape((3,3)), R2).reshape((9,))

        #env._data.site("dir_arrow_ball").xpos = env._data.body("dir_arrow").xpos + [-0.1*np.cos(x), -0.1*np.sin(x), 0]


        print(env._obs)



        env.render()
        i += 1



    """

    """
    
    Did:
        finetuned xml - more stable version (changed the position of the mass)
        changed observation space -> changed has_fallen
        leave out initial stepping
        created bigger datasets
        added interpolation to gail/vail -> wrote own create_dataset method
        -> intermediate step -> 
        added traj to gail for init position
        
        refactoring of base_qudrued
        fixed problems because of merging/new constructor of base humanoid
        removed flag action_normalization & generatet dataset
        tried to generate 250k dataset
        
        
    Questions:
        normalization ranges sligthly different
        init weight position ok?
        Talks in the Oberseminar
        
    
    
    """
"""
[ 2.63039797e+01 -1.72159116e+00 -1.79501342e-01 -4.79486061e-02
  9.66714284e-03  2.95516467e-03  1.11039073e-01  7.48249769e-01
 -1.94188781e+00 -1.04573289e-01  8.11246088e-01 -2.30495642e+00
 -4.40014626e-02  8.70473827e-01 -2.27038650e+00  4.75352151e-02
  7.88706282e-01 -1.86333508e+00  5.30650250e-01 -9.58593118e-02
  3.05199296e-02  1.91519152e-01  1.16558842e-01 -1.22728722e-01
  2.11449957e-01  2.35350118e+00 -7.83321843e-01  5.69501340e-02
 -4.40937427e+00  3.06797058e+00 -9.05516404e-02 -5.27219641e+00
  3.25290910e+00 -6.67275858e-03  2.02203667e+00  4.05815065e-02]
60024"""



"""

    def fit(self, dataset):
        state, action, reward, next_state, absorbing, last = parse_dataset(dataset)
        for x in np.arange(0, 2*np.pi, np.pi/180):
            #TODO: stimmt dass in state nicht x,y coord?? --------------------------------------------------------------
            new_state = state.copy()
            #rotate y rotation of trunk
            new_state[1]+=x
            #rotate x,y velocity of trunk
            new_state[16] = np.cos(x) * state[16] - np.sin(x) * state[17]
            new_state[17] = np.sin(x) * state[16] + np.cos(x) * state[17]

            #rotate direction arrow
            R = np.array([[np.cos(x), -np.sin(x), 0], [np.sin(x), np.cos(x), 0], [0, 0, 1]])
            arrow = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0]).reshape((3, 3))
            rotation_matrix_dir = np.dot(arrow, R)
            new_state[34] = np.dot(state[34].reshape((3, 3)), rotation_matrix_dir).reshape((9,))
            else
            new_state[34] += x

            #TODO test if direction  arrow is matrix or angle
            #TODO Wichtig auch in datenset erzeugen

            #TODO Datensätze aus log erzeugen und launcher vorbereiten



            x = state.astype(np.float32)
            u = action.astype(np.float32)
            r = reward.astype(np.float32)
            xn = next_state.astype(np.float32)

            obs = to_float_tensor(x, self.policy.use_cuda)
            act = to_float_tensor(u, self.policy.use_cuda)

            # update running mean and std if neccessary
            if self._trpo_standardizer is not None:
                self._trpo_standardizer.update_mean_std(x)

            # create reward
            if self._env_reward_frac < 1.0:

                # create reward from the discriminator(can use fraction of environment reward)
                r_disc = self.make_discrim_reward(x, u, xn)
                r = r * self._env_reward_frac + r_disc * (1 - self._env_reward_frac)

            v_target, np_adv = compute_gae(self._V, x, xn, r, absorbing, last,
                                           self.mdp_info.gamma, self._lambda())
            np_adv = (np_adv - np.mean(np_adv)) / (np.std(np_adv) + 1e-8)
            adv = to_float_tensor(np_adv, self.policy.use_cuda)

            # Policy update
            self._old_policy = deepcopy(self.policy)
            old_pol_dist = self._old_policy.distribution_t(obs)
            old_log_prob = self._old_policy.log_prob_t(obs, act).detach()

            zero_grad(self.policy.parameters())
            loss = self._compute_loss(obs, act, adv, old_log_prob)

            prev_loss = loss.item()

            # Compute Gradient
            loss.backward()
            g = get_gradient(self.policy.parameters())

            # Compute direction through conjugate gradient
            stepdir = self._conjugate_gradient(g, obs, old_pol_dist)

            # Line search
            self._line_search(obs, act, adv, old_log_prob, old_pol_dist, prev_loss, stepdir)

            # VF update
            if self._trpo_standardizer is not None:
                for i in range(self._critic_fit_params["n_epochs"]):
                    self._trpo_standardizer.update_mean_std(x)  # update running mean
            self._V.fit(x, v_target, **self._critic_fit_params)

            # fit discriminator
            self._fit_discriminator(x, u, xn)

            # Print fit information
            # create dataset with discriminator reward
            new_dataset = arrays_as_dataset(x, u, r, xn, absorbing, last)
            self._logging_sw(dataset, new_dataset, x, v_target, old_pol_dist)
            #self._log_info(dataset, x, v_target, old_pol_dist)
            self._iter += 1

"""





