

import time
from abc import abstractmethod
import mujoco

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

class UnitreeA1(BaseQuadruped):
    """
    Mujoco simulation of unitree A1 model
    to switch between torque and position control: adjust xml file (and if needed action_position.npz/action_position.npz)
    to switch between freejoint and mul_joint: adapt obs space and xml path
    """
    def __init__(self, gamma=0.99, horizon=1000, n_substeps=10, random_start=False, init_step_no=None,
                 traj_params=None, timestep=0.001, goal_reward=None, goal_reward_params=None, use_torque_ctrl=False):
        """
        Constructor.
        for clipping in torques need to adjust xml gear 34 and ctrllimited
        """
        if use_torque_ctrl:
            xml_path = (Path(__file__).resolve().parent.parent / "data" / "quadrupeds" /
                    "unitree_a1_torque_mul_joint.xml").as_posix()
        else:
            xml_path = (Path(__file__).resolve().parent.parent / "data" / "quadrupeds" /
                        "unitree_a1_position_mul_joint.xml").as_posix()
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
        super().__init__(xml_path, action_spec, observation_spec, gamma=gamma, horizon=horizon, n_substeps=n_substeps,
                         timestep=timestep, collision_groups=collision_groups, traj_params=traj_params, init_step_no=init_step_no,
                         goal_reward=goal_reward, goal_reward_params=goal_reward_params, random_start=random_start)



    def setup(self, substep_no=None):
        self.goal_reward.reset_state()
        if self.trajectory is not None:
            if self._random_start:
                sample = self.trajectory.reset_trajectory()
            else:
                sample = self.trajectory.reset_trajectory(self._init_step_no)

            self.set_qpos_qvel(sample)
        else: # TODO: add this fuctionality in base_humanoid for all env
            self._data.qpos = [0, 0, -0.16, 0, 0, 0, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8]
            self._data.qvel = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

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
        trunk_condition = ((trunk_euler[0] < -0.5) or (trunk_euler[0] > 0.02)
                            or (trunk_euler[1] < -0.095) or (trunk_euler[1] > 0.095)
                            or (trunk_euler[2] < -0.075) or (trunk_euler[2] > 0.075)
                            or state[0] < -.22 #.25
                            )"""

        #for cluster datasets
        trunk_condition = ((trunk_euler[1] < -0.6981) or (trunk_euler[1] > 0.6981)
                           or (trunk_euler[2] < -0.6981) or (trunk_euler[2] > 0.6981)
                           or state[0] < -.25
                           )
        #if trunk_condition:
        #    print("con1: ", (trunk_euler[0] < -0.5) or (trunk_euler[0] > 0.02), trunk_euler[0])
        #    print("con2: ", (trunk_euler[1] < -0.095) or (trunk_euler[1] > 0.095), trunk_euler[1])
        #    print("con3: ", (trunk_euler[2] < -0.075) or (trunk_euler[2] > 0.075), trunk_euler[2])
        #    print("con4: ", state[0] < -.22, state[0])
        #    print(state)
        return trunk_condition

@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start

if __name__ == '__main__':
    # TODO: different behavior, action control completed?, for clipping in torques need to adjust xml gear 34 and ctrllimited
    """
    #trajectory demo:
    np.random.seed(1)
    # define env and data frequencies
    env_freq = 1000  # hz, added here as a reminder
    traj_data_freq = 500  # hz, added here as a reminder
    desired_contr_freq = 100  # hz
    n_substeps = env_freq // desired_contr_freq

    # prepare trajectory params
    traj_params = dict(traj_path='/home/tim/Documents/locomotion_simulation/log/states_50k_noise1.npz',
                       traj_dt=(1 / traj_data_freq),
                       control_dt=(1 / desired_contr_freq))
    gamma = 0.99
    horizon = 1000

    env = UnitreeA1(timestep=1/env_freq, gamma=gamma, horizon=horizon, n_substeps=n_substeps,
                    traj_params=traj_params, init_step_no=0)


    with catchtime() as t:
        env.play_trajectory_demo(desired_contr_freq, view_from_other_side=True)
        print("Time: %fs" % t())

    print("Finished")
    # still problem with different behaviour (if robot rolls to the side - between freejoint and muljoints) action[1] and [7] = -1 (with action clipping)
    """

    #solref="0.004 1000" /damping 500, stiffness from 0,93 to 62,5
    #0.004 1000000
    #0.004-0.005 1000000 kp=1000
    # favorite 0.005 1000000 | solref="-0.000001 -400"
    # final: solref="-0.0000000001 -250"




    # action demo - need action clipping to be off
    env_freq = 1000  # hz, added here as a reminder simulation freq
    traj_data_freq = 500  # hz, added here as a reminder  controll_freq of data model -> sim_freq/n_substeps
    desired_contr_freq = 500  # hz contl freq.
    n_substeps =  env_freq // desired_contr_freq
    # TODO: unstable so that it falls if des_contr_freq!= data_freq
    #to interpolate
    demo_dt = (1 / traj_data_freq)
    control_dt = (1 / desired_contr_freq)


    gamma = 0.99
    horizon = 1000



    env = UnitreeA1(timestep=1/env_freq, gamma=gamma, horizon=horizon, n_substeps=n_substeps, use_torque_ctrl=False)


    action_dim = env.info.action_space.shape[0]
    print("Dimensionality of Obs-space:", env.info.observation_space.shape[0])
    print("Dimensionality of Act-space:", env.info.action_space.shape[0])

    env.reset()


    env.play_action_demo(action_path='/home/tim/Documents/locomotion_simulation/log/actions_position_50k_noise4.npz', #actions_torque.npz
                         states_path='/home/tim/Documents/locomotion_simulation/log/states_50k_noise4.npz',
                         control_dt=control_dt, demo_dt=demo_dt)#,
                         #dataset_path='/home/tim/Documents/IRL_unitreeA1/data')


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


    gamma = 0.99
    horizon = 1000



    env = UnitreeA1(timestep=1/env_freq, gamma=gamma, horizon=horizon, n_substeps=n_substeps, use_torque_ctrl=True)
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
            print("------ RESET ------")
            env.reset()
            i = 0
            absorbing = False
        
        action = np.random.randn(action_dim)
        nstate, _, absorbing, _ = env.step(action)
        print(nstate)
        if(i%2):
            env.reset()
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





