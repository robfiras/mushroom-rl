

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
    """
    def __init__(self, gamma=0.99, horizon=1000, n_substeps=10,
                 traj_params=None, timestep=0.001):
        """
        Constructor.
        """
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
                         timestep=timestep, collision_groups=collision_groups, traj_params=traj_params)


    @staticmethod
    def has_fallen(state):
        """
        # with freejoint
        trunk_euler = quat_to_euler(state[3:7])
        # 0: rollen
        # 1: wiehern
        # 2: lenken
        trunk_condition = ((trunk_euler[0] < -np.pi * 40 / 180) or (trunk_euler[0] > np.pi * 40 / 180)
                           or (trunk_euler[1] < (-np.pi * 40 / 180)) or (trunk_euler[1] > (np.pi * 40 / 180))
                           )
        """

        # without freejoint
        trunk_euler = state[3:6]
        # 0: lenken/laufrichtung
        # 1: neigung x achse: schultern wackeln/rollen
        # 2: wiehern
        trunk_condition = ((trunk_euler[1] < -np.pi * 40 / 180) or (trunk_euler[1] > np.pi * 40 / 180)
                            or (trunk_euler[2] < (-np.pi * 40 / 180)) or (trunk_euler[2] > (np.pi * 40 / 180))
                            or state[2] < -.31
                            )

        return trunk_condition

@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start

if __name__ == '__main__':


    np.random.seed(1)
    # define env and data frequencies
    env_freq = 1000  # hz, added here as a reminder
    traj_data_freq = 500  # hz, added here as a reminder
    desired_contr_freq = 100  # hz
    n_substeps = env_freq // desired_contr_freq

    # prepare trajectory params
    traj_params = dict(traj_path='/home/tim/Documents/locomotion_simulation/log/states.npz',
                       traj_dt=(1 / traj_data_freq),
                       control_dt=(1 / desired_contr_freq))
    gamma = 0.99
    horizon = 1000

    env = UnitreeA1(timestep=1/500, gamma=gamma, horizon=horizon, n_substeps=n_substeps, traj_params=traj_params)


    with catchtime() as t:
        env.play_trajectory_demo(desired_contr_freq, view_from_other_side=True)
        print("Time: %fs" % t())

    print("Finished")

    #TODO: frequencies
    #TODO: error in trajectory class
    #TODO: changes offset in xml?? 0.2622972779494144 0.2622972779494144

    # removed xml from gitignore
    # still problem with different behavior -> des not for with mul joints

    """
    env = UnitreeA1(timestep=1/500,n_substeps=20)

    action_dim = env.info.action_space.shape[0]
    #print("get_obs", env._get_observation_space())
    print("Dimensionality of Obs-space:", env.info.observation_space.shape[0])
    print("Dimensionality of Act-space:", env.info.action_space.shape[0])




    env.reset()
    env.render()

    absorbing = False
    i = 0
    while True:
        if i == 1000: #or absorbing
            print("------ RESET ------")
            env.reset()
            i = 0
            absorbing = False
        #print("state", env._obs.copy())
        #print("obs_keys", env.get_all_observation_keys())
        #print("rotation z,x,y axis: ", env._obs.copy()[3:6]*180/np.pi) no freejoint

        #print("State: ", env._obs.copy()) #freejoint
        #print("Coord: ", env._obs.copy()[:3])
        #print("Rotation: ", env._obs.copy()[3:6]*180/np.pi)

        #action = np.random.randn(action_dim)
        action = np.zeros(action_dim)
        if i > 30: #test turn around rotation
            action[1] = 10
            action[7] = 10
        #action[1] = -1
        #action[7] = -1

        #action[4] = -.5
        #action[10] = -.5
        #action[5] = 1
        #action[11] = 1
        #action = [0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8] -> when clipping action space of -> default pos
        #print("Action", action)
        nstate, _, absorbing, _ = env.step(action)
        #print(absorbing)


        env.render()
        i += 1
        """









