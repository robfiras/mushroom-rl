from pathlib import Path

from mushroom_rl.environments.mujoco_envs.humanoids.base_humanoid import BaseHumanoid
from mushroom_rl.utils.angles import quat_to_euler
from mushroom_rl.utils.running_stats import *
from mushroom_rl.utils.mujoco import *


class Atlas(BaseHumanoid):
    """
    Mujoco simulation of the Atlas robot.

    """
    def __init__(self, **kwargs):
        """
        Constructor.

        """
        xml_path = (Path(__file__).resolve().parent.parent / "data" / "atlas" / "model.xml").as_posix()

        "hip_flexion_r_actuator",
        "hip_adduction_r_actuator",
        "hip_rotation_r_actuator",
        "knee_angle_r_actuator",
        "ankle_angle_r_actuator",

        "hip_flexion_l_actuator",
        "hip_adduction_l_actuator",
        "hip_rotation_l_actuator",
        "knee_angle_l_actuator",
        "ankle_angle_l_actuator",

        action_spec = ["hip_flexion_r_actuator","hip_adduction_r_actuator","hip_rotation_r_actuator",
                       "knee_angle_r_actuator","ankle_angle_r_actuator","hip_flexion_l_actuator",
                       "hip_adduction_l_actuator","hip_rotation_l_actuator","knee_angle_l_actuator",
                       "ankle_angle_l_actuator"]

        observation_spec = [# ------------- JOINT POS -------------
                            ("q_pelvis_tx", "pelvis_tx", ObservationType.JOINT_POS),
                            ("q_pelvis_tz", "pelvis_tz", ObservationType.JOINT_POS),
                            ("q_pelvis_ty", "pelvis_ty", ObservationType.JOINT_POS),
                            ("q_pelvis_tilt", "pelvis_tilt", ObservationType.JOINT_POS),
                            ("q_pelvis_list", "pelvis_list", ObservationType.JOINT_POS),
                            ("q_pelvis_rotation", "pelvis_rotation", ObservationType.JOINT_POS),
                            ("q_hip_flexion_r", "hip_flexion_r", ObservationType.JOINT_POS),
                            ("q_hip_adduction_r", "hip_adduction_r", ObservationType.JOINT_POS),
                            ("q_hip_rotation_r", "hip_rotation_r", ObservationType.JOINT_POS),
                            ("q_knee_angle_r", "knee_angle_r", ObservationType.JOINT_POS),
                            ("q_ankle_angle_r", "ankle_angle_r", ObservationType.JOINT_POS),
                            ("q_hip_flexion_l", "hip_flexion_l", ObservationType.JOINT_POS),
                            ("q_hip_adduction_l", "hip_adduction_l", ObservationType.JOINT_POS),
                            ("q_hip_rotation_l", "hip_rotation_l", ObservationType.JOINT_POS),
                            ("q_knee_angle_l", "knee_angle_l", ObservationType.JOINT_POS),
                            ("q_ankle_angle_l", "ankle_angle_l", ObservationType.JOINT_POS),

                            # ------------- JOINT VEL -------------
                            ("dq_pelvis_tx", "pelvis_tx", ObservationType.JOINT_VEL),
                            ("dq_pelvis_tz", "pelvis_tz", ObservationType.JOINT_VEL),
                            ("dq_pelvis_ty", "pelvis_ty", ObservationType.JOINT_VEL),
                            ("dq_pelvis_tilt", "pelvis_tilt", ObservationType.JOINT_VEL),
                            ("dq_pelvis_list", "pelvis_list", ObservationType.JOINT_VEL),
                            ("dq_pelvis_rotation", "pelvis_rotation", ObservationType.JOINT_VEL),
                            ("dq_hip_flexion_r", "hip_flexion_r", ObservationType.JOINT_VEL),
                            ("dq_hip_adduction_r", "hip_adduction_r", ObservationType.JOINT_VEL),
                            ("dq_hip_rotation_r", "hip_rotation_r", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_r", "knee_angle_r", ObservationType.JOINT_VEL),
                            ("dq_ankle_angle_r", "ankle_angle_r", ObservationType.JOINT_VEL),
                            ("dq_hip_flexion_l", "hip_flexion_l", ObservationType.JOINT_VEL),
                            ("dq_hip_adduction_l", "hip_adduction_l", ObservationType.JOINT_VEL),
                            ("dq_hip_rotation_l", "hip_rotation_l", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_l", "knee_angle_l", ObservationType.JOINT_VEL),
                            ("dq_ankle_angle_l", "ankle_angle_l", ObservationType.JOINT_VEL)]

        collision_groups = [("floor", ["ground"]),
                            ("foot_r", ["right_foot_back"]),
                            ("front_foot_r", ["right_foot_front"]),
                            ("foot_l", ["left_foot_back"]),
                            ("front_foot_l", ["left_foot_front"])]

        super().__init__(xml_path, action_spec, observation_spec, collision_groups, **kwargs)

    @staticmethod
    def has_fallen(state):
        torso_euler = quat_to_euler(state[1:5])
        return ((state[0] < 0.80) or (state[0] > 1.20)
                or abs(torso_euler[0]) > np.pi / 12
                or (torso_euler[1] < -np.pi / 12) or (torso_euler[1] > np.pi / 8)
                or (torso_euler[2] < (-np.pi / 14)+np.pi/2) or (torso_euler[2] > (np.pi / 14)+np.pi/2)
                )


if __name__ == '__main__':

    env = Atlas()

    action_dim = env.info.action_space.shape[0]

    print(env.info.observation_space.shape[0])

    env.reset()
    env.render()

    absorbing = False
    while True:
        print('test')
        action = np.random.randn(action_dim)/10
        nstate, _, absorbing, _ = env.step(action)

        env.render()
