from pathlib import Path

from mushroom_rl.environments.mujoco_envs.humanoids.base_humanoid import BaseHumanoid
from mushroom_rl.utils.running_stats import *
from mushroom_rl.utils.mujoco import *

# optional imports
try:
    mujoco_viewer_available = True
    import mujoco_viewer
except ModuleNotFoundError:
    mujoco_viewer_available = False


class ReducedHumanoidTorque(BaseHumanoid):
    """
    Mujoco simulation of simplified humanoid model with torque actuation.

    """
    def __init__(self, **kwargs):
        """
        Constructor.

        """
        xml_path = (Path(__file__).resolve().parent.parent / "data" / "reduced_humanoid_torque" /
                    "reduced_humanoid_torque.xml").as_posix()

        action_spec = [# motors
                       "mot_lumbar_ext", "mot_lumbar_bend", "mot_lumbar_rot", "mot_shoulder_flex_r",
                       "mot_shoulder_add_r", "mot_shoulder_rot_r", "mot_elbow_flex_r", "mot_pro_sup_r",
                       "mot_wrist_flex_r", "mot_wrist_dev_r", "mot_shoulder_flex_l", "mot_shoulder_add_l",
                       "mot_shoulder_rot_l", "mot_elbow_flex_l", "mot_pro_sup_l", "mot_wrist_flex_l",
                       "mot_wrist_dev_l", "mot_hip_flexion_r", "mot_hip_adduction_r", "mot_hip_rotation_r",
                       "mot_knee_angle_r", "mot_ankle_angle_r", "mot_subtalar_angle_r", "mot_mtp_angle_r",
                       "mot_hip_flexion_l", "mot_hip_adduction_l", "mot_hip_rotation_l", "mot_knee_angle_l",
                       "mot_ankle_angle_l", "mot_subtalar_angle_l", "mot_mtp_angle_l"]

        observation_spec = [#------------- JOINT POS -------------
                            ("q_pelvis_tx", "pelvis_tx", ObservationType.JOINT_POS),
                            ("q_pelvis_tz", "pelvis_tz", ObservationType.JOINT_POS),
                            ("q_pelvis_ty", "pelvis_ty", ObservationType.JOINT_POS),
                            ("q_pelvis_tilt", "pelvis_tilt", ObservationType.JOINT_POS),
                            ("q_pelvis_list", "pelvis_list", ObservationType.JOINT_POS),
                            ("q_pelvis_rotation", "pelvis_rotation", ObservationType.JOINT_POS),
                            # --- lower limb right ---
                            ("q_hip_flexion_r", "hip_flexion_r", ObservationType.JOINT_POS),
                            ("q_hip_adduction_r", "hip_adduction_r", ObservationType.JOINT_POS),
                            ("q_hip_rotation_r", "hip_rotation_r", ObservationType.JOINT_POS),
                            ("q_knee_angle_r", "knee_angle_r", ObservationType.JOINT_POS),
                            ("q_ankle_angle_r", "ankle_angle_r", ObservationType.JOINT_POS),
                            ("q_subtalar_angle_r", "subtalar_angle_r", ObservationType.JOINT_POS),
                            ("q_mtp_angle_r", "mtp_angle_r", ObservationType.JOINT_POS),
                            # --- lower limb left ---
                            ("q_hip_flexion_l", "hip_flexion_l", ObservationType.JOINT_POS),
                            ("q_hip_adduction_l", "hip_adduction_l", ObservationType.JOINT_POS),
                            ("q_hip_rotation_l", "hip_rotation_l", ObservationType.JOINT_POS),
                            ("q_knee_angle_l", "knee_angle_l", ObservationType.JOINT_POS),
                            ("q_ankle_angle_l", "ankle_angle_l", ObservationType.JOINT_POS),
                            ("q_subtalar_angle_l", "subtalar_angle_l", ObservationType.JOINT_POS),
                            ("q_mtp_angle_l", "mtp_angle_l", ObservationType.JOINT_POS),
                            # --- lumbar ---
                            ("q_lumbar_extension", "lumbar_extension", ObservationType.JOINT_POS),
                            ("q_lumbar_bending", "lumbar_bending", ObservationType.JOINT_POS),
                            ("q_lumbar_rotation", "lumbar_rotation", ObservationType.JOINT_POS),
                            # q-- upper body right ---
                            ("q_arm_flex_r", "arm_flex_r", ObservationType.JOINT_POS),
                            ("q_arm_add_r", "arm_add_r", ObservationType.JOINT_POS),
                            ("q_arm_rot_r", "arm_rot_r", ObservationType.JOINT_POS),
                            ("q_elbow_flex_r", "elbow_flex_r", ObservationType.JOINT_POS),
                            ("q_pro_sup_r", "pro_sup_r", ObservationType.JOINT_POS),
                            ("q_wrist_flex_r", "wrist_flex_r", ObservationType.JOINT_POS),
                            ("q_wrist_dev_r", "wrist_dev_r", ObservationType.JOINT_POS),
                            # --- upper body left ---
                            ("q_arm_flex_l", "arm_flex_l", ObservationType.JOINT_POS),
                            ("q_arm_add_l", "arm_add_l", ObservationType.JOINT_POS),
                            ("q_arm_rot_l", "arm_rot_l", ObservationType.JOINT_POS),
                            ("q_elbow_flex_l", "elbow_flex_l", ObservationType.JOINT_POS),
                            ("q_pro_sup_l", "pro_sup_l", ObservationType.JOINT_POS),
                            ("q_wrist_flex_l", "wrist_flex_l", ObservationType.JOINT_POS),
                            ("q_wrist_dev_l", "wrist_dev_l", ObservationType.JOINT_POS),

                            # ------------- JOINT VEL -------------
                            ("dq_pelvis_tx", "pelvis_tx", ObservationType.JOINT_VEL),
                            ("dq_pelvis_tz", "pelvis_tz", ObservationType.JOINT_VEL),
                            ("dq_pelvis_ty", "pelvis_ty", ObservationType.JOINT_VEL),
                            ("dq_pelvis_tilt", "pelvis_tilt", ObservationType.JOINT_VEL),
                            ("dq_pelvis_list", "pelvis_list", ObservationType.JOINT_VEL),
                            ("dq_pelvis_rotation", "pelvis_rotation", ObservationType.JOINT_VEL),
                            # --- lower limb right ---
                            ("dq_hip_flexion_r", "hip_flexion_r", ObservationType.JOINT_VEL),
                            ("dq_hip_adduction_r", "hip_adduction_r", ObservationType.JOINT_VEL),
                            ("dq_hip_rotation_r", "hip_rotation_r", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_r", "knee_angle_r", ObservationType.JOINT_VEL),
                            ("dq_ankle_angle_r", "ankle_angle_r", ObservationType.JOINT_VEL),
                            ("dq_subtalar_angle_r", "subtalar_angle_r", ObservationType.JOINT_VEL),
                            ("dq_mtp_angle_r", "mtp_angle_r", ObservationType.JOINT_VEL),
                            # --- lower limb left ---
                            ("dq_hip_flexion_l", "hip_flexion_l", ObservationType.JOINT_VEL),
                            ("dq_hip_adduction_l", "hip_adduction_l", ObservationType.JOINT_VEL),
                            ("dq_hip_rotation_l", "hip_rotation_l", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_l", "knee_angle_l", ObservationType.JOINT_VEL),
                            ("dq_ankle_angle_l", "ankle_angle_l", ObservationType.JOINT_VEL),
                            ("dq_subtalar_angle_l", "subtalar_angle_l", ObservationType.JOINT_VEL),
                            ("dq_mtp_angle_l", "mtp_angle_l", ObservationType.JOINT_VEL),
                            # --- lumbar ---
                            ("dq_lumbar_extension", "lumbar_extension", ObservationType.JOINT_VEL),
                            ("dq_lumbar_bending", "lumbar_bending", ObservationType.JOINT_VEL),
                            ("dq_lumbar_rotation", "lumbar_rotation", ObservationType.JOINT_VEL),
                            # --- upper body right ---
                            ("dq_arm_flex_r", "arm_flex_r", ObservationType.JOINT_VEL),
                            ("dq_arm_add_r", "arm_add_r", ObservationType.JOINT_VEL),
                            ("dq_arm_rot_r", "arm_rot_r", ObservationType.JOINT_VEL),
                            ("dq_elbow_flex_r", "elbow_flex_r", ObservationType.JOINT_VEL),
                            ("dq_pro_sup_r", "pro_sup_r", ObservationType.JOINT_VEL),
                            ("dq_wrist_flex_r", "wrist_flex_r", ObservationType.JOINT_VEL),
                            ("dq_wrist_dev_r", "wrist_dev_r", ObservationType.JOINT_VEL),
                            # --- upper body left ---
                            ("dq_arm_flex_l", "arm_flex_l", ObservationType.JOINT_VEL),
                            ("dq_arm_add_l", "arm_add_l", ObservationType.JOINT_VEL),
                            ("dq_arm_rot_l", "arm_rot_l", ObservationType.JOINT_VEL),
                            ("dq_elbow_flex_l", "elbow_flex_l", ObservationType.JOINT_VEL),
                            ("dq_pro_sup_l", "pro_sup_l", ObservationType.JOINT_VEL),
                            ("dq_wrist_flex_l", "wrist_flex_l", ObservationType.JOINT_VEL),
                            ("dq_wrist_dev_l", "wrist_dev_l", ObservationType.JOINT_VEL)]

        collision_groups = [("floor", ["floor"]),
                            ("foot_r", ["foot"]),
                            ("front_foot_r", ["bofoot"]),
                            ("foot_l", ["l_foot"]),
                            ("front_foot_l", ["l_bofoot"])]

        super().__init__(xml_path, action_spec, observation_spec, collision_groups, **kwargs)

    @staticmethod
    def has_fallen(state):
        pelvis_euler = state[1:4]
        pelvis_condition = ((state[0] < -0.46) or (state[0] > 0.0)
                            or (pelvis_euler[0] < (-np.pi / 4.5)) or (pelvis_euler[0] > (np.pi / 12))
                            or (pelvis_euler[1] < -np.pi / 12) or (pelvis_euler[1] > np.pi / 8)
                            or (pelvis_euler[2] < (-np.pi / 10)) or (pelvis_euler[2] > (np.pi / 10))
                           )
        lumbar_euler = state[18:21]
        lumbar_condition = ((lumbar_euler[0] < (-np.pi / 6)) or (lumbar_euler[0] > (np.pi / 10))
                            or (lumbar_euler[1] < -np.pi / 10) or (lumbar_euler[1] > np.pi / 10)
                            or (lumbar_euler[2] < (-np.pi / 4.5)) or (lumbar_euler[2] > (np.pi / 4.5))
                            )
        return pelvis_condition or lumbar_condition


if __name__ == '__main__':

    env = ReducedHumanoidTorque(timestep=1/1000, n_substeps=10)

    action_dim = env.info.action_space.shape[0]

    print("Dimensionality of Obs-space:", env.info.observation_space.shape[0])
    print("Dimensionality of Act-space:", env.info.action_space.shape[0])

    env.reset()
    env.render()

    absorbing = False
    i = 0
    while True:
        if i == 1000:
            env.reset()
            i = 0
        action = np.random.randn(action_dim)
        nstate, _, absorbing, _ = env.step(action)

        env.render()
        i += 1