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


class FullHumanoid(BaseHumanoid):
    """
    Mujoco simulation of full humanoid with muscle-actuated lower limb and torque-actuated upper body.

    """
    def __init__(self, **kwargs):
        """
        Constructor.

        """
        xml_path = (Path(__file__).resolve().parent.parent / "data" / "full_humanoid" / "full_humanoid.xml").as_posix()

        action_spec = [# motors
                       "lumbar_ext", "lumbar_bend", "lumbar_rot", "shoulder_flex_r", "shoulder_add_r", "shoulder_rot_r",
                       "elbow_flex_r", "pro_sup_r", "wrist_flex_r", "wrist_dev_r", "shoulder_flex_l", "shoulder_add_l",
                       "shoulder_rot_l", "elbow_flex_l", "pro_sup_l", "wrist_flex_l", "wrist_dev_l",
                       # muscles
                       "addbrev_r", "addlong_r", "addmagDist_r", "addmagIsch_r", "addmagMid_r", "addmagProx_r",
                       "bflh_r", "bfsh_r", "edl_r", "ehl_r", "fdl_r", "fhl_r", "gaslat_r", "gasmed_r", "glmax1_r",
                       "glmax2_r", "glmax3_r", "glmed1_r", "glmed2_r", "glmed3_r", "glmin1_r", "glmin2_r",
                       "glmin3_r", "grac_r", "iliacus_r", "perbrev_r", "perlong_r", "piri_r", "psoas_r", "recfem_r",
                       "sart_r", "semimem_r", "semiten_r", "soleus_r", "tfl_r", "tibant_r", "tibpost_r", "vasint_r",
                       "vaslat_r", "vasmed_r", "addbrev_l", "addlong_l", "addmagDist_l", "addmagIsch_l", "addmagMid_l",
                       "addmagProx_l", "bflh_l", "bfsh_l", "edl_l", "ehl_l", "fdl_l", "fhl_l", "gaslat_l", "gasmed_l",
                       "glmax1_l", "glmax2_l", "glmax3_l", "glmed1_l", "glmed2_l", "glmed3_l", "glmin1_l", "glmin2_l",
                       "glmin3_l", "grac_l", "iliacus_l", "perbrev_l", "perlong_l", "piri_l", "psoas_l", "recfem_l",
                       "sart_l", "semimem_l", "semiten_l", "soleus_l", "tfl_l", "tibant_l", "tibpost_l", "vasint_l",
                       "vaslat_l", "vasmed_l"]

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
                            ("q_knee_angle_r_translation2", "knee_angle_r_translation2", ObservationType.JOINT_POS),
                            ("q_knee_angle_r_translation1", "knee_angle_r_translation1", ObservationType.JOINT_POS),
                            ("q_knee_angle_r", "knee_angle_r", ObservationType.JOINT_POS),
                            ("q_knee_angle_r_rotation2", "knee_angle_r_rotation2", ObservationType.JOINT_POS),
                            ("q_knee_angle_r_rotation3", "knee_angle_r_rotation3", ObservationType.JOINT_POS),
                            ("q_ankle_angle_r", "ankle_angle_r", ObservationType.JOINT_POS),
                            ("q_subtalar_angle_r", "subtalar_angle_r", ObservationType.JOINT_POS),
                            ("q_mtp_angle_r", "mtp_angle_r", ObservationType.JOINT_POS),
                            ("q_knee_angle_r_beta_translation2", "knee_angle_r_beta_translation2", ObservationType.JOINT_POS),
                            ("q_knee_angle_r_beta_translation1", "knee_angle_r_beta_translation1", ObservationType.JOINT_POS),
                            ("q_knee_angle_r_beta_rotation1", "knee_angle_r_beta_rotation1", ObservationType.JOINT_POS),
                            # --- lower limb left ---
                            ("q_hip_flexion_l", "hip_flexion_l", ObservationType.JOINT_POS),
                            ("q_hip_adduction_l", "hip_adduction_l", ObservationType.JOINT_POS),
                            ("q_hip_rotation_l", "hip_rotation_l", ObservationType.JOINT_POS),
                            ("q_knee_angle_l_translation2", "knee_angle_l_translation2", ObservationType.JOINT_POS),
                            ("q_knee_angle_l_translation1", "knee_angle_l_translation1", ObservationType.JOINT_POS),
                            ("q_knee_angle_l", "knee_angle_l", ObservationType.JOINT_POS),
                            ("q_knee_angle_l_rotation2", "knee_angle_l_rotation2", ObservationType.JOINT_POS),
                            ("q_knee_angle_l_rotation3", "knee_angle_l_rotation3", ObservationType.JOINT_POS),
                            ("q_ankle_angle_l", "ankle_angle_l", ObservationType.JOINT_POS),
                            ("q_subtalar_angle_l", "subtalar_angle_l", ObservationType.JOINT_POS),
                            ("q_mtp_angle_l", "mtp_angle_l", ObservationType.JOINT_POS),
                            ("q_knee_angle_l_beta_translation2", "knee_angle_l_beta_translation2", ObservationType.JOINT_POS),
                            ("q_knee_angle_l_beta_translation1", "knee_angle_l_beta_translation1", ObservationType.JOINT_POS),
                            ("q_knee_angle_l_beta_rotation1", "knee_angle_l_beta_rotation1", ObservationType.JOINT_POS),
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
                            ("dq_knee_angle_r_translation2", "knee_angle_r_translation2", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_r_translation1", "knee_angle_r_translation1", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_r", "knee_angle_r", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_r_rotation2", "knee_angle_r_rotation2", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_r_rotation3", "knee_angle_r_rotation3", ObservationType.JOINT_VEL),
                            ("dq_ankle_angle_r", "ankle_angle_r", ObservationType.JOINT_VEL),
                            ("dq_subtalar_angle_r", "subtalar_angle_r", ObservationType.JOINT_VEL),
                            ("dq_mtp_angle_r", "mtp_angle_r", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_r_beta_translation2", "knee_angle_r_beta_translation2", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_r_beta_translation1", "knee_angle_r_beta_translation1", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_r_beta_rotation1", "knee_angle_r_beta_rotation1", ObservationType.JOINT_VEL),
                            # --- lower limb left ---
                            ("dq_hip_flexion_l", "hip_flexion_l", ObservationType.JOINT_VEL),
                            ("dq_hip_adduction_l", "hip_adduction_l", ObservationType.JOINT_VEL),
                            ("dq_hip_rotation_l", "hip_rotation_l", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_l_translation2", "knee_angle_l_translation2", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_l_translation1", "knee_angle_l_translation1", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_l", "knee_angle_l", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_l_rotation2", "knee_angle_l_rotation2", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_l_rotation3", "knee_angle_l_rotation3", ObservationType.JOINT_VEL),
                            ("dq_ankle_angle_l", "ankle_angle_l", ObservationType.JOINT_VEL),
                            ("dq_subtalar_angle_l", "subtalar_angle_l", ObservationType.JOINT_VEL),
                            ("dq_mtp_angle_l", "mtp_angle_l", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_l_beta_translation2", "knee_angle_l_beta_translation2", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_l_beta_translation1", "knee_angle_l_beta_translation1", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_l_beta_rotation1", "knee_angle_l_beta_rotation1", ObservationType.JOINT_VEL),
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
                            ("foot_r", ["r_foot"]),
                            ("front_foot_r", ["r_bofoot"]),
                            ("foot_l", ["l_foot"]),
                            ("front_foot_l", ["l_bofoot"])]

        super().__init__(xml_path, action_spec, observation_spec, collision_groups, **kwargs)

    @staticmethod
    def has_fallen(state):
        pelvis_euler = state[1:4]
        pelvis_condition = ((state[0] < -0.35) or (state[0] > 0.10)
                            or (pelvis_euler[0] < (-np.pi / 4.5)) or (pelvis_euler[0] > (np.pi / 12))
                            or (pelvis_euler[1] < -np.pi / 12) or (pelvis_euler[1] > np.pi / 8)
                            or (pelvis_euler[2] < (-np.pi / 10)) or (pelvis_euler[2] > (np.pi / 10))
                           )
        lumbar_euler = state[32:35]
        lumbar_condition = ((lumbar_euler[0] < (-np.pi / 6)) or (lumbar_euler[0] > (np.pi / 10))
                            or (lumbar_euler[1] < -np.pi / 10) or (lumbar_euler[1] > np.pi / 10)
                            or (lumbar_euler[2] < (-np.pi / 4.5)) or (lumbar_euler[2] > (np.pi / 4.5))
                            )
        return pelvis_condition or lumbar_condition


if __name__ == '__main__':
    import time

    env = FullHumanoid()

    action_dim = env.info.action_space.shape[0]

    print("Dimensionality of Obs-space:", env.info.observation_space.shape[0])
    print("Dimensionality of Act-space:", env.info.action_space.shape[0])

    env.reset()
    env.render()

    absorbing = False
    while True:

        action = np.random.randn(action_dim)
        nstate, _, absorbing, _ = env.step(action)

        env.render()
