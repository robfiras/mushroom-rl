import os.path
import time
from pathlib import Path
from tempfile import mkdtemp
from copy import deepcopy

import numpy as np
from dm_control import mjcf

from mushroom_rl.environments.mujoco_envs.humanoids.base_humanoid import BaseHumanoid
from mushroom_rl.utils.running_stats import *
from mushroom_rl.utils.mujoco import *

# optional imports
try:
    mujoco_viewer_available = True
    import mujoco_viewer
except ModuleNotFoundError:
    mujoco_viewer_available = False


class ReducedHumanoidTorquePOMDP(BaseHumanoid):
    """
    Mujoco simulation of simplified humanoid model with torque actuation.

    """
    def __init__(self, scaling=None, use_brick_foots=False, disable_arms=False, tmp_dir_name=None, **kwargs):
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

        # 0.4 ~ 1 year old baby which just started walking
        # 0.6 ~ 5 year old boy
        # 0.8 ~ 12 year old boy
        # 1.0 ~ 20 year old man
        allowed_scalings = [0.4, 0.6, 0.8, 1.0]
        if scaling is None:
            self._scalings = allowed_scalings
        else:
            assert scaling in allowed_scalings
            self._scalings = [scaling]

        self._use_brick_foots = use_brick_foots
        self._disable_arms = disable_arms
        joints_to_remove = []
        motors_to_remove = []
        equ_constr_to_remove = []
        if use_brick_foots:
            joints_to_remove +=["subtalar_angle_l", "mtp_angle_l", "subtalar_angle_r", "mtp_angle_r"]
            motors_to_remove += ["mot_subtalar_angle_l", "mot_mtp_angle_l", "mot_subtalar_angle_r", "mot_mtp_angle_r"]
            equ_constr_to_remove += [j + "_constraint" for j in joints_to_remove]
            # ToDo: think about a smarter way to not include foot force twice for bricks
            collision_groups = [("floor", ["floor"]),
                                ("foot_r", ["foot_brick_r"]),
                                ("front_foot_r", ["foot_brick_r"]),
                                ("foot_l", ["foot_brick_l"]),
                                ("front_foot_l", ["foot_brick_l"])]

        if disable_arms:
            joints_to_remove +=["arm_flex_r", "arm_add_r", "arm_rot_r", "elbow_flex_r", "pro_sup_r", "wrist_flex_r",
                                "wrist_dev_r", "arm_flex_l", "arm_add_l", "arm_rot_l", "elbow_flex_l", "pro_sup_l",
                                "wrist_flex_l", "wrist_dev_l"]
            motors_to_remove += ["mot_shoulder_flex_r", "mot_shoulder_add_r", "mot_shoulder_rot_r", "mot_elbow_flex_r",
                                 "mot_pro_sup_r", "mot_wrist_flex_r", "mot_wrist_dev_r", "mot_shoulder_flex_l",
                                 "mot_shoulder_add_l", "mot_shoulder_rot_l", "mot_elbow_flex_l", "mot_pro_sup_l",
                                 "mot_wrist_flex_l", "mot_wrist_dev_l"]
            equ_constr_to_remove += ["wrist_flex_r_constraint", "wrist_dev_r_constraint",
                                    "wrist_flex_l_constraint", "wrist_dev_l_constraint"]

        xml_handle = mjcf.from_path(xml_path)
        xml_handles = [self.scale_body(deepcopy(xml_handle), scaling) for scaling in self._scalings]

        if use_brick_foots or disable_arms:
            obs_to_remove = ["q_" + j for j in joints_to_remove] + ["dq_" + j for j in joints_to_remove]
            observation_spec = [elem for elem in observation_spec if elem[0] not in obs_to_remove]
            action_spec = [ac for ac in action_spec if ac not in motors_to_remove]

            for handle, scale in zip(xml_handles, self._scalings):
                handle = self.delete_from_xml_handle(handle, joints_to_remove,
                                                         motors_to_remove, equ_constr_to_remove)

                if use_brick_foots:
                    handle = self.add_brick_foots_to_xml_handle(handle, scale)

        xml_paths = [self.save_xml_handle(handle, tmp_dir_name) for handle in xml_handles]

        super().__init__(xml_paths, action_spec, observation_spec, collision_groups, **kwargs)

    def delete_from_xml_handle(self, xml_handle, joints_to_remove, motors_to_remove, equ_constraints):

        for j in joints_to_remove:
            j_handle = xml_handle.find("joint", j)
            j_handle.remove()
        for m in motors_to_remove:
            m_handle = xml_handle.find("actuator", m)
            m_handle.remove()
        for e in equ_constraints:
            e_handle = xml_handle.find("equality", e)
            e_handle.remove()

        return xml_handle

    def scale_body(self, xml_handle, scaling):
        body_scaling = scaling
        mesh_handle = xml_handle.find_all("mesh")

        head_geoms = ["hat_skull", "hat_jaw", "hat_ribs_cap"]

        for h in mesh_handle:
            if h.name not in head_geoms: # don't scale head
                h.scale *= body_scaling

        for h in xml_handle.find_all("geom"):
            if h.name in head_geoms: # change position of head
                h.pos = [0.0, -0.5*(1 - scaling), 0.0]

        body_handle = xml_handle.find_all("body")
        for h in body_handle:
            h.pos *= body_scaling
            h.inertial.mass *= body_scaling**3
            # Diagonal elements of the inertia matrix change quintically with scaling.
            # As all off-diagonal elements are 0 here.
            h.inertial.fullinertia *= body_scaling**5
            assert np.array_equal(h.inertial.fullinertia[3:], np.zeros(3)), "Some of the diagonal elements of the" \
                                                                            "inertia matrix are not zero! Scaling is" \
                                                                            "not done correctly. Double-Check!"
        actuator_handle = xml_handle.find_all("actuator")
        for h in actuator_handle:
            h.gear *= body_scaling**2

        return xml_handle

    def add_brick_foots_to_xml_handle(self, xml_handle, scaling):

        # find foot and attach bricks
        toe_l = xml_handle.find("body", "toes_l")
        size = np.array([0.112, 0.03, 0.05]) * scaling
        pos = np.array([-0.09, 0.019, 0.0]) * scaling
        toe_l.add("geom", name="foot_brick_l", type="box", size=size.tolist(), pos=pos.tolist(),
                  rgba=[0.5, 0.5, 0.5, 0.5], euler=[0.0, 0.15, 0.0])
        toe_r = xml_handle.find("body", "toes_r")
        toe_r.add("geom", name="foot_brick_r", type="box", size=size.tolist(), pos=pos.tolist(),
                  rgba=[0.5, 0.5, 0.5, 0.5], euler=[0.0, -0.15, 0.0])

        # make true foot uncollidable
        foot_r = xml_handle.find("geom", "foot")
        bofoot_r = xml_handle.find("geom", "bofoot")
        foot_l = xml_handle.find("geom", "l_foot")
        bofoot_l = xml_handle.find("geom", "l_bofoot")
        foot_r.contype = 0
        foot_r.conaffinity = 0
        bofoot_r.contype = 0
        bofoot_r.conaffinity = 0
        foot_l.contype = 0
        foot_l.conaffinity = 0
        bofoot_l.contype = 0
        bofoot_l.conaffinity = 0

        return xml_handle

    def save_xml_handle(self, xml_handle, tmp_dir_name):

        if tmp_dir_name is not None:
            assert os.path.exists(tmp_dir_name), "specified directory (\"%s\") does not exist." % tmp_dir_name

        dir = mkdtemp(dir=tmp_dir_name)
        file_name = "humanoid.xml"
        file_path = os.path.join(dir, file_name)

        # dump data
        mjcf.export_with_assets(xml_handle, dir, file_name)

        return file_path

    def render(self):
        super(BaseHumanoid, self).render()

    def has_fallen(self, state):
        pelvis_euler = state[1:4]
        pelvis_condition = ((pelvis_euler[0] < (-np.pi / 4.5)) or (pelvis_euler[0] > (np.pi / 12))
                            or (pelvis_euler[1] < -np.pi / 12) or (pelvis_euler[1] > np.pi / 8)
                            or (pelvis_euler[2] < (-np.pi / 10)) or (pelvis_euler[2] > (np.pi / 10))
                           )

        lumbar_euler = state[18:21]
        if self._use_brick_foots:
            lumbar_euler = state[14:17]
        else:
            lumbar_euler = state[18:21]

        lumbar_condition = ((lumbar_euler[0] < (-np.pi / 6)) or (lumbar_euler[0] > (np.pi / 10))
                            or (lumbar_euler[1] < -np.pi / 10) or (lumbar_euler[1] > np.pi / 10)
                            or (lumbar_euler[2] < (-np.pi / 4.5)) or (lumbar_euler[2] > (np.pi / 4.5))
                            )
        return (pelvis_condition or lumbar_condition) and False


if __name__ == '__main__':

    env = ReducedHumanoidTorquePOMDP(scaling=0.4, timestep=1/1000, n_substeps=10, use_brick_foots=True, random_start=False,
                                     disable_arms=True, tmp_dir_name="/home/moore/Downloads/teso")

    action_dim = env.info.action_space.shape[0]

    print("Dimensionality of Obs-space:", env.info.observation_space.shape[0])
    print("Dimensionality of Act-space:", env.info.action_space.shape[0])

    env.reset()
    env.render()
    absorbing = False
    i = 0
    while True:
        if i == 200 or absorbing:
            env.reset()
            env.render()
            i = 0
        action = np.random.randn(action_dim) * 0.01
        nstate, _, absorbing, _ = env.step(action)

        env.render()
        i += 1