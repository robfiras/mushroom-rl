import time
import mujoco_py

from mushroom_rl.environments.mujoco import MuJoCo, ObservationType
from pathlib import Path

from mushroom_rl.utils import spaces
from mushroom_rl.utils.angles import quat_to_euler
from mushroom_rl.utils.running_stats import *

from mushroom_rl.environments.mujoco_envs.humanoids.reward_goals import NoGoalReward, NoGoalRewardRandInit,\
    ChangingVelocityTargetReward, CustomReward


class FullHumanoid(MuJoCo):
    """
    Mujoco simulation of full humanoid with muscle-actuated lower limb and torque-actuated upper body.

    """
    def __init__(self, gamma=0.99, horizon=1000, n_substeps=10,  goal_reward=None, goal_reward_params=None):
        """
        Constructor.

        """
        xml_path = (Path(__file__).resolve().parent.parent.parent / "data" / "full_humanoid" / "full_humanoid.xml").as_posix()

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
                            ("d_pelvis_tx", "pelvis_tx", ObservationType.JOINT_POS),
                            ("d_pelvis_tz", "pelvis_tz", ObservationType.JOINT_POS),
                            ("d_pelvis_ty", "pelvis_ty", ObservationType.JOINT_POS),
                            ("d_pelvis_tilt", "pelvis_tilt", ObservationType.JOINT_POS),
                            ("d_pelvis_list", "pelvis_list", ObservationType.JOINT_POS),
                            ("d_pelvis_rotation", "pelvis_rotation", ObservationType.JOINT_POS),
                            # --- lower limb right ---
                            ("d_hip_flexion_r", "hip_flexion_r", ObservationType.JOINT_POS),
                            ("d_hip_adduction_r", "hip_adduction_r", ObservationType.JOINT_POS),
                            ("d_hip_rotation_r", "hip_rotation_r", ObservationType.JOINT_POS),
                            ("d_knee_angle_r_translation2", "knee_angle_r_translation2", ObservationType.JOINT_POS),
                            ("d_knee_angle_r_translation1", "knee_angle_r_translation1", ObservationType.JOINT_POS),
                            ("d_knee_angle_r", "knee_angle_r", ObservationType.JOINT_POS),
                            ("d_knee_angle_r_rotation2", "knee_angle_r_rotation2", ObservationType.JOINT_POS),
                            ("d_knee_angle_r_rotation3", "knee_angle_r_rotation3", ObservationType.JOINT_POS),
                            ("d_ankle_angle_r", "ankle_angle_r", ObservationType.JOINT_POS),
                            ("d_subtalar_angle_r", "subtalar_angle_r", ObservationType.JOINT_POS),
                            ("d_mtp_angle_r", "mtp_angle_r", ObservationType.JOINT_POS),
                            ("d_knee_angle_r_beta_translation2", "knee_angle_r_beta_translation2", ObservationType.JOINT_POS),
                            ("d_knee_angle_r_beta_translation1", "knee_angle_r_beta_translation1", ObservationType.JOINT_POS),
                            ("d_knee_angle_r_beta_rotation1", "knee_angle_r_beta_rotation1", ObservationType.JOINT_POS),
                            # --- lower limb left ---
                            ("d_hip_flexion_l", "hip_flexion_l", ObservationType.JOINT_POS),
                            ("d_hip_adduction_l", "hip_adduction_l", ObservationType.JOINT_POS),
                            ("d_hip_rotation_l", "hip_rotation_l", ObservationType.JOINT_POS),
                            ("d_knee_angle_l_translation2", "knee_angle_l_translation2", ObservationType.JOINT_POS),
                            ("d_knee_angle_l_translation1", "knee_angle_l_translation1", ObservationType.JOINT_POS),
                            ("d_knee_angle_l", "knee_angle_l", ObservationType.JOINT_POS),
                            ("d_knee_angle_l_rotation2", "knee_angle_l_rotation2", ObservationType.JOINT_POS),
                            ("d_knee_angle_l_rotation3", "knee_angle_l_rotation3", ObservationType.JOINT_POS),
                            ("d_ankle_angle_l", "ankle_angle_l", ObservationType.JOINT_POS),
                            ("d_subtalar_angle_l", "subtalar_angle_l", ObservationType.JOINT_POS),
                            ("d_mtp_angle_l", "mtp_angle_l", ObservationType.JOINT_POS),
                            ("d_knee_angle_l_beta_translation2", "knee_angle_l_beta_translation2", ObservationType.JOINT_POS),
                            ("d_knee_angle_l_beta_translation1", "knee_angle_l_beta_translation1", ObservationType.JOINT_POS),
                            ("d_knee_angle_l_beta_rotation1", "knee_angle_l_beta_rotation1", ObservationType.JOINT_POS),
                            #  --- lumbar ---
                            ("d_lumbar_extension", "lumbar_extension", ObservationType.JOINT_POS),
                            ("d_lumbar_bending", "lumbar_bending", ObservationType.JOINT_POS),
                            ("d_lumbar_rotation", "lumbar_rotation", ObservationType.JOINT_POS),
                            # --- upper body right ---
                            ("d_arm_flex_r", "arm_flex_r", ObservationType.JOINT_POS),
                            ("d_arm_add_r", "arm_add_r", ObservationType.JOINT_POS),
                            ("d_arm_rot_r", "arm_rot_r", ObservationType.JOINT_POS),
                            ("d_elbow_flex_r", "elbow_flex_r", ObservationType.JOINT_POS),
                            ("d_pro_sup_r", "pro_sup_r", ObservationType.JOINT_POS),
                            ("d_wrist_flex_r", "wrist_flex_r", ObservationType.JOINT_POS),
                            ("d_wrist_dev_r", "wrist_dev_r", ObservationType.JOINT_POS),
                            # --- upper body left ---
                            ("d_arm_flex_l", "arm_flex_l", ObservationType.JOINT_POS),
                            ("d_arm_add_l", "arm_add_l", ObservationType.JOINT_POS),
                            ("d_arm_rot_l", "arm_rot_l", ObservationType.JOINT_POS),
                            ("d_elbow_flex_l", "elbow_flex_l", ObservationType.JOINT_POS),
                            ("d_pro_sup_l", "pro_sup_l", ObservationType.JOINT_POS),
                            ("d_wrist_flex_l", "wrist_flex_l", ObservationType.JOINT_POS),
                            ("d_wrist_dev_l", "wrist_dev_l", ObservationType.JOINT_POS),

                            # ------------- JOINT VEL -------------
                            ("dq_pelvis_tz", "pelvis_tz", ObservationType.JOINT_VEL),
                            ("dq_pelvis_ty", "pelvis_ty", ObservationType.JOINT_VEL),
                            ("dq_pelvis_tx", "pelvis_tx", ObservationType.JOINT_VEL),
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

        # TODO: add collision groups
        # collision_groups = [("ground", ["ground"]),
        #                    ("right_foot_back", ["right_foot_back"]),
        #                    ("right_foot_front", ["right_foot_front"]),
        #                    ("left_foot_back", ["left_foot_back"]),
        #                    ("left_foot_front", ["left_foot_front"]),
        #                    ]

        super().__init__(xml_path, action_spec, observation_spec, gamma=gamma, horizon=horizon,
                         n_substeps=n_substeps, collision_groups=[])  # TODO: add collision groups here

        self._n_substeps = n_intermediate_steps

        # specify the reward
        if goal_reward == "changing_vel":
            self.goal_reward = ChangingVelocityTargetReward(self._sim, **goal_reward_params)
        elif goal_reward == "no_goal_rand_init":
            self.goal_reward = NoGoalRewardRandInit(self._sim, **goal_reward_params)
        elif goal_reward == "custom":
            self.goal_reward = CustomReward(sim=self._sim, **goal_reward_params)
        elif goal_reward is None:
            self.goal_reward = NoGoalReward()
        else:
            raise NotImplementedError("The specified goal reward has not been"
                                      "implemented: ", goal_reward)

        self.info.observation_space = spaces.Box(*self._get_observation_space())

        low, high = self.info.action_space.low.copy(),\
                    self.info.action_space.high.copy()
        self.norm_act_mean = (high + low) / 2.0
        self.norm_act_delta = (high - low) / 2.0
        self.info.action_space.low[:] = -1.0
        self.info.action_space.high[:] = 1.0

        # TODO: Maybe modify the observation space
        #self.info.observation_space.low[0] = 0      # pelvis height
        #self.info.observation_space.high[0] = 3
        #self.info.observation_space.low[1:5] = -1   # quaternions
        #self.info.observation_space.high[1:5] = 1
        #self.info.observation_space.low[15:18] = -10    # translational velocity pelvis
        #self.info.observation_space.high[15:18] = 10
        #self.info.observation_space.low[18:21] = -10    # rotational velocity pelvis
        #self.info.observation_space.high[18:21] = 10
        #self.info.observation_space.low[21:31] = -100   # rotational velocity joints
        #self.info.observation_space.high[21:31] = 100   # rotational velocity joints

    def _get_observation_space(self):
        sim_low, sim_high = (self.info.observation_space.low[2:],
                             self.info.observation_space.high[2:])

        # TODO: add ground forces here once collision groups are added
        #grf_low, grf_high = (-np.ones((12,)) * np.inf,
        #                     np.ones((12,)) * np.inf)

        r_low, r_high = self.goal_reward.get_observation_space()

        return (np.concatenate([sim_low, r_low]),   # TODO: add grf here as well
                np.concatenate([sim_high, r_high]))

    def _create_observation(self):
        """
        Creates full vector of observations:

        obs[0:15] -> qpos(from mujoco obs)
        obs[0] -> torso z pos
        obs[1:5] -> torso quaternion orientation
        obs[5:15] -> leg joints angle

        obs[15:31] -> qvel(from mujoco obs)
        obs[15:18] -> torso linear velocity
        obs[18:21] -> torso angular velocity
        obs[21:31] -> leg joints angular velocity

        obs[31:34] -> grf right foot back
        obs[34:37] -> grf right foot front
        obs[37:40] -> grf left foot back
        obs[40:43] -> grf left foot front

        obs[43:] -> reward observation (if available)

        """

        obs = np.concatenate([super()._create_observation()[2:],
                              # TODO: add grf forces here
                              #self._get_collision_force("ground", "right_foot_back")[:3]/10000.,
                              #self._get_collision_force("ground", "right_foot_front")[:3]/10000.,
                              #self._get_collision_force("ground", "left_foot_back")[:3]/10000.,
                              #self._get_collision_force("ground", "left_foot_front")[:3]/10000.,
                              self.goal_reward.get_observation(),
                              ]).flatten()

        return obs

    def _reward(self, state, action, next_state):
        goal_reward = self.goal_reward(state, action, next_state)
        return goal_reward

    def _setup(self):
        self.goal_reward.reset_state()

    def _preprocess_action(self, action):
        unnormalized_action = ((action.copy() * self.norm_act_delta) + self.norm_act_mean)
        return unnormalized_action

    def _is_absorbing(self, state):
        return self._has_fallen(state)

    def _set_state(self, qpos, qvel):
        old_state = self._sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self._sim.set_state(new_state)
        self._sim.forward()

    @staticmethod
    def _has_fallen(state):
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


    def render(self):
        if self._viewer is None:
            self._viewer = mujoco_py.MjViewer(self._sim)

        self._viewer.render()
        time.sleep(self._dt*self._n_substeps)

    def _reset_model(self, qpos_noise=0.0, qvel_noise=0.0):
        self._set_state(self._sim.data.qpos + np.random.uniform(
            low=-qpos_noise, high=qpos_noise, size=self._sim.model.nq),
                        self._sim.data.qvel + np.random.uniform(low=-qvel_noise,
                                                                high=qvel_noise,
                                                                size=self._sim.model.nv)
                        )

        return self._create_observation()

    def _set_state(self, qpos, qvel):
        old_state = self._sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self._sim.set_state(new_state)
        self._sim.forward()



if __name__ == '__main__':
    import time

    env = FullHumanoid()

    action_dim = env.info.action_space.shape[0]

    print(env.info.observation_space.shape[0])

    env.reset()
    env.render()

    state = env._sim.get_state()
    state.qpos[2] = 2
    env._set_state(state.qpos, state.qvel)

    absorbing = False
    while True:
        print('test')
        action = np.random.randn(action_dim)
        action = np.ones_like(action)
        nstate, _, absorbing, _ = env.step(action)

        env.render()
