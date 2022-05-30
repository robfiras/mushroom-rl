import time
import mujoco_py

from mushroom_rl.environments.mujoco import MuJoCo, ObservationType
from pathlib import Path

from mushroom_rl.utils import spaces
from mushroom_rl.utils.angles import quat_to_euler
from mushroom_rl.utils.running_stats import *

from mushroom_rl.environments.mujoco_envs.humanoids.reward_goals import NoGoalReward, NoGoalRewardRandInit,\
    ChangingVelocityTargetReward, CustomReward


class Atlas(MuJoCo):
    """
    Mujoco simulation of the Atlas robot.

    """
    def __init__(self, gamma=0.99, horizon=1000, n_intermediate_steps=5,  goal_reward=None, goal_reward_params=None):
        """
        Constructor.

        """
        xml_path = (Path(__file__).resolve().parent.parent.parent / "data" / "atlas" / "model.xml").as_posix()

        action_spec = ["r_leg_hpx_actuator", "r_leg_hpy_actuator", "r_leg_hpz_actuator", "r_leg_kny_actuator",
                       "r_leg_aky_actuator", "l_leg_hpx_actuator", "l_leg_hpy_actuator", "l_leg_hpz_actuator",
                       "l_leg_kny_actuator", "l_leg_aky_actuator"]

        observation_spec = [("root", ObservationType.JOINT_POS),
                            ("r_leg_hpx", ObservationType.JOINT_POS),
                            ("r_leg_hpy", ObservationType.JOINT_POS),
                            ("r_leg_hpz", ObservationType.JOINT_POS),
                            ("r_leg_kny", ObservationType.JOINT_POS),
                            ("r_leg_aky", ObservationType.JOINT_POS),
                            ("l_leg_hpx", ObservationType.JOINT_POS),
                            ("l_leg_hpy", ObservationType.JOINT_POS),
                            ("l_leg_hpz", ObservationType.JOINT_POS),
                            ("l_leg_kny", ObservationType.JOINT_POS),
                            ("l_leg_aky", ObservationType.JOINT_POS),

                            ("root", ObservationType.JOINT_VEL),
                            ("r_leg_hpy", ObservationType.JOINT_VEL),
                            ("r_leg_hpx", ObservationType.JOINT_VEL),
                            ("r_leg_hpz", ObservationType.JOINT_VEL),
                            ("r_leg_kny", ObservationType.JOINT_VEL),
                            ("r_leg_aky", ObservationType.JOINT_VEL),
                            ("l_leg_hpy", ObservationType.JOINT_VEL),
                            ("l_leg_hpx", ObservationType.JOINT_VEL),
                            ("l_leg_hpz", ObservationType.JOINT_VEL),
                            ("l_leg_kny", ObservationType.JOINT_VEL),
                            ("l_leg_aky", ObservationType.JOINT_VEL),
                            ]

        collision_groups = [("ground", ["ground"]),
                            ("right_foot_back", ["right_foot_back"]),
                            ("right_foot_front", ["right_foot_front"]),
                            ("left_foot_back", ["left_foot_back"]),
                            ("left_foot_front", ["left_foot_front"]),
                            ]

        super().__init__(xml_path, action_spec, observation_spec, gamma=gamma, horizon=horizon,
                         n_substeps=n_intermediate_steps, collision_groups=collision_groups)

        self.info.action_space.low[:] = -1.0
        self.info.action_space.high[:] = 1.0

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

        # modify the observation space
        self.info.observation_space.low[0] = 0      # pelvis height
        self.info.observation_space.high[0] = 3
        self.info.observation_space.low[1:5] = -1   # quaternions
        self.info.observation_space.high[1:5] = 1
        self.info.observation_space.low[15:18] = -10    # translational velocity pelvis
        self.info.observation_space.high[15:18] = 10
        self.info.observation_space.low[18:21] = -10    # rotational velocity pelvis
        self.info.observation_space.high[18:21] = 10
        self.info.observation_space.low[21:31] = -100   # rotational velocity joints
        self.info.observation_space.high[21:31] = 100   # rotational velocity joints


    def _get_observation_space(self):
        sim_low, sim_high = (self.info.observation_space.low[2:],
                             self.info.observation_space.high[2:])

        grf_low, grf_high = (-np.ones((12,)) * np.inf,
                             np.ones((12,)) * np.inf)

        r_low, r_high = self.goal_reward.get_observation_space()

        return (np.concatenate([sim_low, grf_low, r_low]),
                np.concatenate([sim_high, grf_high, r_high]))

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
                              self._get_collision_force("ground", "right_foot_back")[:3]/10000.,
                              self._get_collision_force("ground", "right_foot_front")[:3]/10000.,
                              self._get_collision_force("ground", "left_foot_back")[:3]/10000.,
                              self._get_collision_force("ground", "left_foot_front")[:3]/10000.,
                              self.goal_reward.get_observation(),
                              ]).flatten()

        return obs

    def _reward(self, state, action, next_state):
        goal_reward = self.goal_reward(state, action, next_state)
        return goal_reward

    def _setup(self):
        self.goal_reward.reset_state()

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
        torso_euler = quat_to_euler(state[1:5])
        return ((state[0] < 0.80) or (state[0] > 1.20)
                or abs(torso_euler[0]) > np.pi / 12
                or (torso_euler[1] < -np.pi / 12) or (torso_euler[1] > np.pi / 8)
                or (torso_euler[2] < -np.pi / 4) or (torso_euler[2] > np.pi / 4)
                )

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

    env = Atlas()

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
        action = np.random.randn(action_dim)/10
        nstate, _, absorbing, _ = env.step(action)

        env.render()
