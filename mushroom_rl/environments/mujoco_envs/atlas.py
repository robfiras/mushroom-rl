import mujoco_py

from mushroom_rl.environments.mujoco import MuJoCo, ObservationType
import numpy as np
from pathlib import Path

from mushroom_rl.utils import spaces
from mushroom_rl.utils.angles import quat_to_euler
from mushroom_rl.utils.running_stats import *


class Atlas(MuJoCo):
    """
    Mujoco simulation of the Atlas robot.

    """
    def __init__(self, gamma=0.99, horizon=2000, n_intermediate_steps=10):
        """
        Constructor.

        """
        xml_path = (Path(__file__).resolve().parent / "data" / "atlas" / "model.xml").as_posix()

        action_spec = ["r_leg_hpy_actuator", "r_leg_hpx_actuator", "r_leg_hpz_actuator", "r_leg_kny_actuator",
                       "r_leg_aky_actuator", "l_leg_hpy_actuator", "l_leg_hpx_actuator", "l_leg_hpz_actuator",
                       "l_leg_kny_actuator", "l_leg_aky_actuator"]

        observation_spec = [("root", ObservationType.JOINT_POS),
                            ("r_leg_hpy", ObservationType.JOINT_POS),
                            ("r_leg_hpx", ObservationType.JOINT_POS),
                            ("r_leg_hpz", ObservationType.JOINT_POS),
                            ("r_leg_kny", ObservationType.JOINT_POS),
                            ("r_leg_aky", ObservationType.JOINT_POS),
                            ("l_leg_hpy", ObservationType.JOINT_POS),
                            ("l_leg_hpx", ObservationType.JOINT_POS),
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

        low, high = self.info.action_space.low.copy(),\
                    self.info.action_space.high.copy()
        self.norm_act_mean = (high + low) / 2.0
        self.norm_act_delta = (high - low) / 2.0
        self.info.action_space.low[:] = -1.0
        self.info.action_space.high[:] = 1.0

        self.info.observation_space = spaces.Box(*self._get_observation_space())
    #
    # def step(self, action):
    #     action = ((action.copy() * self.norm_act_delta) + self.norm_act_mean)
    #
    #     state, reward, absorbing, info = super().step(action)
    #
    #     return state, reward, absorbing, info

    def _get_observation_space(self):
        sim_low, sim_high = (self.info.observation_space.low[2:],
                             self.info.observation_space.high[2:])

        return sim_low, sim_high

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

        """

        obs = np.concatenate([super()._create_observation()[2:],
                              self._get_collision_force("ground", "right_foot_back")[:3],
                              self._get_collision_force("ground", "right_foot_front")[:3],
                              self._get_collision_force("ground", "left_foot_back")[:3],
                              self._get_collision_force("ground", "left_foot_front")[:3]])

        return obs

    def _reward(self, state, action, next_state):
        return 0.

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
        return ((state[0] < 0.90) or (state[0] > 1.20)
                or abs(torso_euler[0]) > np.pi / 12
                or (torso_euler[1] < -np.pi / 12) or (torso_euler[1] > np.pi / 8)
                or (torso_euler[2] < -np.pi / 4) or (torso_euler[2] > np.pi / 4)
                )


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
        action = np.zeros(action_dim)
        nstate, _, absorbing, _ = env.step(action)

        f_rfb = nstate[31:34]
        f_rff = nstate[34:37]
        f_lfb = nstate[37:40]
        f_lff = nstate[40:43]

        print("grf right_back:", f_rfb)
        print("grf right_front:",f_rff)
        print("grf left_back:", f_lfb)
        print("grf left_front:", f_lff)
        env.render()
        time.sleep(0.5)
