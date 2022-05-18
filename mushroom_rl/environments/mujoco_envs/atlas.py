from mushroom_rl.environments.mujoco import MuJoCo, ObservationType
import numpy as np
from pathlib import Path

from mushroom_rl.utils.angles import quat_to_euler


class Atlas(MuJoCo):
    """
    Mujoco simulation of the Atlas robot.

    """
    def __init__(self):
        """
        Constructor.

        """
        xml_path = (Path(__file__).resolve().parent / "data" / "atlas" / "model.xml").as_posix()

        action_spec = ["l_leg_hpz_actuator", "l_leg_hpx_actuator", "l_leg_hpy_actuator", "l_leg_kny_actuator",
                       "l_leg_aky_actuator", "r_leg_hpz_actuator", "r_leg_hpx_actuator", "r_leg_hpy_actuator",
                       "r_leg_kny_actuator", "r_leg_aky_actuator"]

        observation_spec = [("root", ObservationType.JOINT_POS),
                            ("root", ObservationType.JOINT_VEL),
                            # Left Leg
                            ("l_leg_hpx", ObservationType.JOINT_POS),
                            ("l_leg_hpx", ObservationType.JOINT_VEL),
                            ("l_leg_hpy", ObservationType.JOINT_POS),
                            ("l_leg_hpy", ObservationType.JOINT_VEL),
                            ("l_leg_hpz", ObservationType.JOINT_POS),
                            ("l_leg_hpz", ObservationType.JOINT_VEL),
                            ("l_leg_kny", ObservationType.JOINT_POS),
                            ("l_leg_kny", ObservationType.JOINT_VEL),
                            ("l_leg_aky", ObservationType.JOINT_POS),
                            ("l_leg_aky", ObservationType.JOINT_VEL),
                            # Right Leg
                            ("r_leg_hpx", ObservationType.JOINT_POS),
                            ("r_leg_hpx", ObservationType.JOINT_VEL),
                            ("r_leg_hpy", ObservationType.JOINT_POS),
                            ("r_leg_hpy", ObservationType.JOINT_VEL),
                            ("r_leg_hpz", ObservationType.JOINT_POS),
                            ("r_leg_hpz", ObservationType.JOINT_VEL),
                            ("r_leg_kny", ObservationType.JOINT_POS),
                            ("r_leg_kny", ObservationType.JOINT_VEL),
                            ("r_leg_aky", ObservationType.JOINT_POS),
                            ("r_leg_aky", ObservationType.JOINT_VEL)]

        super().__init__(xml_path, action_spec, observation_spec, 0.99, 1000, n_substeps=5)

    def _get_observation_space(self):
        sim_low, sim_high = (self.info.observation_space.low[2:],
                             self.info.observation_space.high[2:])

        return sim_low, sim_high

    def _create_observation(self):
        obs = super()._create_observation()[2:]

        return obs

    def _reward(self, state, action, next_state):
        return 0.

    def _is_absorbing(self, state):

        return self._has_fallen(state)

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

    absorbing = False
    while not absorbing:
        print('test')
        action = np.random.randn(action_dim)/3
        action = np.zeros(action_dim)
        _, _, absorbing, _ = env.step(action)
        env.render()
        time.sleep(0.5)
