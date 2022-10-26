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

        action_spec = ["r_leg_hpx_actuator", "r_leg_hpy_actuator", "r_leg_hpz_actuator", "r_leg_kny_actuator",
                       "r_leg_aky_actuator", "l_leg_hpx_actuator", "l_leg_hpy_actuator", "l_leg_hpz_actuator",
                       "l_leg_kny_actuator", "l_leg_aky_actuator"]

        observation_spec = [("root", "root", ObservationType.JOINT_POS),
                            ("r_leg_hpx", "r_leg_hpx", ObservationType.JOINT_POS),
                            ("r_leg_hpy", "r_leg_hpy", ObservationType.JOINT_POS),
                            ("r_leg_hpz", "r_leg_hpz", ObservationType.JOINT_POS),
                            ("r_leg_kny", "r_leg_kny", ObservationType.JOINT_POS),
                            ("r_leg_aky", "r_leg_aky", ObservationType.JOINT_POS),
                            ("l_leg_hpx", "l_leg_hpx", ObservationType.JOINT_POS),
                            ("l_leg_hpy", "l_leg_hpy", ObservationType.JOINT_POS),
                            ("l_leg_hpz", "l_leg_hpz", ObservationType.JOINT_POS),
                            ("l_leg_kny", "l_leg_kny", ObservationType.JOINT_POS),
                            ("l_leg_aky", "l_leg_aky", ObservationType.JOINT_POS),

                            ("root", "root", ObservationType.JOINT_VEL),
                            ("r_leg_hpy", "r_leg_hpy", ObservationType.JOINT_VEL),
                            ("r_leg_hpx", "r_leg_hpx", ObservationType.JOINT_VEL),
                            ("r_leg_hpz", "r_leg_hpz", ObservationType.JOINT_VEL),
                            ("r_leg_kny", "r_leg_kny", ObservationType.JOINT_VEL),
                            ("r_leg_aky", "r_leg_aky", ObservationType.JOINT_VEL),
                            ("l_leg_hpy", "l_leg_hpy", ObservationType.JOINT_VEL),
                            ("l_leg_hpx", "l_leg_hpx", ObservationType.JOINT_VEL),
                            ("l_leg_hpz", "l_leg_hpz", ObservationType.JOINT_VEL),
                            ("l_leg_kny", "l_leg_kny", ObservationType.JOINT_VEL),
                            ("l_leg_aky", "l_leg_aky", ObservationType.JOINT_VEL)]

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
