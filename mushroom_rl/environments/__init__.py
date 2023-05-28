try:
    Atari = None
    from .atari import Atari
    Atari.register()
except ImportError:
    pass

try:
    Gym = None
    from .gym_env import Gym
    Gym.register()
    from .gym_mujoco_pomdp import NoisyDelayedGym, RandomizedMassGym
except ImportError:
    pass

try:
    DMControl = None
    from .dm_control_env import DMControl
    DMControl.register()
except ImportError:
    pass

try:
    MiniGrid = None
    from .minigrid_env import MiniGrid
    MiniGrid.register()
except ImportError:
    pass

try:
    iGibson = None
    from .igibson_env import iGibson
    iGibson.register()
except ImportError:
    import logging
    logging.disable(logging.NOTSET)

try:
    Habitat = None
    from .habitat_env import Habitat
    Habitat.register()
except ImportError:
    pass

try:
    MuJoCo = None
    from .mujoco import MuJoCo
    from .multi_mujoco import MultiMuJoCo
    from .mujoco_envs import *
except ImportError:
    pass


try:
    PyBullet = None
    from .pybullet import PyBullet
    from .pybullet_envs import *
except ImportError:
    pass

from .generators.simple_chain import generate_simple_chain

from .car_on_hill import CarOnHill
CarOnHill.register()

from .cart_pole import CartPole
CartPole.register()

from .finite_mdp import FiniteMDP
FiniteMDP.register()

from .grid_world import GridWorld, GridWorldVanHasselt
GridWorld.register()
GridWorldVanHasselt.register()

from .inverted_pendulum import InvertedPendulum
InvertedPendulum.register()

from .lqr import LQR
LQR.register()

from .puddle_world import PuddleWorld
PuddleWorld.register()

from .segway import Segway
Segway.register()

from .ship_steering import ShipSteering
ShipSteering.register()

if Gym:
    from gym.envs.registration import register

    register(
        id='AntPOMDP-v3',
        entry_point='mushroom_rl.environments.gym_mujoco_pomdp:AntEnvPOMPD',
    )
    register(
        id='HalfCheetahPOMDP-v3',
        entry_point='mushroom_rl.environments.gym_mujoco_pomdp:HalfCheetahEnvPOMPD',
    )
    register(
        id='HopperPOMDP-v3',
        entry_point='mushroom_rl.environments.gym_mujoco_pomdp:HopperEnvPOMPD',
    )
    register(
        id='HumanoidPOMDP-v3',
        entry_point='mushroom_rl.environments.gym_mujoco_pomdp:HumanoidEnvPOMPD',
    )
    register(
        id='Walker2dPOMDP-v3',
        entry_point='mushroom_rl.environments.gym_mujoco_pomdp:Walker2dEnvPOMPD',
    )
    register(
        id='InvertedPendulumEnvPOMDP',
        entry_point='mushroom_rl.environments.gym_mujoco_pomdp:InvertedPendulumEnvPOMDP',
    )
    register(
        id='PendulumEnvPOMDP',
        entry_point='mushroom_rl.environments.gym_mujoco_pomdp:PendulumEnvPOMDP',
    )

    register(
        id='BipedalWalkerPOMDP',
        entry_point='mushroom_rl.environments.other_pomdp_envs:BipedalWalkerPOMDP',
    )

