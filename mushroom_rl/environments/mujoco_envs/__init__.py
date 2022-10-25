from .ball_in_a_cup import BallInACup
try:
    from .humanoids import HumanoidGait
    HumanoidGait.register()
except:
    pass
from .humanoids import Atlas, AtlasTrajectory
from .humanoids import FullHumanoid, FullHumanoidTrajectory

BallInACup.register()

Atlas.register()