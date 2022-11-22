from .ball_in_a_cup import BallInACup
from .air_hockey import AirHockeyHit, AirHockeyDefend, AirHockeyPrepare, AirHockeyRepel
from .humanoids import BaseHumanoid
from .humanoids import Atlas
from .humanoids import ReducedHumanoidTorque
from .humanoids import FullHumanoid

BallInACup.register()
Atlas.register()
FullHumanoid.register()
ReducedHumanoidTorque.register()
AirHockeyHit.register()
AirHockeyDefend.register()
AirHockeyPrepare.register()
AirHockeyRepel.register()
