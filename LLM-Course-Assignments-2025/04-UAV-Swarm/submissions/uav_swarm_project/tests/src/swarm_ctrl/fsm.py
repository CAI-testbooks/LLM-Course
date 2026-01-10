from enum import Enum

class DroneState(Enum):
    CRUISE = 1
    AVOID = 2
    ARRIVED = 3