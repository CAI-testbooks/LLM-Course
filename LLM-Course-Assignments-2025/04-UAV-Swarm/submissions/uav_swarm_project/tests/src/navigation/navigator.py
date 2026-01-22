import numpy as np
from utils.math_utils import normalize
from swarm_ctrl.fsm import DroneState
from config.params import *

class Navigator:
    def decide(self, curr_pos, target, avoid_vec):
        dist = np.linalg.norm(curr_pos - target)

        if dist < ARRIVAL_DIST:
            return DroneState.ARRIVED, np.zeros(3)

        if np.linalg.norm(avoid_vec) > AVOID_THRESHOLD:
            return DroneState.AVOID, normalize(avoid_vec) * AVOID_GAIN

        cruise_vec = normalize(target - curr_pos) * CRUISE_SPEED
        return DroneState.CRUISE, cruise_vec
