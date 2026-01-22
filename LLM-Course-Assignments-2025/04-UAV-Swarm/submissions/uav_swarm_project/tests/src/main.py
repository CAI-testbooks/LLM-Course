import time
import numpy as np

from swarm_ctrl.swarm_manager import SwarmManager
from navigation.navigator import Navigator
from visualization.airsim_debug import *
from config.params import *

DRONES = ["Drone1", "Drone2", "Drone3"]

def main():
    manager = SwarmManager(DRONES)
    navigator = Navigator()

    targets = {
        "Drone1": np.array([20, 20, FLY_HEIGHT]),
        "Drone2": np.array([40, 20, FLY_HEIGHT]),
        "Drone3": np.array([20, 40, FLY_HEIGHT])
    }

    manager.takeoff_all()

    active = set(DRONES)

    while active:
        states = manager.get_all_states()

        for name in list(active):
            curr = np.array([states[name]['x'], states[name]['y'], states[name]['z']])

            lidar = manager.client.getLidarData("Lidar1", name)
            points = np.array(lidar.point_cloud).reshape(-1,3) if lidar.point_cloud else []

            avoid = manager.get_all_repulsions(name, states)
            state, move_vec = navigator.decide(curr, targets[name], avoid)

            print_state(manager.client, name, state)
            draw_velocity(manager.client, curr, move_vec)

            if state.name == "ARRIVED":
                active.remove(name)
                continue

            next_pos = curr + move_vec
            manager.client.moveToPositionAsync(
                float(next_pos[0]), float(next_pos[1]), float(next_pos[2]),
                velocity=2, vehicle_name=name
            )

        time.sleep(0.1)

if __name__ == "__main__":
    main()
