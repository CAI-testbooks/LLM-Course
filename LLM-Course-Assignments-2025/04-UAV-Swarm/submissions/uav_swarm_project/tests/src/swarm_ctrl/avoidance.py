import numpy as np

def compute_repulsion(lidar_points, safety_radius):
    rep = np.zeros(3)
    for p in lidar_points:
        d = np.linalg.norm(p)
        if d < safety_radius and d > 0.2:
            rep += (-p / d) * ((safety_radius - d) / safety_radius)
    return rep
