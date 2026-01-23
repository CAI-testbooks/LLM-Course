import numpy as np

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.zeros_like(v)
