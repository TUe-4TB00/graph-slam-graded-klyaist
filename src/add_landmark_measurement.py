import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_landmark_measurement(graph, initial_estimate, result):
    # Determine the correct rotation (bearing) and distance from X(4) to L(2) 
    
    X4 = result.atPose2(X(4))
    L2 = result.atPoint2(L(2))

    dx = L2.x() - X4.x()
    dy = L2.y() - X4.y()

    distance = math.hypot(dx, dy)

    rotation = math.atan2(dy, dx) - X4.theta()

    graph.add(gtsam.BearingRangeFactor2D(X(4), L(2), gtsam.Rot2.fromRadians(rotation), distance, MEASUREMENT_NOISE))
    return graph