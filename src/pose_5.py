import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    # Initialize the optimizer 

    params = gtsam.LevenbergMarquardtParams()

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)

    # Perform the optimization and print the result

    result = optimizer.optimize()

    return result

def minimize_marginals(graph, initial_estimate, pose_options):

    sum_of_marginals = float("inf")
    best_pose = None
    best_landmark = None
    best_total = None

    for pose_name, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            trial_graph = graph.clone()
            trial_estimate = gtsam.Values(initial_estimate)
            trial_graph, trial_estimate = add_pose(trial_graph, trial_estimate, pose_5)
            result = optimize(trial_graph, trial_estimate)
            trial_graph = add_landmark_measurement(trial_graph, result, pose_5, landmark)
            result = optimize(trial_graph, trial_estimate)

            marginals = gtsam.Marginals(trial_graph, result)

            score = np.trace(marginals.marginalCovariance(L(landmark)))

            total = marginals.marginalCovariance(L(1)).sum() + marginals.marginalCovariance(L(2)).sum()

            if score < sum_of_marginals:
                sum_of_marginals = score
                best_pose = pose_name
                best_landmark = landmark
                best_total = total

    return best_pose, best_landmark, best_total

def minimize_errors(graph, initial_estimate, pose_options):
    best_error = float('inf')
    best_pose = None
    best_landmark = None
    list_of_errors = []

    for pose_name, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            trial_graph = graph.clone()
            trial_estimate = gtsam.Values(initial_estimate)
            trial_graph, trial_estimate = add_pose(trial_graph, trial_estimate, pose_5)
            result = optimize(trial_graph, trial_estimate)
            trial_graph = add_landmark_measurement(trial_graph, result, pose_5, landmark)
            result = optimize(trial_graph, trial_estimate)

            candidate_errors = []
            for pose_idx in [1, 2, 3]:
                pose_error = 0.0
                for i in range(trial_graph.size()):
                    factor = trial_graph.at(i)
                    if X(pose_idx) in factor.keys():
                        pose_error += factor.error(result)
                candidate_errors.append(pose_error)
            
            total = sum(candidate_errors)
            list_of_errors.append(total)

            if total < best_error:
                best_error = total
                best_pose = pose_name
                best_landmark = landmark

    sum_of_errors = sum(list_of_errors)

    return best_pose, best_landmark, sum_of_errors