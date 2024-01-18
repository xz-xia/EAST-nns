from scipy.spatial.distance import euclidean
import numpy as np
def Frechet_distance(exp_data,num_data):
    """
    cal fs by dynamic programming
    :param exp_data: array_like, (M,N) shape represents (data points, dimensions)
    :param num_data: array_like, (M,N) shape represents (data points, dimensions)
    # e.g. P = [[2,1] , [3,1], [4,2], [5,1]]
    # Q = [[2,0] , [3,0], [4,0]]
    :return:
    """
    P=exp_data
    Q=num_data
    p_length = len(P)
    q_length = len(Q)
    distance_matrix = np.ones((p_length, q_length)) * -1

    # fill the first value with the distance between
    # the first two points in P and Q
    distance_matrix[0, 0] = euclidean(P[0], Q[0])

    # load the first column and first row with distances (memorize)
    for i in range(1, p_length):
        distance_matrix[i, 0] = max(distance_matrix[i - 1, 0], euclidean(P[i], Q[0]))
    for j in range(1, q_length):
        distance_matrix[0, j] = max(distance_matrix[0, j - 1], euclidean(P[0], Q[j]))

    for i in range(1, p_length):
        for j in range(1, q_length):
            distance_matrix[i, j] = max(
                min(distance_matrix[i - 1, j], distance_matrix[i, j - 1], distance_matrix[i - 1, j - 1]),
                euclidean(P[i], Q[j]))
    return distance_matrix[p_length-1,q_length-1] # 最后一步即为弗雷彻距离