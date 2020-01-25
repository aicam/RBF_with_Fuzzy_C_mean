import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt




def fcm(data,Y, classes_count=2, n_clusters=1, landa=.1, n_init=30, m=2, max_iter=300, tol=1e-16):
    um = None
    min_cost = np.inf
    for iter_init in range(n_init):
        # Randomly initialize centers
        centers = data[np.random.choice(
            data.shape[0], size=n_clusters, replace=False
        ), :]
        # Compute initial distances
        # Zeros are replaced by eps to avoid division issues
        dist = np.fmax(
            cdist(centers, data, metric='sqeuclidean'),
            np.finfo(np.float64).eps
        )
        for iter1 in range(max_iter):

            # Compute memberships
            u = (1 / dist) ** (1 / (m - 1))
            um = (u / u.sum(axis=0)) ** m

            # Recompute centers
            prev_centers = centers
            centers = um.dot(data) / um.sum(axis=1)[:, None]

            dist = cdist(centers, data, metric='sqeuclidean')

            if np.linalg.norm(centers - prev_centers) < tol:
                break

        # Compute cost
        cost = np.sum(um * dist)
        if cost < min_cost:
            min_cost = cost
            min_centers = centers
            mem = um.argmax(axis=0)
    plt.plot(data[:, 0], data[:, 1], 'go', min_centers[:, 0], min_centers[:, 1], 'bs')
    plt.show()
    C = []
    for i in range(len(min_centers)):
        C_numerator = 0
        C_denominator = 0
        for j in range(len(data)):
            C_numerator += (um[i][j] ** m)
        for j in range(len(data)):

            C_denominator += ((um[i][j] ** m) * np.matmul(np.transpose(np.array(data[j]) - np.array(min_centers[i])),
                                                          np.array(np.array(data[j]) - np.array(min_centers[i]))))
        C.append(C_numerator / C_denominator)
    G = np.zeros([len(data), len(min_centers)])

    for k in range(len(data)):
        for i in range(len(min_centers)):
            G[k][i] += np.exp(-landa * np.matmul(
                np.matmul(np.transpose(np.array(data[k]) - np.array(min_centers[i])), generate_C_inverse(data[k], min_centers[i])),
                np.array(np.array(data[k]) - np.array(min_centers[i]))))
    try:
        V = np.matmul(np.transpose(G),G)
        print(V)
        W = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(G),G)),np.transpose(G)),generate_Y(len(data),Y,classes_count))
    except np.linalg.LinAlgError:
        W = np.matmul(np.matmul(np.random.rand([len(min_centers),len(min_centers)]), np.transpose(G)), generate_Y(len(data), Y,classes_count))
    y_hat = np.argmax(np.matmul(G,W), axis=1)
    # print(W)
    return min_centers

def generate_Y(x_d, y_vector, classes_count):
    Y = []
    for i in range(x_d):
        y_i = [0 for i in range(classes_count)]
        y_i[y_vector[i]] = 1
        Y.append(y_i)
    return np.array(Y).reshape([x_d, classes_count])

def generate_C_inverse(x, v):
    ci = np.zeros([len(v), len(v)])
    for i in range(len(v)):
        for j in range(len(x)):
            ci[i][j] += v[i] * x[j]
    try:
        ci_inverse = np.linalg.inv(ci)
    except np.linalg.LinAlgError:
        ci_inverse = np.zeros([len(v),len(v)])
        for i in range(len(v)):
            ci_inverse[i][i] = 1
    return ci_inverse
