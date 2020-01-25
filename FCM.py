import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt




def fcm(data,data_train,Y_train,data_test,Y_test, classes_count=2, n_clusters=1, landa=.1, n_init=30, m=2, max_iter=300, tol=1e-16):
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
    C = calculate_covariance_mat(data,min_centers,um,m)
    G = np.zeros([len(data_train), len(min_centers)])
    for k in range(len(data_train)):
        for i in range(len(min_centers)):
            G[k][i] += np.exp(-landa * np.matmul(
                np.matmul(np.transpose(np.array(data_train[k]) - np.array(min_centers[i])), np.linalg.inv(C[i])),
                np.array(np.array(data_train[k]) - np.array(min_centers[i]))))
    try:
        W = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(G),G)),np.transpose(G)),generate_Y(len(data_train),Y_train,classes_count))
    except np.linalg.LinAlgError:
        W = np.matmul(np.matmul(np.random.rand([len(min_centers),len(min_centers)]), np.transpose(G)), generate_Y(len(data_train), Y_train,classes_count))
    y_hat = np.argmax(np.matmul(G, W), axis=1)
    acc = accuracy(Y_train, y_hat, len(data))
    print("train data accuracy: " , acc)
    G = np.zeros([len(data_test), len(min_centers)])
    for k in range(len(data_test)):
        for i in range(len(min_centers)):
            G[k][i] += np.exp(-landa * np.matmul(
                np.matmul(np.transpose(np.array(data_test[k]) - np.array(min_centers[i])), np.linalg.inv(C[i])),
                np.array(np.array(data_test[k]) - np.array(min_centers[i]))))
    y_hat = np.argmax(np.matmul(G,W), axis=1)
    acc = accuracy(Y_test,y_hat,len(data))
    print("test data accuracy: " , acc)
    return min_centers

def generate_Y(x_d, y_vector, classes_count):
    Y = []
    for i in range(x_d):
        y_i = [0 for i in range(classes_count)]
        y_i[y_vector[i]] = 1
        Y.append(y_i)
    return np.array(Y).reshape([x_d, classes_count])

def calculate_covariance_mat(data, centers, U, m):
    C = []
    centers = np.array(centers)
    for i in range(len(centers)):
        shape = (2,2)
        divided_sum = np.zeros(shape)
        divisor_sum = .0
        for k in range(len(data)):
            divided_sum += np.power(U[i][k],m)* np.multiply(
                np.array([data[k] - centers[i]]), np.transpose(
                    np.array([data[k] - centers[i]])
                )
            )
            divisor_sum += np.power(U[i][k], m)
        C.append(divided_sum/ divisor_sum)
    return C

def accuracy(y, y_hat, n):
    return 1 - (np.sum(np.abs(np.sign(np.subtract(y, y_hat)))) / n)