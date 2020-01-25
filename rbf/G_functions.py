import numpy as np
from scipy.spatial import distance

'''
    G is the map of data into new groups with different axis
'''
def G_matrix_exponential(X_vector, V_vector, C_vector, landa):
    g = []


    for i in range(len(X_vector)):
        for j in range(len(V_vector)):
            g.append(np.exp(-landa*np.power(X_vector[i] - V_vector[j], 2)*C_vector[j]))
    return np.array(g).reshape([len(X_vector), len(V_vector)])

def covariance(U_vector,X_vector,V_vector, m):
    C = []
    U_vector = []
    for x in X_vector:
        for v in V_vector:
            U_vector.append(distance.euclidean(X_vector,v))
    U_vector = np.array(U_vector).reshape([len()])
    for i in range(len(V_vector)):
        c_i = 0
        uki = 0
        for j in range(len(X_vector)):
            uki += U_vector[j][i]
        for j in range(len(X_vector)):
            c_i += np.power(U_vector[j][i], m)*np.power(X_vector[j] - V_vector[i], 2)
        C.append(c_i)
    return C