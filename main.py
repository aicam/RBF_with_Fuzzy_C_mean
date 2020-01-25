from plot_inputs import plot_inputs
from rbf.G_functions import G_matrix_exponential, covariance
from rbf.train import trainRBF,np
from FCM import fcm
import csv
import random
'''
    Initial attributes
'''

FILENAME = 'data/1.csv'
M = 5
RADUIS = .1
'''
    Start with plotting inputs
'''
plot_inputs(FILENAME)

INPUT = []
Y = []


def read_data(filename):
    with  open(filename) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if row != []:
                INPUT.append([float(row[0]), float(row[1])])
                Y.append(0 if float(row[2]) == 1.0 else 1)
    csv_file.close()

INPUT = np.array(INPUT)
Y = np.array(Y)
indices = np.arange(INPUT.shape[0])
np.random.shuffle(indices)
INPUT = INPUT[indices]
Y = INPUT[indices]
INPUT = np.ndarray.tolist(INPUT)
Y = np.ndarray.tolist(Y)
read_data(FILENAME)
x_train = INPUT[0:840]
x_test = INPUT[841:1200]
y_train = Y[0:840]
y_test = Y[841:1200]
fcm(np.array(INPUT),x_train,y_train,x_test,y_test,2,M)
# trainRBF(INPUT,Y,.1,fcm(np.array(INPUT),M),2,M)