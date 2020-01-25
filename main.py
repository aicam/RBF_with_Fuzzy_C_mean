from plot_inputs import plot_inputs
from rbf.G_functions import G_matrix_exponential, covariance
from rbf.train import trainRBF,np
from FCM import fcm
import csv

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


read_data(FILENAME)
print(INPUT[0])
fcm(np.array(INPUT),M)
# trainRBF(INPUT,Y,.1,fcm(np.array(INPUT),M),2,M)