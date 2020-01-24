import csv
import matplotlib.pyplot as plt


def plot_inputs(filename):
    inputs = []
    with  open(filename) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if row != []:
                inputs.append(row)
    csv_file.close()

    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for item in inputs:
        if float(item[2]) == 1.0:
            x1.append(float(item[0]))
            y1.append(float(item[1]))
        else:
            x2.append(float(item[0]))
            y2.append(float(item[1]))
    plt.plot(x1, y1, 'g^', x2, y2, 'bs')
    # plt.show()
