import numpy as np
import matplotlib.pyplot as plt
import csv

dates = []
values = []

with open('data.csv', 'rb') as csvfile:
    datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in datareader:
        dates.append(int(row[0]))
        values.append(float(row[1]))

x = np.array(dates)
y = np.array(values)
plt.plot(x, y)

plt.show()
