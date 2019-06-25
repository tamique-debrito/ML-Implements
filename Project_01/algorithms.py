# Format data for binary
import data_opener
import math
import numpy as np
import matplotlib.pyplot as plt

# Data formattted as list of (features, label) pairs
aawData = data_opener.getData("Absenteeism_at_work.csv")

# Data formatted as list of featureValues_i, where featureValues_i
#   is a list of all the values of feature i in the data set
valuesByFeature = data_opener.getData("Absenteeism_at_work.csv", transpose=True)

numFeatures = len(valuesByFeature)

dim = int(math.ceil(math.sqrt(numFeatures)))

fig1 = plt.figure()
axes = []

featuresRemaining = numFeatures
for i in range(dim):
    for j in range(dim):
        axes.append(fig1.add_subplot(dim,dim,i*dim + j+1))
for ax, dat in zip(axes, valuesByFeature):
    ax.hist(dat)


fig1.show()