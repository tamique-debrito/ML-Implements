# Created by Tamique de Brito.
# Updated 22 June, 2019

"""
Basic data generation and manipulation tools

*******************************

Names and descriptions:

Remove labels from labelled data set: returns a list of points with labels removed. Assumes data is a list of (point, label) pairs.
    Name: removeLabels
Cluster generator: Generates a cluster of 'n' points at point 'p' = (x, y) with std sd:
    Name: clusterGen
Cluster generator (higher dimensions): Generates a cluster of 'n' points at point 'p' = (x, y) with std sd:
    Name: clusterGenND
Data labeller: Labels data 'data' with label 'label' in the form of a list composed of (datapoint, label) pairs:
    Name: dataLabel
Data formatter for drawing: Gets x,y lists corresponding to a particular label for data given in (point,label) format:
    Name: dataFormatForDraw
Data drawer: draws data according to label, assumes data is in (point,label) format:
    Name dataDraw
Label by algorithm: Labels data according to prediction from model M. Assumes data is a list of (x,y) pairs:
    Name: labelByModel
*******************************

Functions:
"""

import numpy as np
import matplotlib.pyplot as plt
import random

removeLabels = lambda data: [d[0] for d in data]
clusterGen = lambda n, p, sd: [np.array([random.gauss(p[0],sd), random.gauss(p[1],sd)]) for i in range(n)]
clusterGenND = lambda n, p, sd, N: [np.array([random.gauss(p[0],sd) for _ in range(N)]) for i in range(n)]
dataLabel = lambda data, label: [(point, label) for point in data]
dataFormatForDraw = lambda data, label: ([d[0][0] for d in data if d[1] == label],[d[0][1] for d in data if d[1] == label])
labelByModel = lambda data, M: [[p, M.predict(p)] for p in data]
def dataDraw(data):
    labelSet = set()
    for point in data:
        labelSet.add(point[1])
    labelEnum = {l : val for val, l in enumerate(labelSet)}
    for l in labelSet:
        x, y = dataFormatForDraw(data, l)
        plt.plot(x,y, '.C'+str(labelEnum[l]))

    


"""
Tests:

*******************************

clusterGen = lambda n, p, sd: [np.array([random.gauss(p[0],sd), random.gauss(p[1],sd)]) for i in range(n)]
dataLabel = lambda data, label: [(point, label) for point in data]
data = dataLabel(clusterGen(10, (2,4), 1),1)+dataLabel(clusterGen(15, (-2,-2), 1),-1)

*******************************

colors = ['b','g','m','c','y','k']
KM = KMeans()
data = numpy.array(clusterGen(6,[1,4],0.1) + clusterGen(20, [-4,-7], 0.3)+ clusterGen(30, [5,7],0.4) + clusterGen(30,[3,-8],1))
KM.train(data, 10, 50, 0.1)
for i in range(len(KM.means)):
    c = KM.getCluster(i)
    x = [p[0] for p in c]; y = [p[1] for p in c]
    plt.plot(x,y,'.C'+str(i), alpha=0.5, markersize=7)
    plt.plot(KM.means[i][0],KM.means[i][1],'sC'+str(i), alpha=1.0, markersize=8)
"""