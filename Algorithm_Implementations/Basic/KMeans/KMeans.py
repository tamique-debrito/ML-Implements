# Created by Tamique de Brito.
# Updated 22 June, 2019

import numpy
import copy
import random

"""
Quick test:

import numpy; import random
KM = KMeans()
data = numpy.array([[random.gauss(-1,1), random.gauss(-1,1)] for _ in range(30)] + [[random.gauss(1,1), random.gauss(1,1)] for _ in range(30)])
KM.train(data, 6, 20, 0.1)

On one line for convenience:

import numpy; import random; KM = KMeans(); data = numpy.array([[random.gauss(-1,1), random.gauss(-1,1)] for _ in range(30)] + [[random.gauss(1,1), random.gauss(1,1)] for _ in range(30)]); KM.train(data, 6, 20, 0.1)


"""



"""
Another test:
    


import numpy; import random
clusterGen = lambda n, p, sd: [[random.gauss(p[0],sd), random.gauss(p[1],sd)] for i in range(n)]
KM = KMeans()
data = numpy.array(clusterGen(6,[1,4],0.1) + clusterGen(20, [-4,-7], 0.3)+ clusterGen(30, [5,7],0.4) + clusterGen(30,[3,-8],1))
KM.train(data, 10, 50, 0.1)
for i in range(len(KM.means)):
    c = KM.getCluster(i)
    x = [p[0] for p in c]; y = [p[1] for p in c]
    plt.plot(x,y,'.C'+str(i), alpha=0.5, markersize=7)
    plt.plot(KM.means[i][0],KM.means[i][1],'sC'+str(i), alpha=1.0, markersize=8)
"""




class KMeans:
    """
    Implements the K-Means clustering algorithm.

    In the context of this class, "point" refers to a 1D numpy array representing
        a Euclidean vector

    self.data: a list of points.
    self.clusteredData: a list of (p, cluster) pairs where p is point
        and cluster is the cluster number p is assigned to.
    self.means: a list of points representing cluster means.
    self.numClusters: number of clusters.
    self.maxIterations: max number of iterations of algorithm.
    self.displacementMinLimit: the minimum displacement which must be undergone by at least
        one cluster mean in order for more iterations to be computed.
    self.converged: a boolean representing whether the last iteration resulted in a maximum
        dislacement of updated means of less than self.displacementMin, indicating that means
        have converged to desired extent.
    """
    def __init__(self):
        self.data = None
        self.clusteredData = None
        self.numClusters = None
        self.means = None
        self.maxIterations = None
        self.displacementMinLimit = None

    def train(self, data, numClusters, maxIterations, displacementMinLimit):
        """
        Uses the K-Means algorithm to determine cluster means for data.

        data: a list of points.
        numClusters: an integer number of clusters.
        maxIterations: an integer representing max number of iterations of algorithm.
        displacementMinLimit: a float the minimum displacement which must be undergone by at least
            one cluster mean in order for more iterations to be computed.
        """
        self.data = data
        self.numClusters = numClusters
        self.maxIterations = maxIterations
        self.displacementMinLimit = displacementMinLimit
        self.means = [copy.copy(random.choice(self.data)) for _ in range(numClusters)]
        self.clusteredData = [[p, self.closestMean(p)] for p in self.data]

        for i in range(self.maxIterations):
            self.trainAux()
            if self.converged:
                break

    def trainAux(self):
        for p in self.clusteredData:
            p[1] = self.closestMean(p[0])
        currentMean = None
        maxDisplacement = 0
        for i in range(self.numClusters):
            currentMean = self.clusterMean(i)
            maxDisplacement = max(maxDisplacement, numpy.linalg.norm(currentMean - self.means[i]))
            self.means[i] = currentMean
        self.converged = maxDisplacement < self.displacementMinLimit
        
    def predict(self, point):
        """
        Same as self.closestMean, but with the semantics of classifying a new point.
        """
        return self.closestMean(point)
    
    def clusterMean(self, cluster):
        """
        Computes center of the set of points assigned to a specific cluster.

        cluster: the cluster index to compute the center of.

        Returns the component-wise average of the points that are assigned to cluster.
        """
        return numpy.mean(self.getCluster(cluster), axis=0)
    
    def closestMean(self, point):
        """
        Finds the closest of the current means to a given point.

        point: a point (1D numpy array)

        Returns the index of the closest cluster mean in self.means.
        """
        
        distance = lambda mu: numpy.linalg.norm(point - mu)

        minMean = min(self.means, key=distance)

        for i in range(len(self.means)):
            if all(self.means[i] == minMean):
                return i

    def getCluster(self, cluster):
        """
        Gets all points in self.clusteredData which are part of
            a specified cluster.
            
        cluster: the cluster index to find members of.
        
        Returns a list of all points in the specified cluster.
        """
        return [x[0] for x in self.clusteredData if x[1] == cluster]
