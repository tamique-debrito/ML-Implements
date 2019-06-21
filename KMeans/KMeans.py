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
    def evaluate(self, point):
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
        assignedToCluster = [x[0] for x in self.clusteredData if x[1] == cluster]
        return numpy.mean(assignedToCluster, axis=0)
    
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

