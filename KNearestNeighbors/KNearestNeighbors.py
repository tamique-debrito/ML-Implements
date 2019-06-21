import heapq

"""
A quick test (should print a sequence of 6 True values):

K = KNearestNeighbors()
import random; data = [[[random.gauss(1,0.5), random.gauss(3,1)], 0] for _ in range(50)] + [[[random.gauss(-1,1), random.gauss(0,1)], 1] for _ in range(10)] + [[[random.gauss(5,2), random.gauss(6,1)], 2] for _ in range(10)]
K.setData(data, 3, 6)
print(K.predict([1,0])==0);print(K.predict([0,3])==0);print(K.predict([-1,0])==1);print(K.predict([-5,0])==1);print(K.predict([5,6])==2);print(K.predict([10,10])==2);

On a single line, for convenience:

K = KNearestNeighbors(); import random; data = [[[random.gauss(1,0.5), random.gauss(3,1)], 0] for _ in range(50)] + [[[random.gauss(-1,1), random.gauss(0,1)], 1] for _ in range(10)] + [[[random.gauss(5,2), random.gauss(6,1)], 2] for _ in range(10)]; K.setData(data, 3, 6); print(K.predict([1,0])==0);print(K.predict([0,3])==0);print(K.predict([-1,0])==1);print(K.predict([-5,0])==1);print(K.predict([5,6])==2);print(K.predict([10,10])==2);


"""


class KNearestNeighbors:
    """
    A class implementing the K-Nearest Neighbors algorithm.
    It is used by passing data + K-parameter in through setData().
        It is assumed that data points are preformatted to be Euclidean vectors.
        (i.e. list of floats).

    Fields:
        data: a list of (point, label) pairs, where data is a list of floats
                representing a Euclidean vector, label is the label for the point.
        K: an integer storing the K-parameter.
        numLabels: the number of labels that the data is classifed by.
        dist: a distance function, which takes two Euclidean vectors as arguments.
            defaults to squared Euclidean distance.
    """
    def __init__(self):
        self.data = None
        self.K = None
        self.numLabels = None
        self.dist = None
    def setData(self, data, numLabels, K, dist = None):
        """
        Sets the data and K-parameter.

        data: a list of (point, label) pairs, where data is a list of floats
                representing a Euclidean vector, label is the label for the point.
        K: an integer storing the K-parameter.
        numLabels: the number of labels that the data is classifed by.
        dist: a distance function, which takes two Euclidean vectors as arguments.

        Returns None.
        """
        self.data = data
        self.K = K
        self.numLabels = numLabels
        if dist == None:
            self.dist = lambda x, y: sum([(x[i]-y[i])**2 for i in range(len(x))])
        else:
            self.dist = dist
    def predict(self, point):
        """
        Computes the predicted class of the point.

        point: a list of floats representing a Euclidean feature vector to be classified.

        Returns the number of the predicted label for the point.
        """
        distance_to_point = lambda p: self.dist(point, p[0])
        # Get K-nearest neigbors
        k_nearest = heapq.nsmallest(self.K, self.data, key= distance_to_point)
        votes = [0]*self.numLabels
        for n in k_nearest:
            votes[n[1]] += 1
        return votes.index(max(votes))

