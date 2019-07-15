# Created by Tamique de Brito.
# Updated 2 July, 2019

import numpy as np
import random


"""
Tests:

*******************************

clusterGen = lambda n, p, sd: [np.array([random.gauss(p[0],sd), random.gauss(p[1],sd)]) for i in range(n)]
dataLabel = lambda data, label: [(point, label) for point in data]
data = dataLabel(clusterGen(10, (2,4), 1),1)+dataLabel(clusterGen(15, (-2,-2), 1),-1)
P = Perceptron()
P.train(data, 10, 2)

*******************************

"""

class Perceptron:
    """
    Implements the basic perceptron algorithm.
    
    self.data: a list of (point, label) pairs, where 'point' is a numpy array
                and 'label' a label that is either +1 or -1. This is the data set
                that is being trained on.
    self.maxIterations: integer maximum number of training iterations through data set.
    self.dimensionality: integer number of dimensions of feature vectors.
    self.normal: numpy array of dimension self.dimensionality. A parameter for the perceptron.
    self.offset: float. A parameter of the perceptron.
    """
    def __init__(self):
        self.data = None
        self.maxIterations = None
        self.dimensionality = None
        self.normal = None
        self.offset = None
    def train(self, data, maxIterations, dimensionality, shuffleAtEachIteration=True):
        """
        Trains the perceptron on data.
        
        data: a list of (point, label) pairs, where 'point' is a numpy array
                and 'label' a label that is either +1 or -1.
        dimensionality: integer number of dimensions of feature vectors.
        maxIterations: integer maximum number of training iterations through data set.
        shuffleAtEachIteration: boolean indicating whether to shuffle data at each
                iteration through all data points.
        """
        self.data = data
        self.maxIterations = maxIterations
        self.dimensionality = len(self.data[0][0])
        if self.dimensionality != dimensionality:
            raise ValueError("Data dimensionality does not match \
                             specified dimensionality!")
        self.normal = np.zeros(self.dimensionality)
        self.offset = 0
        
        for i in range(self.maxIterations):
            if shuffleAtEachIteration:
                random.shuffle(self.data)
            for point, label in self.data:
                self.trainIteration(point, label)

    def trainIteration(self, point, label):
        """
        Trains on a single point.
        
        point: numpy array.
        label: an integer label for 'point', which is either +1 or -1.
        
        If self.normal and self.offset are initialized to numpy array and float
            respectively, 'point' is a numpy array, and 'label' is either +1 or -1 then self.normal and
            self.offset will be updated to reflect the result of running a single
            iteration of the perceptron training algorithm.
            
        Returns None.
        """
        if self.evaluate(point) * label >= 0:
            self.normal = self.normal + label * point
            self.offset = self.offset - label
    def predict(self, point):
        """
        Predicts label of point based on model.
        Same as self.evaluate, but with semantics of being used to predict.
        
        point: a numpy array.
        
        Returns the predicted label.
        """
        return 1 if np.dot(point, self.normal) > self.offset else -1
        
    def evaluate(self, point):
        """
        Evaluates label of point based on current model.
        Same as self.predict, but with semantics of being used in training.
        
        point: a numpy array.
        
        Returns the predicted label.
        """
        return 1 if np.dot(point, self.normal) > self.offset else -1
    
