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

class AveragedPerceptron:
    """
    Implements the averaged perceptron algorithm.
    
    self.data: a list of (point, label) pairs, where 'point' is a numpy array
                and 'label' a label that is either +1 or -1. This is the data set
                that is being trained on.
    self.maxIterations: integer maximum number of training iterations through data set.
    self.dimensionality: integer number of dimensions of feature vectors.
    self.currentNormal: numpy array of dimension self.dimensionality.
            This is used for training the perceptron.
    self.currentOffset: float. This is used for training the perceptron.
    self.averagedNormal: numpy array of dimension self.dimensionality.
            This is a parameter used for prediction.
    self.currentOffset: float. This is a parameter used for prediction.
    """
    def __init__(self):
        self.data = None
        self.maxIterations = None
        self.dimensionality = None
        self.currentNormal = None
        self.currentOffset = None
        self.dimensionality = None
        self.averagedNormal = None
        self.averagedOffset = None
    def train(self, data, maxIterations, dimensionality, shuffleAtEachIteration=True, lifetimeStart=0):
        """
        Trains the perceptron on data.
        
        data: a list of (point, label) pairs, where 'point' is a numpy array
                and 'label' a label that is either +1 or -1.
        maxIterations: maximum number of training iterations through data set.
        dimensionality: integer number of dimensions of feature vectors.
        shuffleAtEachIteration: boolean indicating whether to shuffle data at each
                iteration through all data points.
        lifetimeStart: integer indicating the weight that each new parameter should initially
                be given in the averaged parameter.
        """
        self.data = data
        self.maxIterations = maxIterations
        self.dimensionality = len(self.data[0][0])
        if self.dimensionality != dimensionality:
            raise ValueError("Data dimensionality does not match \
                             specified dimensionality!")
        self.currentNormal = np.zeros(self.dimensionality)
        self.currentOffset = 0
        self.averagedNormal = np.zeros(self.dimensionality)
        self.averagedOffset = 0
        
        currentLifetime = lifetimeStart
        for i in range(self.maxIterations):
            if shuffleAtEachIteration:
                random.shuffle(self.data)
            for point, label in self.data:
                if self.trainIteration(point, label):
                    currentLifetime = currentLifetime + 1
                # When current parameters make an error, add their weighted contributions
                # to the averaged parameters.
                else:
                    self.averagedNormal += self.averagedNormal + currentLifetime * self.currentNormal
                    self.averagedOffset += self.averagedOffset + currentLifetime * self.currentOffset
                    currentLifetime = lifetimeStart

    def trainIteration(self, point, label):
        """
        Trains on a single point.
        
        point: numpy array.
        label: an integer label for 'point', which is either +1 or -1.
        
        If self.normal and self.offset are initialized to numpy array and float
            respectively, 'point' is a numpy array, and 'label' is either +1 or -1 then self.normal and
            self.offset will be updated to reflect the result of running a single
            iteration of the perceptron training algorithm.
            
        Returns boolean. True if self.currentNormal and self.currentOffset
            classify 'point' to 'label', False otherwise
        """
        classifiedCorrectly = self.evaluate(point) * label >= 0
        if not classifiedCorrectly:
            self.currentNormal = self.currentNormal + label * point
            self.currentOffset = self.currentOffset - label
        return classifiedCorrectly
    def predict(self, point):
        """
        Predicts label of point based on model.
        Usess self.averagedNormal and self.averagedOffset.
        
        point: a numpy array.
        
        Returns the predicted label.
        """
        return 1 if np.dot(point, self.averagedNormal) > self.averagedOffset else -1
        
    def evaluate(self, point):
        """
        Evaluates label of point based on current parameters. For use in training.
        Uses self.currentNormal and self.currentOffset
               
        point: a numpy array.
        
        Returns the predicted label.
        """
        return 1 if np.dot(point, self.currentNormal) > self.currentOffset else -1
    
