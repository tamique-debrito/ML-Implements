# Created by Tamique de Brito.
# Updated 2 July, 2019

# Example (preformatted) dataset:
# data = [[[0,2,1], 1], [[2,2,0], 0], [[0,0,1], 1], [[1,0,1], 2], [[2,2,0], 0], [[1,1,1], 1]]

# Example (preformatted) dataset generator:
#       "n" is size of dataset, "c" is number of features,
#       "catSize" is a list of length "c" where each entry is the
#            number of categories of the feature it corresponds to,
#       and "l" is size of label category.
#
# dataGen = lambda n, c, catSize, l: [[[random.randint(0, catSize[j]-1) for j in range(c)], random.randint(0,l-1)] for i in range(n)]

"""
# A test:

import random
dataGen = lambda n, c, catSize, l: [[[random.randint(0, catSize[j]-1) for j in range(c)], random.randint(0,l-1)] for i in range(n)]
MDT = MultiDecisionTree()
data = dataGen(400, 4, [3,2,5,2], 4)
MDT.train(data, 4, [3,2,5,2], 4, preformatted=True)
drawTree(MDT.root, 0)

# In one line, for convenience:

import random; dataGen = lambda n, c, catSize, l: [[[random.randint(0, catSize[j]-1) for j in range(c)], random.randint(0,l-1)] for i in range(n)];MDT = MultiDecisionTree();data = dataGen(400, 4, [3,2,5,2], 4);MDT.train(data, 4, [3,2,5,2], 4, preformatted=True);drawTree(MDT.root, 0)
"""

class Leaf:
    """
    A leaf class for creating a tree.
    """
    def __init__(self, value=None):
        self.value = value

    def getValue(self):
        return self.value

    def setValue(self, value):
        self.value = value

        
class Node:
    """
    A node class for creating a tree.
    """
    def __init__(self, children=None, value=None):
        self.children = children
        self.value = value

    def getChildren(self, children):
        return self.children

    def setChildren(self, children):
        self.children = children

    def getChild(self, index):
        return self.children[index]

    def setChild(self, child, index):
        self.children[index] = child

    def getValue(self):
        return self.value

    def setValue(self, value):
        self.value = value



class MultiDecisionTree:
    """
    An implementation of a decision tree algorithm for multiclass classification
    of feature vectors (where features are an index into a finite category).

    The nodes of the tree store as their value the feature index which they split on.
    The children of each node correspond to the categories of the feature the node splits on.

    This class expects to be given funtions which convert the arbitrary values of input data into finite-size categories,
        whose members are represented by a finite prefix of the natural numbers.
        However, it is possible to preformat data into the correct format (described below) and pass "preformatted=True" on training.
        The expected functions are described below.

    self.featureCategoryFunction is a list of functions, one for each feature (passed to self.train()),
        where each function indexes feature values into integer category indexes for the feature corresponding to its index.
    self.labelCategoryFunction is a function which maps label values to their category index.
    The self.categorizeData() function is used to convert an input training set into pairs of integer
        feature vectors and integer label.
    """
    def __init__(self):
        self.root = None
        self.numFeatures = None
        self.featureCategorySizes = None
        self.featureCategoryFunction = None
        self.labelCategoryFunction = None

    def train(self, data, numFeatures, featureCategorySizes, labelCategorySize, featureCategoryFunction=None, labelCategoryFunction=None, preformatted=True):
        """
        Trains the decision tree on training data.
        
        data: a list of (point, label) pairs.
            point: vector of features (not in categories).
            label: Integer

        numFeatures: an integer storing the number of features
        
        featureCategorySizes: a list or tuple containing the size of each category.
        
        labelCategorySize: an integer storing the number of label categories.
        
        featureCategoryFuntions: a list of functions, each mapping the values of the
            input feature corresponding to its index to feature category.

        labelCategoryFunction: a function mapping label values to their category indices.

        Returns None.
        """
        if preformatted == False and not all([featureCategoryFunction, labelCategoryFunction]):
            raise Exception("Not preformatted, and no formatting functions specified!")

        self.featureCategorySizes = featureCategorySizes
        self.labelCategorySize = labelCategorySize
        self.numFeatures = numFeatures

        featuresList = list(range(numFeatures)) # A list of all feature indices
        
        if preformatted:
            self.root = self.trainAux(data, featuresList)
        else:
            self.featureCategoryFunction = featureCategoryFunction
            self.labelCategoryFunction = labelCategoryFunction
            processedData = self.categorizeData(data)
            self.root = self.trainAux(processedData, featuresList)

        

    def trainAux(self, data, remainingFeatures):
        """
        Auxiliary function for training the data.
        Recursively finds best feature for prediction, then trains on remaining features
    

        data: a list of (point, label) pairs.
            point: Boolean vector of features.
            label: Boolean.

        remainingFeatures: A list containing the indices of the features left to test on.

        Returns a decision tree for remaining features
        """

        ### Check for trivial cases: labels all agree or no labels left to train on
        ###
        
        test = self.labelsAllSame(data)
        if test[0] or len(remainingFeatures) == 0:
            return Leaf(test[1])


        ### Find best label to predict on
        ###
        
        scores = [] # Stores pairs (f, s) where f is feature index and s is the score of f
        for f in remainingFeatures:
            scores.append((f, self.featureAccuracy(data, f)))
        pi = lambda x: x[1] # A projection map to help with finding the maximal second coordinate of scores

        maxFeature = max(scores, key = pi)[0] # Get a (there could be multiple) feature with the maximal score

        partitioned = self.splitOnFeature(data, maxFeature)


        ### Create and return subtree
        ###
        
        subtree = Node()

        # For each category in corresponding to maxFeature, makes a new list of remaining features.
        # Each list is identical, though they are non-linked copies.
        # They all consist of "remainingFeatures" minus the value of maxFeature.
        copies = []
        for i in range(self.featureCategorySizes[maxFeature]):
            copies.append([f for f in remainingFeatures if f != maxFeature])

        subtree.setChildren([self.trainAux(p, copy) for (p, copy) in zip(partitioned, copies)])
        subtree.setValue(maxFeature)
        
        return subtree

    
    def categorizePoint(self, point):
        """
        Converts a point from domain into an integer feature vector

        point: a feature vector.
        
        Returns the formatted vector.
        """
        return [self.featureCategoryFunctions[i](point[i]) for i in range(self.numFeatures)]

    def categorizeData(self, data):
        """
        Converts input data into a list of (integer feature vector, integer label) pairs.

        data: the input data.

        Returns the formatted data.
        """
        return tuple([(self.categorizePoint(d[0]), self.labelCategoryFunction(d[1]))] for d in data)
    
    def predict(self, point, preformatted=True):
        """
        Predicts label of data point.

        point: Boolean vector of features.

        Returns label, the predicted label of parameter "example".
        """
        if preformatted:
            return self.predictAux(point, self.root)
        else:
            return self.predictAux(self.categorizePoint(point), self.root)

    def predictAux(self, point, node):
        """
        Predicts label of data point based on subtree rooted at parameter "node".

        point: Boolean vector of features.

        Returns label, the predicted label of parameter "example".
        """
        if isinstance(node, Leaf):
            return node.getValue()
        else:
            featureIndex = node.getValue()
            return self.predictAux(point, node.getChild(point[featureIndex]))

    def labelsAllSame(self, data):
        """
        Determines whether labels of a subset of data all agree.
        Assumes data has been pre-processed by self.categorizeData().

        data: the list of data being tested.

        Returns a tuple (allSame, label).
            allSame:        Boolean:    True if all labels are the same, false otherwise.
            label:          Int: category index of most common label, regardless of whether all labels of the same.
                                    Defaults to first category (value of zero) in case where "data" is empty.
        """
        if len(data) == 0:
            return True, 0
        partitioned = self.splitOnLabel(data)

        numEachLabel = [len(p) for p in partitioned]

        maxNum = max(numEachLabel)
        maxInd = numEachLabel.index(maxNum)

        numNonzero = [n for n in numEachLabel if n > 0]
        
        if len(numNonzero) == 1:
            return True, maxInd
        elif len(numNonzero) > 1:
            return False, maxInd
        else:
            raise Exception("Shouldn't get here")

    def featureAccuracy(self, data, feature):
        """
        Calculates the accuracy, as proportion of examples correctly predicted,
        when using a particular feature to classify a subset of data.

        data: the list of examples that the feature's predictive accuracy is tested for.
        feature: the index of the particular feature that is being tested.

        Returns a float representing the proportion of examples that are classified correctly by the current feature.
        """
        
        partitioned = self.splitOnFeature(data, feature)
        # Get the number that would be correctly predicted (by simple majority "vote") for each element of the partition
        numPredictedInEachCategory = [self.numMajorityLabel(p) for p in partitioned]
        accuracy = sum(numPredictedInEachCategory)
        return float(accuracy)/float(len(data))

    def numMajorityLabel(self, data):
        """
        For a given list/tuple of data, finds the maximum number of data points that belong to any one label.
            Alternatively, can be stated as: for each possible label, find the number of elements in "data" with that label.
            Then, return the maximum of these numbers.
        Assumes data has been pre-processed by self.categorizeData().

        data: a list or tuple of (feature, label) pairs.

        Returns: An integer; the number of data points in the label category with the most data points.
        """
        numberInEachCategory = [len([d for d in data if d[1] == i]) for i in range(self.labelCategorySize)]
        return max(numberInEachCategory)
            
    def splitOnFeature(self, data, feature):
        """
        Partitions data on feature category.
        Assumes data has been pre-processed by self.categorizeData().

        data: a list or tuple of data to split.
        
        feature: the index of the particular feature to split on.

        Returns a tuple of data (d1, d2, ..., dn) where "di" is a tuple containing
            the subset of "data" belonging to category "i" of feature.
        """
        return tuple([tuple([d for d in data if d[0][feature] == i]) for i in range(self.featureCategorySizes[feature])])
    
    def splitOnLabel(self, data):
        """
        Partitions data on feature category.
        Assumes data has been pre-processed by self.categorizeData().

        data: a list or tuple of data to split.
        
        feature: the index of the particular feature to split on.

        Returns a tuple of data (d1, d2, ..., dn) where "di" is a tuple containing
            the subset of "data" belonging to category "i" of feature.
        """
        return tuple([tuple([d for d in data if d[1] == i]) for i in range(self.labelCategorySize)])

def drawTree(node, depth):
    """
    A quick function to draw trees
    """
    if isinstance(node, Leaf):
        print("    "*depth + str(node.getValue()))
    else:
        print("    "*depth + "N:" + str(node.getValue()))
        for i in range(len(node.children)):
            print("   "*depth + "----")
            drawTree(node.getChild(i), depth + 1)
