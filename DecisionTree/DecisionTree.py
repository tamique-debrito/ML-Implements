class Node:
    """
    A node class for creating a tree.
    """
    def __init__(self, left=None, right=None, value=None):
        self.left = left
        self.right = right
        self.value = value

    def getLeft(self):
        return self.left

    def getRight(self):
        return self.right

    def setLeft(self, left):
        self.left = left

    def setRight(self, right):
        self.right = right

    def getValue(self):
        return self.value

    def setValue(self, value):
        self.value = value


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


class DecisionTree:
    """
    An implementation of a basic decision tree algorithm for binary classification
    of Boolean feature vectors.

    The nodes of the tree store as their value the feature index which they split on.
    Left subtree corresponds to feature being False; right subtree corresponds to feature being True.
    """
    def __init__(self):
        self.root = None
        self.numFeatures = None

    def train(self, data, numFeatures):
        """
        Trains the decision tree on training data.
        
        data: a list of (point, label) pairs.
            point: Boolean vector of features.
            label: Boolean.

        Returns None.
        """
        self.numFeatures = numFeatures
        featuresList = list(range(numFeatures)) # A list of all feature indices
        self.root = self.trainAux(data, featuresList)

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

        yes, no = self.splitOnFeature(data, maxFeature)


        ### Create and return subtree
        ###
        
        subtree = Node()

        # Makes two (2) copies of remainingFeatures, expect takes out the current feature, one for each recursive call
        remFeatCopy1 = [f for f in remainingFeatures if f != maxFeature]
        remFeatCopy2 = [f for f in remainingFeatures if f != maxFeature]

        subtree.setLeft(self.trainAux(no, remFeatCopy1))
        subtree.setRight(self.trainAux(yes, remFeatCopy2))
        subtree.setValue(maxFeature)
        
        return subtree

    def evaluate(self, point):
        """
        Predicts label of data point.

        point: Boolean vector of features.

        Returns label, the predicted label of parameter "example".
        """
        return self.evaluateAux(point, self.root)

    def evaluateAux(self, point, node):
        """
        Predicts label of data point based on subtree rooted at parameter "node".

        point: Boolean vector of features.

        Returns label, the predicted label of parameter "example".
        """
        if isinstance(node, Leaf):
            return node.getValue()
        else:
            if point[node.getValue()] == False:
                return self.evaluateAux(point, node.getLeft())
            else:
                return self.evaluateAux(point, node.getRight())

    def labelsAllSame(self, data):
        """
        Determines whether labels of a subset of data all agree.

        data: the list of data being tested.

        Returns a tuple (allSame, label).
            allSame:        Boolean:    True if all labels are the same, false otherwise.
            label:          Boolean:    If all labels are the same, is the value (True/False) of all labels.
                                                If labels not all same, truth value of most common label
        """
        length = len(data)
        numTrue = len([d for d in data if d[1] == True])
        numFalse = length - numTrue
        if numTrue == length:
            return True, True
        elif numFalse == length:
            return True, False
        else:
            return False, True if numTrue > numFalse else False

    def featureAccuracy(self, data, feature):
        """
        Calculates the accuracy, as proportion of examples correctly predicted,
        when using a particular feature to classify a subset of data.

        data: the list of examples that the feature's predictive accuracy is tested for.
        feature: the index of the particular feature that is being tested.

        Returns a float representing the proportion of examples that are classified correctly by the current feature.
        """
        # Subset of features where given feature is "no"
        no = [d for d in data if d[0][feature] == False]
        # Subset of features where given feature is "no"
        yes = [d for d in data if d[0][feature] == True]
        lenNo = len(no); lenYes = len(yes)
        # Number of "True" labels in "no" set and "yes set respectively
        numPredictedByNo = len([d for d in no if d[1] == True])
        numPredictedByYes = len([d for d in yes if d[1] == True])
        
        return float(max(numPredictedByNo, lenNo - numPredictedByNo) + max(numPredictedByYes, lenYes - numPredictedByYes)) / float(lenNo + lenYes)

    def splitOnFeature(self, data, feature):
        """
        Splits the data into two sets based on whether the value of their feature indicated by the parameter "feature" is True or False.

        data: a list of data to split.
        feature: the particular feature to split on.

        Returns a tuple (yes, no) which are lists of the subsets of data such that the value of "feature" is yes/no respectively.
        """
        return [d for d in data if d[0][feature] == True], [d for d in data if d[0][feature] == False]

def drawTree(node, depth):
    """
    A quick function to draw trees
    """
    if isinstance(node, Leaf):
        print("   "*depth + str(node.getValue()))
    else:
        drawTree(node.getLeft(), depth + 1)
        print("   "*depth + "/")
        print("   "*depth + "\\")
        drawTree(node.getRight(), depth + 1)
