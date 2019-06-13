class Node:
    """
    A node class for creating a tree.
    """
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def left(self):
        return self.left

    def right(self):
        return self.right
    
class Leaf:
    """
    A leaf class for creating a tree.
    """
    def __init__(self, value=None):
        self.value = value

    def value(self):
        return self.value

class DecisionTree:
    """
    An implementation of a basic decision tree algorithm for binary classification
    of Boolean feature vectors.
    """
    def __init__(self):
        pass

    def train(self, data):
        """
        Trains the decision tree on training data.
        
        data: a list of (point, label) pairs.
            point: Boolean vector of features.
            label: Boolean.

        Returns None.
        """
        pass

    def evaluate(self, example):
        """
        Predicts label of example.

        example: Boolean vector of features.

        Returns None.
        """
        pass

    def labelsAllSame(self, subset):
        """
        Determines whether labels of a subset of data all agree.

        subset: the set of data being tested.

        Returns a pair of Booleans (allSame, label).
            allSame: True if all labels are the same, false otherwise.
            label: if all labels are the same, is the value of all labels.
                        No specification otherwise.
        """
        pass

    def featureAccuracy(self, subset, feature):
        """
        Calculates the accuracy, as proportion of examples correctly predicted,
        when using a particular feature to classify a subset of data.

        subset: the set of examples that the feature's predictive accuracy is tested for.
        feature: the index of the particular feature that is being tested.
        """
        pass
