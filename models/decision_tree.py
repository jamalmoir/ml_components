import operator

import numpy as np

class DecisionTree(object):
    """Implements a decision tree.

    Implements a decision tree utilising the ID3 algorithm for training.
    """

    def __init__(self, data, num_classes):
        self.data = data
        self.num_classes = num_classes
        self.start_node = None

    def train(self):
        pass

    def predict(self):
        pass


class Node(object):
    """Implements a node in a decision tree."""

    def __init__(self, node_type, data, num_classes):
        self.node_type = node_type
        self.data = data
        self.num_classes = num_classes
        self.attribute = None
        self.child_nodes = []
        self.output = None

    def _get_class_count(self, s):
        """Get the count of each class appearing within the node."""
        class_counts = []

        for i in range(self.num_classes):
            count = (s[:, -1] == i).sum()
            class_counts.append(count)

        return class_counts

    def _get_entropy(self, s=None):
        """Get the entropy of an attribute in the node."""
        if s is None:
            s = self.data

        entropy = 0
        total = s.shape()[0]
        class_counts = self._get_class_count(s)

        for count in class_counts:
            pv = count / total  # The probability of the current class.
            entropy -= pv * np.log2(pv)

        return entropy

    def _get_information_gain(self, attribute):
        """Get the information gain for a given split on an attribute."""
        values = list(set(self.data[attribute]))  # Get unique list of attribute values.
        new_sets = []

        for value in values:
            new_sets.append(self.data[attribute] == value)

        entropy_current = self._get_entropy(self.data)
        entropy_new = sum([self._get_entropy(s) for s in new_sets])  # Get the sum of the entropy of each data split.

        return entropy_current - entropy_new

    def _get_best_attribute(self):
        """Get the attribute with the most information gain."""
        attributes = self.data.shape[1]
        gains = []

        for attribute in attributes:
            gains.append(self._get_information_gain(attribute))

        best, _ = max(enumerate(gains), key=operator.itemgetter(1))

        return best
