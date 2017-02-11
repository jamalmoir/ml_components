import operator

import numpy as np

class DecisionTree(object):
    """Implements a decision tree.

    Implements a decision tree utilising the ID3 algorithm for training.
    """

    def __init__(self, model=None):
        if model is None:
            self.start_node = None
        else:
            self.start_node = Node('attribute')
            self.start_node.build_from_model(model)

    def train(self, data):
        self.start_node = Node('attribute')
        self.start_node.split(data)

        return self.start_node.get_model()

    def predict(self, data):
        prediction = np.apply_along_axis(self.start_node.predict, axis=1, arr=data)

        return prediction


class Node(object):
    """Implements a node in a decision tree."""

    def __init__(self, node_type):
        self.node_type = node_type
        self.attribute = None
        self.child_nodes = []
        self.output = None

    def _get_entropy(self, data):
        """Get the entropy of an attribute in the node."""
        '''entropy = 0
        total = data.shape[0]
        class_counts = np.bincount(data[:, -1])

        for count in class_counts:
            pv = count / total  # The probability of the current class.
            #class_entropy = 0 if pv == 0 else pv * np.log2(pv)
            #entropy -= class_entropy
            entropy = pv * np.log2(pv)'''

        class_counts = np.bincount(data[:, -1])
        pv = class_counts[np.nonzero(class_counts)] / len(data[:, -1])
        entropy = np.sum(-pv * np.log2(pv))

        return entropy

    def _get_information_gain(self, data, attribute):
        """Get the information gain for a given split on an attribute."""
        values = list(set(data[:, attribute]))  # Get unique list of attribute values.
        new_sets = []

        for value in values:
            mask = data[:, attribute] == value
            new_data = data[mask, :]

            new_sets.append(new_data)

        entropy_current = self._get_entropy(data)
        entropy_new = sum([(s.shape[0] / data.shape[0]) * self._get_entropy(s) for s in new_sets])  # Get the sum of the entropy of each data split.


        return entropy_current - entropy_new

    def _get_best_attribute(self, data):
        """Get the attribute with the most information gain."""
        attributes = data.shape[1] - 1 if len(data.shape) == 2 else data.shape[0] - 1
        gains = []

        for attribute in range(attributes):
            gains.append(self._get_information_gain(data, attribute))

        best, _ = max(enumerate(gains), key=operator.itemgetter(1))

        return best

    def _one_class(self, data):
        """Check to see if the data has only one class in."""
        comparison_class = data[0][-1]

        return np.all(data[:, -1] == comparison_class)

    def split(self, data, i=0):
        #print("NEW SPLIT with data:\n {}".format(data))
        self.attribute = self._get_best_attribute(data)
        values = list(set(data[:, self.attribute]))

        for value in values:
            mask = data[:, self.attribute] == value
            new_data = data[mask, :]

            if self._get_entropy(new_data) == 0:
                new_node = Node('leaf')
                new_node.output = new_data[0][-1]
                #print("LEAF: {}".format(new_data))
            else:
                #print("Creating new split at level {} on attribute {}".format(i, self.attribute))
                new_node = Node('attribute')
                new_node.split(new_data, i=i + 1)

            self.child_nodes.append(new_node)

    def predict(self, data):
        if self.node_type == 'leaf':
            return self.output
        else:
            attribute_value = data[self.attribute]
            try:
                return self.child_nodes[attribute_value].predict(data)
            except IndexError:
                print("Unexpected value ({}) given for attribute {}, terminating...".format(attribute_value, self.attribute))
                exit()

    def get_model(self):
        if self.node_type == 'leaf':
            model = [-1, self.output]
        else:
            model = [self.attribute]

            for i in range(len(self.child_nodes)):
                model.append(self.child_nodes[i].get_model())

        return model

    def build_from_model(self, model):
        if model[0] == -1:
            self.node_type = 'leaf'
            self.output = model[1]
        else:
            self.node_type = 'attribute'
            self.attribute = model[0]

            for node_model in model[1:]:
                self.child_nodes.append(Node('attribute'))
                self.child_nodes[-1].build_from_model(node_model)
