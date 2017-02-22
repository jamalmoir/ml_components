import operator

import numpy as np

from ml_components.models.utils.data_tools import reduce_dimensions


class DecisionTree(object):
    """Implements a decision tree.

    Implements a decision tree utilising the ID3 algorithm for training.
    """

    def __init__(self, model=None):
        if model is None:
            self.start_node = None
            self.num_dims = 3
            self.kmeans_models = []
        else:
            self.model = model
            self.start_node = Node('attribute')
            self.start_node.build_from_model(model['structure'])
            self.depth = model['depth']
            self.max_breadth = model['max_breadth']
            self.num_dims = model['num_dims']
            self.kmeans_models = model['kmeans_models']

    def train(self, X, y, max_iter=100):
        X, self.kmeans_models = reduce_dimensions(X=X, num_dimensions=self.num_dims)

        self.start_node = Node('attribute')
        self.start_node.split(X=X, y=y, max_iter=max_iter)

        get_depth = lambda alist: isinstance(alist, list) and max(map(get_depth, alist)) + 1

        structure = self.start_node.get_model()
        class_count = X.shape[1]
        self.depth = get_depth(structure)
        self.max_breadth = (self.depth - 1) ** class_count

        self.model = {
            'depth': self.depth,
            'max_breadth': self.max_breadth,
            'structure': structure,
            'num_dims': self.num_dims,
            'kmeans_models': self.kmeans_models,
        }

        return self.model

    def predict(self, data):
        data = reduce_dimensions(X=data, models=self.kmeans_models)
        prediction = np.apply_along_axis(self.start_node.predict, axis=1, arr=data, num_dims=self.num_dims)

        return prediction


class Node(object):
    """Implements a node in a decision tree."""

    def __init__(self, node_type):
        self.node_type = node_type
        self.attribute = None
        self.child_nodes = []
        self.output = None

    def _get_entropy(self, y):
        """Get the entropy of an attribute in the node."""
        class_counts = np.bincount(y)
        pv = class_counts[np.nonzero(class_counts)] / len(y)
        entropy = np.sum(-pv * np.log2(pv))

        return entropy

    def _get_information_gain(self, X, y, attribute):
        """Get the information gain for a given split on an attribute."""
        values = list(set(X[:, attribute]))  # Get unique list of attribute values.
        new_sets = []

        for value in values:
            mask = X[:, attribute] == value
            new_X = X[mask, :]
            new_y = y[mask]

            new_sets.append((new_X, new_y))

        entropy_current = self._get_entropy(y)

        # Get the sum of the entropy of each data split.
        entropy_new = sum([(s[1].shape[0] / X.shape[0]) * self._get_entropy(s[1]) for s in new_sets])

        return entropy_current - entropy_new

    def _get_best_attribute(self, X, y):
        """Get the attribute with the most information gain."""
        attributes = X.shape[1] if len(X.shape) == 2 else X.shape[0]
        gains = []

        for attribute in range(attributes):
            gains.append(self._get_information_gain(X, y, attribute))

        best, _ = max(enumerate(gains), key=operator.itemgetter(1))

        return best

    def _one_class(self, data):
        """Check to see if the data has only one class in."""
        comparison_class = data[0][-1]

        return np.all(data[:, -1] == comparison_class)

    def split(self, X, y, max_iter, i=0):
        self.attribute = self._get_best_attribute(X, y)
        values = list(set(X[:, self.attribute]))

        for value in values:
            mask = X[:, self.attribute] == value
            new_X = X[mask, :]
            new_y = y[mask]

            if self._get_entropy(new_y) == 0 or i >= max_iter:
                new_node = Node('leaf')
                new_node.output = new_y[0]
            else:
                new_node = Node('attribute')
                new_node.split(X=new_X, y=new_y, max_iter=max_iter, i=i + 1)

            self.child_nodes.append(new_node)

    def predict(self, data, num_dims):
        if self.node_type == 'leaf':
            return self.output
        else:
            attribute_value = data[self.attribute]
            try:
                return self.child_nodes[attribute_value].predict(data=data, num_dims=num_dims)
            except IndexError:
                #print("Warning: Unexpected value ({}) given for attribute {}, terminating...".format(attribute_value,
                #                                                                                     self.attribute))
                random_branch = np.random.randint(0, len(self.child_nodes))
                return self.child_nodes[random_branch].predict(data=data, num_dims=num_dims)

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
