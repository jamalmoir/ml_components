import operator

import numpy as np

from ml_components.models.model import Classifier
from ml_components.models.utils.data_tools import reduce_dimensions


class DecisionTree(Classifier):
    """Implements a decision tree.

    Implements a decision tree utilising the ID3 algorithm for training.
    """
    def __init__(self, num_dims=3, model=None):
        if model is None:
            self._start_node = None
            self._num_dims = num_dims
            self._kmeans_models = []
        else:
            self._start_node = Node('attribute')
            self._start_node.build_from_model(model['structure'])
            self._depth = model['depth']
            self._max_breadth = model['max_breadth']
            self._num_dims = model['num_dims']
            self._kmeans_models = model['kmeans_models']

    @property
    def model(self):
        """Return a model representing the tree."""
        model = {
            'depth': self._depth,
            'max_breadth': self._max_breadth,
            'structure': self._start_node.get_model(),
            'num_dims': self._num_dims,
            'kmeans_models': self._kmeans_models,
        }

        return model

    def train(self, X, y, max_iter=100):
        """Train a Decision Tree.

        Train a decision tree using the ID3 algorithm.

        Parameters
        ~~~~~~~~~~
        X : numpy.ndarray
            A matrix of training data.
        y : numpy.ndarray
            A vector of expected outputs.
        max_iter : int
            The maximum number of training iterations.
        """
        X, self._kmeans_models = reduce_dimensions(X=X, num_dimensions=self._num_dims)

        self._start_node = Node('attribute')
        self._start_node.split(X=X, y=y, max_iter=max_iter)

        get_depth = lambda alist: isinstance(alist, list) and max(map(get_depth, alist)) + 1

        class_count = X.shape[1]
        self._depth = get_depth(self._start_node.get_model())
        self._max_breadth = (self._depth - 1) ** class_count

        return self.model

    def predict(self, X):
        """Make a prediction with a trained tree.

        Make a prediction for a set of unseen data using the trained tree.

        Parameters
        ~~~~~~~~~~
        X : numpy.ndarray
            The data to make a prediction from.
        """
        data = reduce_dimensions(X=X, models=self._kmeans_models)
        prediction = np.apply_along_axis(self._start_node.predict, axis=1, arr=data, num_dims=self._num_dims)

        return prediction


class Node(object):
    """Implements a node in a Decision Tree."""

    def __init__(self, node_type):
        self._node_type = node_type
        self._attribute = None
        self._child_nodes = []
        self._output = None

    def _get_entropy(self, y):
        """Get the entropy of an attribute in the node.

        Calculate the entropy of an attribute in the current node.

        Parameters
        ~~~~~~~~~~
        y : np.ndarray
            A vector of expected outputs.
        """
        mask = ~np.isnan(y)
        class_counts = np.bincount(y[mask])
        pv = class_counts[np.nonzero(class_counts)] / len(y)
        entropy = np.sum(-pv * np.log2(pv))

        return entropy

    def _get_information_gain(self, X, y, attribute):
        """Get the information gain for a given split on an attribute.

        Calculate the information gain for a given split on an attribute in the current node.

        Parameters
        ~~~~~~~~~~
        X : numpy.ndarray
            A matrix of training data.
        y : numpy.ndarray
            A vector of expected outputs.
        attribute : int
            The attribute to make a split on.
        """
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
        """Get the attribute.

        Calculate the 'best' attribute with the most information gain.

        Parameters
        ~~~~~~~~~~
        X : numpy.ndarray
            A matrix of training data.
        y : numpy.ndarray
            A vector of expected output values."""
        attributes = X.shape[1] if len(X.shape) == 2 else X.shape[0]
        gains = []

        for attribute in range(attributes):
            gains.append(self._get_information_gain(X, y, attribute))

        best, _ = max(enumerate(gains), key=operator.itemgetter(1))

        return best

    def _one_class(self, data):
        """Check to see if the data has only one class in.

        Checks to see if the condition 'data has only one class present' is True.

        Parameters
        ~~~~~~~~~~
        data : np.ndarray
            The data to test.
        """
        comparison_class = data[0][-1]

        return np.all(data[:, -1] == comparison_class)

    def split(self, X, y, max_iter, i=0):
        """Split node on the best attribute.

        Split current node on the attribute with the most information gain, create children and recursively split child
        nodes, producing a tree if current node has more than one class in. Otherwise make leaf node.

        Parameters
        ~~~~~~~~~~
        X : numpy.ndarray
            A matrix of training data.
        y : numpy.ndarray
            A vector of expected outputs.
        max_iter : int
            The maximum number of training iterations.
        i : int
            The current iteration.
        """
        self._attribute = self._get_best_attribute(X, y)
        values = list(set(X[:, self._attribute]))

        for value in values:
            mask = X[:, self._attribute] == value
            new_X = X[mask, :]
            new_y = y[mask]

            if self._get_entropy(new_y) == 0 or i >= max_iter:
                new_node = Node('leaf')
                new_node._output = new_y[0]
            else:
                new_node = Node('attribute')
                new_node.split(X=new_X, y=new_y, max_iter=max_iter, i=i + 1)

            self._child_nodes.append(new_node)

    def predict(self, X, num_dims):
        """Make a prediction with a trained node.

        Make a prediction for a set of unseen data using the trained node.

        Parameters
        ~~~~~~~~~~
        X : numpy.ndarray
            The data to make a prediction from.
        num_dims : int
            The number of features in the data.
        """
        if self._node_type == 'leaf':
            return self._output
        else:
            attribute_value = X[self._attribute]
            try:
                return self._child_nodes[attribute_value].predict(data=X, num_dims=num_dims)
            except IndexError:
                random_branch = np.random.randint(0, len(self._child_nodes))
                return self._child_nodes[random_branch].predict(data=X, num_dims=num_dims)

    def get_model(self):
        """Return the node and child node's model."""
        if self._node_type == 'leaf':
            model = [-1, self._output]
        else:
            model = [self._attribute]

            for i in range(len(self._child_nodes)):
                model.append(self._child_nodes[i].get_model())

        return model

    def build_from_model(self, model):
        """Build node and child nodes from a model.

        Build node and child nodes from the model of a previously trained tree.

        Parameters
        ~~~~~~~~~~
        model : dict
            The model to build from."""
        if model[0] == -1:
            self._node_type = 'leaf'
            self._output = model[1]
        else:
            self._node_type = 'attribute'
            self._attribute = model[0]

            for node_model in model[1:]:
                self._child_nodes.append(Node('attribute'))
                self._child_nodes[-1].build_from_model(node_model)
