import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../../')

import unittest

import numpy as np
import ml_components.models.k_means as k_means

from sklearn import datasets
from ml_components.models import decision_tree, neural_network


def load_data(train_test=False):
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    if train_test:
        np.random.seed(0)
        indices = np.random.permutation(len(X))
        train_size = int(np.round(X.shape[0] / 100 * 15))
        X_train = X[indices[:-train_size]]
        y_train = y[indices[:-train_size]]
        X_test = X[indices[-train_size:]]
        y_test = y[indices[-train_size:]]

        return X_train, y_train, X_test, y_test

    return X, y


class TestKMeans(unittest.TestCase):
    def test_train_returns_correct_number_of_labels(self):
        X, y = load_data()
        expected = X.shape[0]
        km = k_means.KMeans()
        _, labels = km.train(X=X, k=3, max_iter=1000)
        result = len(labels)

        self.assertEqual(expected, result, msg='Did not return correct number of labels.')


class TestDecisionTree(unittest.TestCase):
    def test_train_returns_complete_model(self):
        X, y = load_data()
        dt = decision_tree.DecisionTree()
        model = dt.train(X=X, y=y)

        self.assertIsNotNone(model['depth'], msg='Key depth not present in model.')
        self.assertIsNotNone(model['max_breadth'], msg='Key max_breadth not present in model.')
        self.assertIsNotNone(model['structure'], msg='Key structure not present in model.')
        self.assertIsNotNone(model['num_dims'], msg='Key num_dims not present in model.')
        self.assertIsNotNone(model['kmeans_models'], msg='Key kmeans_models not present in model.')

    def test_predict_returns_correct_number_of_predictions(self):
        X, y = load_data()
        expected = y.size
        dt = decision_tree.DecisionTree()
        dt.train(X=X, y=y)
        result = dt.predict(X).size

        self.assertEqual(expected, result, msg='Did not return correct number of predictions.')

    def test_loading_from_model(self):
        X, y = load_data()
        expected = y.size
        dt = decision_tree.DecisionTree()
        model = dt.train(X, y)
        dt2 = decision_tree.DecisionTree(model)
        result = dt2.predict(X).size

        self.assertEqual(dt.model, dt2.model, msg='Original tree and tree loaded from model\'s models do not match.')
        self.assertEqual(expected, result, msg='Prediction failed.')

    def test_accuracy_above_50_percent(self):
        X_train, y_train, X_test, y_test = load_data(train_test=True)
        dt = decision_tree.DecisionTree()
        dt.train(X_train, y_train)
        pred = dt.predict(X_test)
        accuracy = np.sum(y_test == pred, axis=0) / X_test.shape[0] * 100

        self.assertGreaterEqual(accuracy, 50, msg='Model accuracy is below 50%.')


class TestNeuralNetwork(unittest.TestCase):
    def test_train_returns_complete_cost_costs_model(self):
        X, y = load_data()
        nn = neural_network.NeuralNetwork(hidden_layer_size=400, activation_func='tanh')
        cost, costs, model = nn.train(X=X, y=y, alpha=0.01, max_epochs=1000, lam=0.01)

        self.assertIsNotNone(cost, msg='Cost is None.')
        self.assertIsNotNone(costs, msg='Costs is None')
        self.assertIsNotNone(model['theta1'], msg='Key theta1 not present in model.')
        self.assertIsNotNone(model['theta2'], msg='Key theta2 not present in model.')
        self.assertIsNotNone(model['activation_func'], msg='Key activation_func not present in model.')
        self.assertIsNotNone(model['input_layer_size'], msg='Key input_layer_size not present in model.')
        self.assertIsNotNone(model['hidden_layer_size'], msg='Key hidden_layer_size not present in model.')
        self.assertIsNotNone(model['output_layer_size'], msg='Key output_layer_size not present in model.')
        self.assertIsNotNone(model['label_map'], msg='Key label_map not present in model.')

    def test_predict_returns_correct_number_of_predictions(self):
        X, y = load_data()
        expected = y.size
        nn = neural_network.NeuralNetwork(hidden_layer_size=400, activation_func='tanh')
        nn.train(X=X, y=y, alpha=0.01, max_epochs=1000, lam=0.01)
        result = nn.predict(X).size

        self.assertEqual(expected, result, msg='Did not return correct number of predictions.')

    def test_loading_from_model(self):
        X, y = load_data()
        expected = y.size
        nn = neural_network.NeuralNetwork(hidden_layer_size=400, activation_func='tanh')
        cost, costs, model = nn.train(X=X, y=y, alpha=0.01, max_epochs=1000, lam=0.01)
        nn2 = neural_network.NeuralNetwork(model)
        result = nn2.predict(X).size

        self.assertEqual(nn.model, nn2.model,
                         msg='Original network and network loaded from model\'s models do not match.')
        self.assertEqual(expected, result, msg='Prediction failed.')

    def test_accuracy_above_50_percent(self):
        X_train, y_train, X_test, y_test = load_data(train_test=True)
        nn = neural_network.NeuralNetwork(hidden_layer_size=400, activation_func='tanh')
        nn.train(X=X_train, y=y_train, alpha=0.01, max_epochs=1000, lam=0.01)
        pred = nn.predict(X_test)
        accuracy = np.sum(y_test == pred, axis=0) / X_test.shape[0] * 100

        self.assertGreaterEqual(accuracy, 50, msg='Model accuracy is below 50%.')
