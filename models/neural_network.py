import numpy as np
import pandas as pd

import models.utils.activation_functions as af


class NeuralNetwork(object):
    """Implements a neural network.

    A Forward feeding, backpropagation Artificial Neural Network.
    """

    def __init__(self, model=None, X=None, y=None, classes=None, hidden_layer_size=None, lam=0.01,
                 activation_func=None):
        """Initiate Neural Network.

        Initiates a Neural Network with the provided parameters.

        Parameters
        ~~~~~~~~~~
        X : numpy.ndarray
            A matrix of training data.
        y : numpy.ndarray
            A vector of expected outputs.
        label_count : int
            The number of classes.
        hlayer_node_count : int
            The number of nodes in each hidden layer.
        lam : float
            The lambda value used for regularisation.
        activation_func : str
            The activation function to be used. (sigmoid/tanh)
        """
        if (X is None or y is None or classes is None or hidden_layer_size is None) and model is None:
            raise RuntimeError('Please supply a model or initialisation parameters')

        if model is None:
            self.X = X
            self.y = pd.get_dummies(y).as_matrix()
            self.label_count = classes
            self.hidden_layer_size = hidden_layer_size
            self.lam = lam

            if activation_func == 'sigmoid' or activation_func is None:
                self.activation_func = af.sigmoid
                self.gradient_func = af.d_sigmoid
            elif activation_func == 'tanh':
                self.activation_func = af.tanh
                self.gradient_func = af.d_tanh

            self.m, self.input_layer_size = X.shape  # m training examples, how many nodes in the input layer
            self.costs = []  # A list of costs accumulated during training.
            self.adaptive = 0  # The value used in adaptive learning.
            self.prev_theta1_delta = 0  # The previous value of theta1_delta during training.
            self.prev_theta2_delta = 0  # The previous value of theta2_delta during training.

            self._init_epsilon()
            self._init_weights()
        else:
            self.theta1 = model['theta1']
            self.theta2 = model['theta2']
            self.activation_func = model['activation_func']

    def _init_epsilon(self):
        """Initialise the value of epsilon.

        Initialise the value of epsilon to be
        sqrt(6)/(sqrt(input_nodes) + sqrt(output_nodes)).
        """
        self.epsilon = (np.sqrt(6)) / (np.sqrt(self.input_layer_size) +
                                       np.sqrt(self.label_count))

    def _init_weights(self):
        """Initialise the Network's weights.

        Initialise weights of input layer and each hidden layer to random
        values between negative epsilon and epsilon.
        """
        # Input layer to first hidden layer weights.
        self.theta1 = np.random.uniform(
            low=-self.epsilon,
            high=self.epsilon,
            size=(self.hidden_layer_size, (self.input_layer_size + 1))
        )

        # Hidden layer to output layer weights.
        self.theta2 = np.random.uniform(
            low=-self.epsilon,
            high=self.epsilon,
            size=(self.label_count, (self.hidden_layer_size + 1))
        )

    def _forward_feed(self) -> np.ndarray:
        """Propagate forward through the Network.

        Propagates forward through the network returning the output given by the output layer.
        """
        self.a1 = np.insert(self.X, 0, 1, axis=1)  # Add input layer with bias.

        self.z2 = self.theta1.dot(self.a1.T)  # Calculate input to hidden layer.
        self.a2 = np.insert(self.activation_func(self.z2), 0, 1, axis=0)  # Add hidden layer activation.

        self.z3 = self.theta2.dot(self.a2)  # Calculate input to output layer.
        self.a3 = self.activation_func(self.z3)  # Add output layer activation.

        return self.a3

    def _get_cost(self):
        """Calculate the cost of the Network.

        Calculate the cost of the Network with current weights."""

        # Normalise a3 with softmax if using hyperbolic tangent activation function.:w
        a3 = np.exp(self.a3) / self.a3.sum() if self.activation_func == af.tanh else self.a3

        J = (-1 / self.m) * np.sum(np.log(a3.T) * self.y + np.log(1 - a3).T * (1 - self.y))
        regulator = (self.lam / (2 * self.m)) * np.sum(np.square(self.theta1[:, 1:])) + np.sum(
            np.square(self.theta2[:, 1:]))
        J += regulator

        return J

    def _calculate_deltas(self):
        """Work out delta values for each node.

        Calculates the error (delta) for each output and hidden layer node in the network.
        """
        err3 = self.a3 - self.y.T  # Calculate delta for output layer.
        err2 = self.theta2[:, 1:].T.dot(err3) * self.gradient_func(self.z2)  # Calculate delta for hidden layer.

        self.delta2 = err2.dot(self.a1) / self.m + ((self.theta1 * self.lam) / self.m)  # + regularisation
        self.delta3 = err3.dot(self.a2.T) / self.m + ((self.theta2 * self.lam) / self.m)  # + regularisation

        return self.delta3

    def _update_weights(self, epoch, alpha, dec_amount):
        """Update the network weights.

        Updates the weights of the network using adaptive online learning for a given epoch, learning rate and decrease
        constant.

        Parameters
        ~~~~~~~~~~
        epoch : int
            The current epoch in the training loop.
        alpha : float
            The learning rate.
        dec_amount : float
            The decrease constant used for adaptive learning.
        """
        self.adaptive /= 1 + dec_amount * epoch

        theta1_delta = self.adaptive * self.delta2
        theta2_delta = self.adaptive * self.delta3

        self.theta1 -= theta1_delta + alpha * self.prev_theta1_delta
        self.theta2 -= theta2_delta + alpha * self.prev_theta2_delta

        self.prev_theta1_delta = theta1_delta
        self.prev_theta2_delta = theta2_delta

    def train(self, alpha=0.5, max_epochs=5000, adaptive=0.001, dec_amount=0.00001, print_cost=False):
        """Train the Network.

        Trains the network using online learning.

        Parameters
        ~~~~~~~~~~
        alpha : float
            The learning rate.
        max_epochs : int
            The maximum number of epochs allowed.
        adaptive : float
            The value used in adaptive learning
        dec_amount : float
            The decrease constant used for adaptive learning.
        print_cost: bool
            Whether or not to print the networks cost after each epoch.
        """
        self.adaptive = adaptive

        for i in range(max_epochs):
            self._forward_feed()
            self._calculate_deltas()
            self._update_weights(i, alpha, dec_amount)

            self.costs.append(self._get_cost())

            if print_cost:
                print(self.costs[i])

        model = {'theta1': self.theta1, 'theta2': self.theta2, 'activation_func': self.activation_func}

        return self.costs[-1], self.costs, model

    def predict(self, X):
        """Make a prediction with a trained network.

        Make a prediction for a set of unseen data using the trained network.

        Parameters
        ~~~~~~~~~~
        X : numpy.ndarray
            The data to make a prediciton from.
        """
        a1 = np.insert(X, 0, 1, axis=1)  # Add input layer with bias.

        z2 = self.theta1.dot(a1.T)  # Calculate input to hidden layer.
        a2 = np.insert(self.activation_func(z2), 0, 1, axis=0)  # Add hidden layer activation.

        z3 = self.theta2.dot(a2)  # Calculate input to output layer.
        prediction = np.argmax(z3, axis=0)

        return prediction
