import numpy as np
import pandas as pd

from ml_components.models.model import Classifier
import ml_components.models.utils.activation_functions as af
import ml_components.models.utils.data_tools as data_tools


class NeuralNetwork(Classifier):
    """Implements a neural network.

    A Forward feeding, back propagation Artificial Neural Network.
    """

    def __init__(self, model=None, hidden_layer_size=None, activation_func=None):
        """Initiate Neural Network.

        Initiates a Neural Network with the provided parameters.

        Parameters
        ~~~~~~~~~~
        hlayer_node_count : int
            The number of nodes in each hidden layer.
        activation_func : str
            The activation function to be used. (sigmoid/tanh)
        """
        if hidden_layer_size is None and model is None:
            raise RuntimeError('Please supply a model or initialisation parameters')

        if model is None:
            self._hidden_layer_size = hidden_layer_size

            if activation_func == 'sigmoid' or activation_func is None:
                self._activation_func = af.sigmoid
                self._gradient_func = af.d_sigmoid
            elif activation_func == 'tanh':
                self._activation_func = af.tanh
                self._gradient_func = af.d_tanh

            self._m = 0  # m training examples.
            self._input_layer_size = 0  # How many nodes in the input layer
            self._costs = []  # A list of costs accumulated during training.
            self._adaptive = 0  # The value used in adaptive learning.
            self._prev_theta1_delta = 0  # The previous value of theta1_delta during training.
            self._prev_theta2_delta = 0  # The previous value of theta2_delta during training.
            self._label_map = {}  # A dictionary mapping labels to integers starting from 0.
        else:
            self._theta1 = model['theta1']
            self._theta2 = model['theta2']
            self._activation_func = model['activation_func']
            self._input_layer_size = model['input_layer_size']
            self._hidden_layer_size = model['hidden_layer_size']
            self._output_layer_size = model['output_layer_size']
            self._label_map = model['label_map']

    @property
    def model(self):
        """Return a model representing the Network."""
        return {
            'theta1': self._theta1,
            'theta2': self._theta2,
            'activation_func': self._activation_func,
            'input_layer_size': self._input_layer_size,
            'hidden_layer_size': self._hidden_layer_size,
            'output_layer_size': self._output_layer_size,
            'label_map': self._label_map,
        }

    def _init_epsilon(self):
        """Initialise the value of epsilon.

        Initialise the value of epsilon to be
        sqrt(6)/(sqrt(input_nodes) + sqrt(output_nodes)).
        """
        self._epsilon = (np.sqrt(6)) / (np.sqrt(self._input_layer_size) +
                                        np.sqrt(self._output_layer_size))

    def _init_weights(self):
        """Initialise the Network's weights.

        Initialise weights of input layer and each hidden layer to random
        values between negative epsilon and epsilon.
        """
        # Input layer to first hidden layer weights.
        self._theta1 = np.random.uniform(
            low=-self._epsilon,
            high=self._epsilon,
            size=(self._hidden_layer_size, (self._input_layer_size + 1))
        )

        # Hidden layer to output layer weights.
        self._theta2 = np.random.uniform(
            low=-self._epsilon,
            high=self._epsilon,
            size=(self._output_layer_size, (self._hidden_layer_size + 1))
        )

    def _forward_feed(self, X) -> np.ndarray:
        """Propagate forward through the Network.

        Propagates forward through the network returning the output given by the output layer.

        Parameters
        ~~~~~~~~~~
        X : numpy.ndarray
            A matrix of training data.
        """
        self._a1 = np.insert(X, 0, 1, axis=1)  # Add input layer with bias.

        self._z2 = self._theta1.dot(self._a1.T)  # Calculate input to hidden layer.
        self._a2 = np.insert(self._activation_func(self._z2), 0, 1, axis=0)  # Add hidden layer activation.

        self._z3 = self._theta2.dot(self._a2)  # Calculate input to output layer.
        self._a3 = self._activation_func(self._z3)  # Add output layer activation.

        return self._a3

    def _get_cost(self, y, lam):
        """Calculate the cost of the Network.

        Calculate the cost of the Network with current weights.

        Parameters
        ~~~~~~~~~~
        y : numpy.ndarray
            The target values.
        lam : float
            The regularisation strength.
        """

        # Normalise a3 with softmax if using hyperbolic tangent activation function.
        a3 = np.exp(self._a3) / self._a3.sum() if self._activation_func == af.tanh else self._a3

        J = (-1 / self._m) * np.sum(np.log(a3.T) * y + np.log(1 - a3).T * (1 - y))
        regulator = (lam / (2 * self._m)) * np.sum(np.square(self._theta1[:, 1:])) + np.sum(
            np.square(self._theta2[:, 1:]))
        J += regulator

        return J

    def _calculate_deltas(self, y, lam):
        """Work out delta values for each node.

        Calculates the error (delta) for each output and hidden layer node in the network.

        Parameters
        ~~~~~~~~~~
        y : numpy.ndarray
            The target values.
        lam : float
            The regularisation strength.
        """
        err3 = self._a3 - y.T  # Calculate delta for output layer.
        err2 = self._theta2[:, 1:].T.dot(err3) * self._gradient_func(self._z2)  # Calculate delta for hidden layer.

        self._delta2 = err2.dot(self._a1) / self._m + ((self._theta1 * lam) / self._m)  # + regularisation
        self._delta3 = err3.dot(self._a2.T) / self._m + ((self._theta2 * lam) / self._m)  # + regularisation

        return self._delta3

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
        self._adaptive /= 1 + dec_amount * epoch

        theta1_delta = self._adaptive * self._delta2
        theta2_delta = self._adaptive * self._delta3

        self._theta1 -= theta1_delta + alpha * self._prev_theta1_delta
        self._theta2 -= theta2_delta + alpha * self._prev_theta2_delta

        self._prev_theta1_delta = theta1_delta
        self._prev_theta2_delta = theta2_delta

    def train(self, X, y, alpha=0.5, max_epochs=5000, lam=0.01, adaptive=0.001, dec_amount=0.00001, print_cost=False):
        """Train the Network.

        Trains the network using online learning.

        Parameters
        ~~~~~~~~~~
        X : numpy.ndarray
            A matrix of training data.
        y : numpy.ndarray
            A vector of expected outputs.
        alpha : float
            The learning rate.
        max_epochs : int
            The maximum number of epochs allowed.
        lam : float
            The regularisation strength.
        adaptive : float
            The value used in adaptive learning.
        dec_amount : float
            The decrease constant used for adaptive learning.
        print_cost: bool
            Whether or not to print the networks cost after each epoch.
        """
        self._m, self._input_layer_size = X.shape
        self._label_map = data_tools.make_label_map(y)
        self._output_layer_size = len(self._label_map)
        self._adaptive = adaptive

        self._init_epsilon()
        self._init_weights()

        y = pd.get_dummies(y).as_matrix()

        for i in range(max_epochs):
            self._forward_feed(X=X)
            self._calculate_deltas(y=y, lam=lam)
            self._update_weights(i, alpha, dec_amount)

            self._costs.append(self._get_cost(y=y, lam=lam))

            if print_cost:
                print(self._costs[i])

        return self._costs[-1], self._costs, self.model

    def predict(self, X, use_label_map=True):
        """Make a prediction with a trained network.

        Make a prediction for a set of unseen data using the trained network.

        Parameters
        ~~~~~~~~~~
        X : numpy.ndarray
            The data to make a prediciton from.
        use_label_map : bool
            Whether to use a label map or not.
        """
        a1 = np.insert(X, 0, 1, axis=1)  # Add input layer with bias.

        z2 = self._theta1.dot(a1.T)  # Calculate input to hidden layer.
        a2 = np.insert(self._activation_func(z2), 0, 1, axis=0)  # Add hidden layer activation.

        z3 = self._theta2.dot(a2)  # Calculate input to output layer.
        prediction = np.argmax(z3, axis=0)

        if use_label_map:
            mapped = [self._label_map[pred] for pred in prediction]
            prediction = mapped
      
        return np.asarray(prediction)
