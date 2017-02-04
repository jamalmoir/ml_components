import model
import numpy as np
import pandas as pd
import activation_functions as af


class NeuralNetwork(model.Model):
    """Implements a neural network.

    A Forward feeding, backpropagation Artificial Neural Network.
    """

    def __init__(self, X, y, label_count, hlayer_count, hlayer_node_count, alpha, momentum, activation_func=af.sigmoid):
        """Initiate Neural Network.

        Initiates a Neural Network with the provided parameters.

        Parameters
        ~~~~~~~~~~
        X : np.ndarray
            A matrix of training data.
        y : np.ndarray
            A vector of expected outputs.
        label_count : int
            The number of classes.
        hlayer_count : int
            The number of hidden layers.
        hlayer_node_count : int
            The number of nodes in each hidden layer.
        alpha : float
            The learning rate of the Neural Network.
        momentum : float
            The momentum used for training the Neural Network.
        """
        self.X = X
        self.y = pd.get_dummies(y).as_matrix()
        self.label_count = label_count
        self.hlayer_count = hlayer_count
        self.hlayer_node_count = hlayer_node_count
        self.alpha = alpha
        self.momentum = momentum
        self.activation_func = activation_func

        self.m, self.input_count = X.shape
        self.z = []
        self.a = []
        self.delta = [None] * (hlayer_count + 1)
        self.error = [0] * (hlayer_count + 1)
        self.bias_diff = []
        self.weight_diff = []

        self._init_epsilon()
        self._init_weights()

    def _init_epsilon(self):
        """Initialise the value of epsilon.

        Initilise the value of epsilon to be
        sqrt(6)/(sqrt(input_nodes) + sqrt(output_nodes)).
        """
        self.epsilon = (np.sqrt(6)) / (np.sqrt(self.input_count) +
                                       np.sqrt(self.label_count))

    def _init_weights(self):
        """Initialise the Network's weights.

        Initialise weights of input layer and each hidden layer to random
        values between negative epsilon and epsilon.
        """
        self.layer_weights = []

        # Input layer to first hidden layer weights.
        self.layer_weights.append(
            np.random.uniform(
                low=-self.epsilon,
                high=self.epsilon,
                size=(self.hlayer_node_count, (self.input_count + 1))
            )
        )

        # Hidden layer to hidden layer weights.
        for l in range(self.hlayer_count - 1):
            self.layer_weights.append(
                np.random.uniform(
                    low=-self.epsilon,
                    high=self.epsilon,
                    size=(self.hlayer_node_count, (self.hlayer_node_count + 1))
                )
            )

        # Hidden layer to output layer weights.
        self.layer_weights.append(
            np.random.uniform(
                low=-self.epsilon,
                high=self.epsilon,
                size=(self.label_count, (self.hlayer_node_count + 1))
            )
        )

    def _forward_feed(self):
        """Propagate forward through the Network.

        Propagates the provided activation of layer i to the next.

        Parameters
        ----------
        a : numpy.ndarray
            The activations for layer i - 1.
        i : int
            The current layer number.
        """
        self.a.append(np.insert(self.X, 0, 1, axis=1))  # Add input layer (a0) with bias.

        # Add hidden layer activations
        for l in range(1, self.hlayer_count + 1):
            self.z.append(self.layer_weights[l - 1].dot(self.a[l - 1].T))
            #z = self.a[l - 1].dot(self.layer_weights[l - 1].T)
            self.a.append(np.insert(self.activation_func(self.z[l - 1].T), 0, 1, axis=1))

        self.z.append(self.layer_weights[-1].dot(self.a[-1].T))
        #z = self.a[-1].dot(self.layer_weights[-1].T)
        self.a.append(self.activation_func(self.z[-1]))  # Add output layer activations.

        return self.a[-1]

    def _get_gradient(self, z):
        return self.activation_func(z) * (1 - self.activation_func(z))

    def _calculate_deltas(self):
        """Work out delta values for each node.

        Calculates the error (delta) for each output and hidden layer node in the network.
        """
        self.delta[self.hlayer_count] = self.a[-1].T - self.y

        for l in reversed(range(self.hlayer_count)):
            print(l)
            self.delta[l] = self.layer_weights[l+ 1][:, 1:].T.dot(self.delta[l + 1].T) * self._get_gradient(self.z[l])
            self.error[l] = self.delta[l].dot(self.a[l])

        return self.error

    def _compute_numerical_gradient(self):
        """Numerically compute the gradient.

        Compute the numerical gradient for gradient checking."""
        num_grad = []
        e = 1 * 10 ^ -4
        for l in range(self.hlayer_count + 1):
            layer_grads = []

            for w in range(len(self.layer_weights[l])):
                loss1 = self.activation_func(self.layer_weights[l][w] - e)
                loss2 = self.activation_func(self.layer_weights[l][w] + e)
                layer_grads.append((loss2 - loss1) / (2 * e))

            num_grad.append(np.asarray(layer_grads))

        return np.asarray(num_grad)
