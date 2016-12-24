import model
import numpy as np
import activation_functions as af


class NeuralNetwork(model.Model):
    """Implements a neural network.

    A Forward feeding, backpropagation Artificial Neural Network.
    """

    def __init__(self, X, y, label_count, hlayer_count, hlayer_node_count):
        self.X = X
        self.y = y
        self.m, self.input_count = X.shape
        self.label_count = label_count
        self.hlayer_count = hlayer_count
        self.hlayer_node_count = hlayer_node_count

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

        for l in range(self.hlayer_count):
            self.layer_weights.append(
                np.random.uniform(
                    low=-self.epsilon,
                    high=self.epsilon,
                    size=(self.hlayer_node_count, self.input_count + 1)
                )
            )

        self.layer_weights.append(
            np.random.uniform(
                low=-self.epsilon,
                high=self.epsilon,
                size=(1, self.hlayer_node_count + 1)
            )
        )

    def _forward_feed(self, a, i=1):
        """Propagate forward through the Network.

        Propagates the provided activation of layer i to the next.

        Parameters
        ----------
        a : numpy.ndarray
            The activation for layer i-1.
        i : int
            The current layer.
        """
        if i == self.hlayer_count + 1:  # + 1 assures propagiton to output layer.
            return a

        a = np.insert(a, 0, 1, axis=1)  # Add bias units.
        z = np.matmul(a, self.layer_weights[i].T)   # Multiply activations by weights.
        a_new = af.sigmoid(z)   # Find next layers activation.
        h_theta = self._forward_feed(a_new, i + 1)  # Recursively propagate through all layers.

        return h_theta
