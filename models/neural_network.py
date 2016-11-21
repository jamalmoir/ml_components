import model
import numpy as np


class NeuralNetwork(model.Model):
    """Implements a neural network."""

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
        """Initilise the value of epsilon to be
        sqrt(6)/(sqrt(input_nodes) + sqrt(output_nodes)).
        """
        self.epsilon = (np.sqrt(6)) / (np.sqrt(self.input_count) +
                                       np.sqrt(self.label_count))

    def _init_weights(self):
        """Initialise weights of input layer and each hidden layer to random
        values between negative epsilon and epsilon.
        """
        self.layer_weights = []

        for l in range(self.hlayer_count + 1):
            self.layer_weights.append(
                np.random.uniform(
                    low=-self.epsilon,
                    high=self.epsilon,
                    size=(self.hlayer_node_count, self.input_count + 1)
                )
            )
