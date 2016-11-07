import model
import numpy as np


class NeuralNetwork(model.Model):
    """Implements a neural network."""

    def __init__(self, input_node_count, output_node_count, hlayer_count,
                 hlayer_node_count, epsilon=0.0001):
        self.input_node_count = input_node_count
        self.ouput_node_count = output_node_count
        self.hlayer_count = hlayer_count
        self.hlayer_node_count = hlayer_node_count
        self.epsilon = 0.0001

        self._init_weights()

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
                    size=self.hlayer_node_count
                )
            )
