import model
import numpy as np
import activation_functions as af


class NeuralNetwork(model.Model):
    """Implements a neural network.

    A Forward feeding, backpropagation Artificial Neural Network.
    """

    def __init__(self, X, y, label_count, hlayer_count, hlayer_node_count):
        self.X = X
        self.y = self._to_out_vec(y, label_count)
        self.m, self.input_count = X.shape
        self.a = []
        self.delta = [None] * (hlayer_count + 1)
        self.error = [0] * (hlayer_count + 1)
        self.label_count = label_count
        self.hlayer_count = hlayer_count
        self.hlayer_node_count = hlayer_node_count

        self._init_epsilon()
        self._init_weights()

    def _to_out_vec(self, y, labels):
        output_vector = []

        for row in y:
            new_row = np.zeros(labels)
            new_row[row] = 1
            output_vector.append(new_row)

        return output_vector

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

        self.layer_weights.append(
            np.random.uniform(
                low=-self.epsilon,
                high=self.epsilon,
                size=(self.hlayer_node_count, self.input_count + 1)
            )
        )

        for l in range(self.hlayer_count - 1):
            self.layer_weights.append(
                np.random.uniform(
                    low=-self.epsilon,
                    high=self.epsilon,
                    size=(self.hlayer_node_count, self.hlayer_node_count + 1)
                )
            )

        self.layer_weights.append(
            np.random.uniform(
                low=-self.epsilon,
                high=self.epsilon,
                size=(self.label_count, self.hlayer_node_count + 1)
            )
        )

    def _forward_feed(self, x, i=0):
        """Propagate forward through the Network.

        Propagates the provided activation of layer i to the next.

        Parameters
        ----------
        a : numpy.ndarray
            The activations for layer i - 1.
        i : int
            The current layer number.
        """
        a_bias = np.insert(x, 0, 1, axis=1)  # Add bias units.
        z = np.dot(a_bias, self.layer_weights[i].T)   # Multiply activations by weights.

        self.a.append(af.sigmoid(z))   # Caluculate layers activations.

        if i == self.hlayer_count:
            return self.a[i]
        else:
            h_theta = self._forward_feed(self.a[i], i + 1)  # Recursively propagate through all layers.

        return h_theta

    def _back_propagate(self):
        """Backpropagate through the Network.

        Propagates the weight erros back through the network.
        """
        for t in range(self.m):
            self.delta[self.hlayer_count] = self.a[self.hlayer_count][t, :] - self.y[t]
            self.error[self.hlayer_count] = np.dot(self.delta[self.hlayer_count], self.a[self.hlayer_count].T)

            for l in reversed(range(self.hlayer_count)):
                self.delta[l] = np.dot(self.layer_weights[l + 1].T, self.delta[l + 1]) * np.insert(self.a[l], 0, 1, axis=1)[t, :]
                self.delta[l] = self.delta[l][1:]
                self.error[l] += np.dot(self.delta[l], self.a[l].T)

        return self.error
