import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

from ml_components.models import neural_network

# ---Load Data--- #

#iris = datasets.load_iris()  # 3 classes
#X = iris.data
#y = iris.target

breast = datasets.load_breast_cancer()  # 2 classes
X = breast.data
y = breast.target

# ---Split Into Training and Testing Data--- #
np.random.seed(0)
indices = np.random.permutation(len(X))
train_size = np.round(X.shape[0] / 100 * 15)
X_train = X[indices[:-train_size]]
y_train = y[indices[:-train_size]]
X_test = X[indices[-train_size:]]
y_test = y[indices[-train_size:]]

# ---Create Neural Network--- #
nn = neural_network.NeuralNetwork(X=X_train, y=y_train, classes=2, hidden_layer_size=400, lam=0.01,
                                  activation_func='tanh')
# ---Train Neural Network--- #
cost, costs, model = nn.train(alpha=0.01, max_epochs=1000, print_cost=True)

# ---Create Neural Network From Previously Trained Model--- #
nn2 = neural_network.NeuralNetwork(model=model)

# ---Predict Classes of Test Data and Calculate Accuracy--- #
pred = nn2.predict(X_test)
accuracy = np.sum(y_test == pred, axis=0) / X_test.shape[0] * 100

# ---Print Results--- #
print("Pred: {}".format(pred))
print("Accuracy: {}".format(accuracy))

# ---Plot the Cost Function During Training--- #
plt.plot(range(len(costs)), costs)
plt.show()
