import sys
import os
from sklearn import datasets

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../../')

import numpy as np

from ml_components.models import decision_tree

# ---Test Data--- #
X_cat = np.array([[2, 2, 2, 0],
                 [2, 2, 2, 1],
                 [1, 2, 2, 0],
                 [0, 1, 2, 0],
                 [0, 0, 2, 0],
                 [0, 0, 2, 1],
                 [1, 0, 2, 1],
                 [2, 1, 2, 0],
                 [2, 0, 2, 0],
                 [0, 1, 2, 0],
                 [2, 1, 2, 1],
                 [1, 1, 2, 1],
                 [1, 2, 2, 0],
                 [0, 1, 2, 1]])

y_cat = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])

data_test = np.array([[2, 2, 2, 0],
                      [1, 0, 2, 1],
                      [1, 1, 1, 1],
                      [2, 1, 2, 0],
                      [0, 0, 0, 0]])

iris = datasets.load_iris()
X = iris.data
y = iris.target

# ---Split Into Training and Testing Data--- #
np.random.seed(0)
indices = np.random.permutation(len(X))
train_size = int(np.round(X.shape[0] / 100 * 15))
X_train = X[indices[:-train_size]]
y_train = y[indices[:-train_size]]
X_test = X[indices[-train_size:]]
y_test = y[indices[-train_size:]]

# ---Create Decision Tree--- #
dt = decision_tree.DecisionTree()

# ---Train Decision Tree--- #
model = dt.train(X_train, y_train)

# ---Create Decision Tree From Previously Trained Model--- #
dt2 = decision_tree.DecisionTree(model)

# ---Predict Classes of Test Data and Calculate Accuracy--- #
pred = dt2.predict(X_test)
accuracy = np.sum(y_test == pred, axis=0) / X_test.shape[0] * 100

# ---Print Results--- #
print("Pred: {}".format(pred))
print("Accuracy: {}".format(accuracy))
