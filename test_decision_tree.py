from models.decision_tree import DecisionTree

import numpy as np

data = np.array([[2, 2, 2, 0, 0],
                 [2, 2, 2, 1, 0],
                 [1, 2, 2, 0, 1],
                 [0, 1, 2, 0, 1],
                 [0, 0, 2, 0, 1],
                 [0, 0, 2, 1, 0],
                 [1, 0, 2, 1, 1],
                 [2, 1, 2, 0, 0],
                 [2, 0, 2, 0, 1],
                 [0, 1, 2, 0, 1],
                 [2, 1, 2, 1, 1],
                 [1, 1, 2, 1, 1],
                 [1, 2, 2, 0, 1],
                 [0, 1, 2, 1, 0]])

data_test = np.array([[2, 2, 2, 0],
                      [1, 0, 2, 1],
                      [1, 1, 1, 1],
                      [2, 1, 2, 0],
                      [0, 0, 0, 0]])

dt = DecisionTree()
dt.train(data)
prediction = dt.predict(data_test)
print(prediction)

