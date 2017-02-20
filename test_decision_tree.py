import numpy as np

from ml_components.models import decision_tree

# ---Test Data--- #
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

# ---Create Decision Tree--- #
dt = decision_tree.DecisionTree()

# ---Train Decision Tree--- #
model = dt.train(data)

# ---Create Decision Tree From Previously Trained Model--- #
dt2 = decision_tree.DecisionTree(model)

# ---Predict Classes of Test Data---#
prediction = dt2.predict(data_test)

# ---Print model and prediction results--- #
print(model)
print(prediction)
print(dt2.max_breadth)
print(dt2.depth)

