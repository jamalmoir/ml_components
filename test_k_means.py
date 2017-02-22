from sklearn import datasets

import ml_components.models.k_means as k_means

# ---Load Data--- #
iris = datasets.load_iris()  # 3 classes
X = iris.data

km = k_means.KMeans()
_, labels = km.train(X, 3, 1000)

print(labels)
