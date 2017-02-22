import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../../')

import ml_components.models.k_means as k_means

from sklearn import datasets


iris = datasets.load_iris()
data = iris.data

km = k_means.KMeans()
_, labels = km.train(X=data, k=3, max_iter=1000)

print(labels)

