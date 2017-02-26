import numpy as np

from ml_components.models.model import Clusterer


class KMeans(Clusterer):
    """Implements K-Means Clustering.

    A K-Means Clusterer.
    """
    def __init__(self, model=None):
        if model is None:
            self._centroids = None
            self._prev_centroids = None
            self._labels = None
        else:
            self._centroids = model['centroids']

    @property
    def model(self):
        """Return a model representing the clusters."""
        model = {
            'k': len(self._centroids),
            'centroids': self._centroids
        }

        return model

    def _random_centroid(self, num_features, maxi, mini):
        """Generate a random centroid for a number of features.

        Generate a random centroid for number of features between a given maximum and minimum.

        Parameters
        ~~~~~~~~~~
        num_features : int
            The number of features the data to create a centroid for has.
        maxi : float
            The maximum value a centroid should be generated within.
        mini : float
            The minimum value a centroid should be generated within.
        """
        return np.random.uniform(low=mini, high=maxi, size=(1, num_features))

    def _get_distance(self, p1, p2):
        """Return the distance between points.

        Return the distance between two given points.

        Paramters
        ~~~~~~~~~
        p1 : numpy.ndarray
            the coordinates of point 1.
        p2 : numpy.ndarray
            The coordinates of point 2.
        """
        return np.linalg.norm(p1 - p2)

    def _get_closest_centroid(self, p):
        """Return the closest centroid to a point.

        Return the closest centroid to a given point.

        Parameters
        ~~~~~~~~~~
        p : numpy.ndarray
            The coordinates of the point to find the closest centroid to.
        """
        closest = -1
        best_distance = -1

        for i, centroid in enumerate(self._centroids):
            if closest == -1:
                closest = i
                best_distance = self._get_distance(p1=p, p2=centroid)
            else:
                distance = self._get_distance(p1=p, p2=centroid)

                if distance < best_distance:
                    closest = i
                    best_distance = distance

        return closest

    def _recalculate_centroids(self, X):
        """Calculate the new centroids.

        Calculate the new centroids after a training iteration.

        Parameters
        ~~~~~~~~~
        X : numpy.ndarray
            A matrix of training data.
        """
        cents = []

        for i, centroid in enumerate(self._centroids):
            cluster = X[self._labels == i]

            if len(cluster) <= 0:
                cents.append(self._random_centroid(num_features=X.shape[1], maxi=X.max(), mini=X.min()))
            else:
                cents.append(cluster.mean(axis=0))

        return cents

    def _terminate(self, i, max_iter):
        """Training termination condition.

        The condition under which to terminate training.

        Parameters
        ~~~~~~~~~~
        i : int
            The iteration number.
        max_iter : int
            The maximum number of allowed training iterations.
        """
        if self._prev_centroids is None:
            return False

        no_cent_change = True

        for i, centroid in enumerate(self._centroids):
            prev_centroid = self._prev_centroids[i]
            if not np.array_equal(centroid, prev_centroid):
                no_cent_change = False

        max_iter_reached = i > max_iter

        return no_cent_change or max_iter_reached

    def train(self, X, k, max_iter=500):
        """Create clusters within given data.

        Create clusters within given data using K-Means clustering.

        Parameters
        ~~~~~~~~~~
        X : numpy.ndarray
            A matrix of data to cluster.
        k : int
            The number of clusters to create.
        max_iter : int
            The maximum number of allowed training iterations.
        """
        if X.ndim <= 1:
            X = np.atleast_2d(X).T

        self._centroids = [self._random_centroid(num_features=X.shape[1], maxi=X.max(), mini=X.min())] * k

        i = 0

        while not self._terminate(i=i, max_iter=max_iter):
            self._labels = np.apply_along_axis(arr=X, func1d=self._get_closest_centroid, axis=1)
            self._prev_centroids = self._centroids
            self._centroids = self._recalculate_centroids(X)

            i += 1

        return self.model, self._labels

    def get_labels(self, X):
        """Get labels for given data.

        Get labels for given data based on previously calculated centroids.

        Parameters
        ~~~~~~~~~~
        X : numpy.ndarray
            The data to get labels for.
        """
        if self._centroids is None:
            raise RuntimeError('No model present.')

        if X.ndim <= 1:
            X = np.atleast_2d(X).T

        labels = np.apply_along_axis(arr=X, func1d=self._get_closest_centroid, axis=1)

        return labels

