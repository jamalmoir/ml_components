import numpy as np


class KMeans(object):
    def __init__(self):
        self.centroids = None
        self.prev_centroids = None
        self.labels = None

    def _random_centroid(self, num_features, maxi, mini):
        """Generates a random centroid for a number of features."""
        return np.random.uniform(low=mini, high=maxi, size=(1, num_features))

    def _get_distance(self, p1, p2):
        """Returns the distance between points."""
        return np.linalg.norm(p1 - p2)

    def _get_closest_centroid(self, p):
        """Returns the closest centroid to a point."""
        closest = -1
        best_distance = -1

        for i, centroid in enumerate(self.centroids):
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
        """Calculates the new centroids."""
        cents = []

        for i, centroid in enumerate(self.centroids):
            cluster = X[self.labels == i]

            if len(cluster) <= 0:
                cents.append(self._random_centroid(num_features=X.shape[1], maxi=X.max(), mini=X.min()))
            else:
                cents.append(cluster.mean(axis=0))

        return cents

    def _terminate(self, i, max_iter):
        """Training termination condition."""
        if self.prev_centroids is None:
            return False

        no_cent_change = True

        for i, centroid in enumerate(self.centroids):
            prev_centroid = self.prev_centroids[i]
            if not np.array_equal(centroid, prev_centroid):
                no_cent_change = False

        max_iter_reached = i > max_iter

        return no_cent_change or max_iter_reached

    def train(self, X, k, max_iter=500):
        self.centroids = [self._random_centroid(num_features=X.shape[1], maxi=X.max(), mini=X.min())] * k

        i = 0

        while not self._terminate(i=i, max_iter=max_iter):
            self.labels = np.apply_along_axis(arr=X, func1d=self._get_closest_centroid, axis=1)
            self.prev_centroids = self.centroids
            self.centroids = self._recalculate_centroids(X)

            i += 1

        return X, self.labels
