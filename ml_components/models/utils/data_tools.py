import numpy as np

from ml_components.models import k_means


def make_label_map(y):
    """Return a dict mapping labels to integers starting from 0.

    Maps given labels to a range of integers starting from 0 and store these mappings in a dictionary.

    Parameters
    ~~~~~~~~~~
    y : numpy.ndarray
        A vector of expected outputs.
    """
    label_map = {}

    for i, label in enumerate(list(set(y))):
        label_map[label] = i

    return label_map


def reduce_dimensions(X, num_dimensions=None, models=None):
    """Reduces the dimensionality of a set of features.

    Reduces the dimensionality of a continuous or higher dimension set of features to the specified
    number of dimensions.

    Parameters
    ~~~~~~~~~~
    X : numpy.ndarray
        The features to be reduced.
    num_dimensions: int
        The number of dimensions to reduce to.
    models: list
        The K-Means models for each feature.
    """
    new_X = None

    if num_dimensions is not None:
        models = []

    if X.ndim <= 1:
        X = np.atleast_2d(X).T

    for i in range(X.shape[1]):
        feature = X[:, i]

        if num_dimensions is not None:
            reduced = reduce_dimension(X=feature, num_dimensions=num_dimensions)
        else:
            reduced = reduce_dimension(X=feature, model=models[i])

        if new_X is None:
            new_X = reduced['labels']
        else:
            new_X = np.c_[new_X, reduced['labels']]

        result = new_X

        if num_dimensions is not None:
            models.append(reduced['model'])
            result = result, models

    return result


def reduce_dimension(X, num_dimensions=None, model=None):
    """Reduces the dimensionality of a feature.

    Reduces the dimensionality of a continuous or higher dimension feature to the s
    number of dimensions.

    Parameters
    ~~~~~~~~~~
    X : numpy.ndarray
        The 1D feature to be reduced.
    num_dimensions: int
        The number of dimensions to reduce to.
    model : dict
        The K-Means model for the feature.
    """

    if num_dimensions is not None:
        km = k_means.KMeans()
        model, labels = km.train(X=X, k=num_dimensions)

        return {'model': model, 'labels': labels}
    else:
        km = k_means.KMeans(model=model)

        return {'labels': km.get_labels(X)}
