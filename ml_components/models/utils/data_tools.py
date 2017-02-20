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
