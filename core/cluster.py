from sys import platform

from scipy import clip
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

def hac(vecs, metric, linkage_method, threshold):
    """
    Hierarchical Agglomerative Clustering.
    """

    if platform != 'darwin':
        # This breaks on OSX 10.9.4, py3.3+, with large arrays:
        # https://stackoverflow.com/questions/11662960/ioerror-errno-22-invalid-argument-when-reading-writing-large-bytestring
        # https://github.com/numpy/numpy/issues/3858

        # This returns the distance matrix in squareform,
        # we use squareform() to convert it to condensed form, which is what linkage() accepts.
        # n_jobs=-2 to use all cores except 1.
        distance_matrix = pairwise_distances(vecs, metric=metric, n_jobs=-2)
        distance_matrix = squareform(distance_matrix, checks=False)

    else:
        distance_matrix = pdist(vecs, metric=metric)

    linkage_matrix = linkage(distance_matrix, method=linkage_method, metric=metric)

    # Floating point errors with the cosine metric occasionally lead to negative values.
    # Round them to 0.
    linkage_matrix = clip(linkage_matrix, 0, np.amax(linkage_matrix))

    labels = fcluster(linkage_matrix, threshold, criterion='distance')

    return labels
