import operator
from itertools import combinations
from functools import reduce

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster

import numpy as np

def hac(vecs, metric, linkage_method, threshold):
    """
    Hierarchical Agglomerative Clustering.
    """
    distance_matrix = pdist(vecs, metric=metric)
    linkage_matrix = linkage(distance_matrix, method=linkage_method, metric=metric)
    labels = fcluster(linkage_matrix, threshold, criterion='distance')

    return labels
