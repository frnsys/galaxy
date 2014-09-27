from sys import platform

from scipy import clip
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from .ihacluster import IHAClusterer

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
        #distance_matrix = pairwise_distances(vecs, metric=metric, n_jobs=-1)
        #distance_matrix = squareform(distance_matrix, checks=False)

        # Just using pdist for now because it seems way faster.
        # Possible that the multicore gains aren't seen until you have many cores going.
        # Also, pdist stores an array in memory that can be very huge, which is why
        # memory constrains the core usage.
        distance_matrix = pdist(vecs, metric=metric)
    else:
        distance_matrix = pdist(vecs, metric=metric)

    linkage_matrix = linkage(distance_matrix, method=linkage_method, metric=metric)

    # Floating point errors with the cosine metric occasionally lead to negative values.
    # Round them to 0.
    linkage_matrix = clip(linkage_matrix, 0, np.amax(linkage_matrix))

    labels = fcluster(linkage_matrix, threshold, criterion='distance')

    return labels


def ihac(vecs, metric, previous_data_file=None):
    """
        Incremental Hierarchical Agglomerative Clustering.

        If some previous clustering results are passed
        it continues from there
    """
    if previous_data_file:
        old_distance_matrix, old_hierarchy, old_vecs = load_state_from_file(previous_data_file)
        distance_matrix = extend_distance_matrix(old_distance_matrix, old_vecs, vecs)
    else:
        old_hierarchy = None
        distance_matrix = pdist(vecs, metric=metric)

    # distance matrix is extending in advance to make
    # incorporation of points more efficient
    state = {"hierarchy": old_hierarchy, "distance_matrix": distance_matrix}
    clusterer = IHAClusterer(state=state)

    for vec in vecs:
        clusterer.incorporate(vec)

    hierarchy = clusterer.hierarchy

    labels = fcluster_from_hierarchy(hierarchy, threshold)
    # QUESTION: can we produce something similar to linkage matrix?
    # in case not, we need to implement our own fcluster, or convert or hierarchy structure
    # to a linkage matrix

    # linkage_matrix = ??????
    # labels = fcluster(linkage_matrix, threshold, criterion='distance')

    return labels



def load_state_from_file(previous_data_file):
    pass


def extend_distance_matrix(old_distance_matrix, old_vecs, vecs):
    """
        Extends the distance matrix for a set of given vectors
        to include new ones
    """
    # TODO: implement extending
    vecs = np.append(oldvecs, vecs)    
    return pdist(vecs, metric=metric)

def fcluster_from_hierarchy(hierarchy, threshold):
    pass