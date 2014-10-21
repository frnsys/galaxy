from sys import platform

from scipy import clip
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from .ihac.hierarchy import IHACHierarchy

def hac(vecs, metric, linkage_method, threshold):
    """
    Hierarchical Agglomerative Clustering.

    `scipy.spatial.distance.pdist` seemed to perform
    quite a bit faster than `sklearn.metrics.pairwise.pairwise_distances`
    when they both operated as a single process. `pdist`, however, does not
    accept sparse matrices as inputs (as of scipy v0.15.2), so there can be
    quite a bit memory cost when using it.

    `pairwise_distances` does accept sparse matrices and has built-in
    support for parallelization (with the `n_jobs` param). Because it can
    have a significantly lower memory footprint, it seems better to use
    that as a multicore job.
    """

    if platform == 'darwin':
        # This breaks on OSX 10.9.4, py3.3+, with large arrays:
        # https://stackoverflow.com/questions/11662960/ioerror-errno-22-invalid-argument-when-reading-writing-large-bytestring
        # https://github.com/numpy/numpy/issues/3858
        # So for OSX ('darwin'), just running it as a single job.
        distance_matrix = pairwise_distances(vecs, metric=metric, n_jobs=1)
    else:
        # n_jobs=-1 to use all cores, n_jobs=-2 to use all cores except 1, etc.
        distance_matrix = pairwise_distances(vecs, metric=metric, n_jobs=-2)

    # `pairwise_distances` returns the distance matrix in squareform,
    # we use `squareform()` to convert it to condensed form, which is what `linkage()` accepts.
    distance_matrix = squareform(distance_matrix, checks=False)

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
