from sys import platform

from scipy import clip
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from core.ihac import IHAC
from core.ihac.node import Node

from scipy import argmax
from core.digbc import DocumentIndexGraphClusterer
from core.digshc.shc import SimilarityHistogramClusterer

"""
Notes:
- for the most part, these take vectors representing articles
  and then some hyperparameters.
- they all return labels which correspond to documents by their order of input.
  e.g. labels[0] is for vecs[0] or docs[0].
"""


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


def ihac(vecs, metric, threshold, lower_limit_scale, upper_limit_scale):
    """
    Convenience method for clustering with IHAC.
    """
    Node.lower_limit_scale = lower_limit_scale
    Node.upper_limit_scale = upper_limit_scale
    model = IHAC(metric=metric)
    model.fit(vecs.toarray())
    clusters, labels = model.clusters(distance_threshold=threshold, with_labels=True)
    return labels


def digbc(docs, threshold):
    # Worth nothing that this takes in plaintext documents, _not_ vectors.
    dig = DocumentIndexGraphClusterer(threshold=threshold)

    for doc in docs:
        dig.index_document(doc)

    doc_clus_map = {}
    for idx, clus in enumerate(dig.formed_clusters):
        for doc_id in clus.doc_ids:
            doc_clus_map.setdefault(doc_id, [])
            doc_clus_map[doc_id].append(idx)

    labels = []
    for id in sorted(doc_clus_map):
        clusters = [dig.get_cluster(cl_id) for cl_id in doc_clus_map[id]]
        sims = [dig.get_cluster_sim(cl, dig.get_doc(id)) for cl in clusters]
        max_i = argmax(sims)
        labels.append(clusters[max_i].id)

    return labels


def digshc(docs, alpha, threshold, epsilon, hr_min):
    shc = SimilarityHistogramClusterer(alpha, threshold, epsilon, hr_min)

    for doc in docs:
        shc.fit(doc)

    doc_clus_map = {}
    for idx, clus in enumerate(shc.formed_clusters):
        for doc_id in clus.doc_ids:
            doc_clus_map.setdefault(doc_id, [])
            doc_clus_map[doc_id].append(idx)

    labels = []
    for id in sorted(doc_clus_map):
        cluster_ids = doc_clus_map[id]
        if len(cluster_ids) == 1:
            best_cl_id = cluster_ids[0]
        else:
            clusters = [shc.get_cluster(cl_id) for cl_id in cluster_ids]
            sims = [shc.get_cluster_sim(cl, shc.get_doc(id)) for cl in clusters]
            max_i = argmax(sims)
            best_cl_id = clusters[max_i].id
        labels.append(best_cl_id)

    return labels

