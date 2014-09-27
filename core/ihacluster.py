from scipy import clip
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


class IHAClusterer(object):
    def __init__(self, state=None):
        """
            state resumes the clusterer with
            previous results
        """
        if state:
            self.hierarchy = state["hierarchy"]
            self.distance_matrix = state["distance_matrix"]

    def incorporate(self, vec):
        if hierarchy is None:
            # first vec, start new hierarchy
            pass
        else:
            pass

    def create_hierarchy(self, first_vec):
        pass
