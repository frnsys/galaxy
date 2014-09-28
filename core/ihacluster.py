from scipy import clip
from scipy.spatial.distance import pdist, squareform, euclidean
from scipy.cluster.hierarchy import linkage, fcluster

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

DISTANCE = 'euclidean'

def distance(vec1, vec2):
    if DISTANCE == 'euclidean':
        return euclidean(vec1, vec2)


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
            self.hierarchy = Hierarchy(vec)
        else:
            # find closest leaf node
            node = self.get_closest_leaf(vec)
            d = 
            found = False

            # Bottom-up host node search:
            # Starting from the parent of the closest leaf node, per-
            # form upward search to locate a cluster (or create a new
            # cluster hierarchy) that can host the new point with min-
            # imal density changes and minimal disruption of the hi-
            # erarchy monotonicity.
            while node and not found:

            # Let N be the node being examined at current level.
            # The placement of a new point N J
            # in the hierarchy is performed according to the follow-
            # ing rules:
            # • if L L ≤ d ≤ U L then perform INS NODE
            # (N, N J ) (see Figure 1a) where d is the distance
            # from the new point N J to the nearest N ’s child
            # node.

            # • if N J forms a higher dense region on N , and N J
            # forms a lower dense region on at least one of N ‘s
            # child nodes then perform INS HIERARCHY
            # (N I , N J ) (see Figure 1b) where N I is the child
            # node of N closest to the new point N J .

            # If none of the rules applies, the search proceeds to the
            # next higher-level cluster. If the search process reaches
            # the top-level cluster, a new cluster will be inserted at
            # the top level using the hierarchy insertion operator.



            # hierarchy restructuring


    def get_closest_leaf(self, vec):
        # if distance matrix dimension is bigger than
        # current number of leaves, it means we preloaded
        # distance matrix for the points we are incorporating
        # (they are assumed to arrive in the same order)

        # otherwise, we must enlarge the distance matrix with
        # distances for the new point





class Hierarchy(object):
    def __init__(self, first_vec):
        self.leaves = [Node(first_vec)]
        self.labels = None
        # complete


class Node(object):
    def __init__(self, vec=None, children=[], parent=None):
        self.center = vec
        self.children = children
        if children:
            self.center = sum([c.center for c in children])/len(children)
            # TODO get distances
            self.distances = []

        self.parent = parent
        self.mu = None
        self.sigma = None
        # a list of tuples (index of nearest sibling, distance) 

    def add_child(self, node):
        n = len(self.children)
        self.center = ((self.center * n) + node.center) / (n + 1)
        # TODO: update distances
        self.children.append(node)

    def remove_child(self, index):
        child = self.children[index]
        n = len(self.children)
        self.center = ((self.center * n) - child.center) / (n - 1)
        # TODO: update distances
        del self.children[index]

    def density_lower_limit(self):
        n = len(self.children)
        if n > 2:
            return mu - sigma
        else:
            return (2.0 / 3) * self.distances[0][0]

    def density_upper_limit(self):
        n = len(self.children)
        if n > 2:
            return mu + sigma
        else:
            return 1.5 * self.distances[0][0]

