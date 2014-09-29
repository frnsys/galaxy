import scipy
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
    def __init__(self):
        self.nodes = []
        self.leaves = [] # Indices of leaf nodes
        self.distances = {} # A dictionary holding cluster center distances for pairs of  node indices

    def incorporate(self, vec):
        if not self.nodes:
            self.nodes = [Node(vec)]
            self.leaves = 0
        else:
            # find closest leaf node
            node, d = self.get_closest_leaf(vec) 
            found = False

            # Bottom-up host node search:
            # Starting from the parent of the closest leaf node, per-
            # form upward search to locate a cluster (or create a new
            # cluster hierarchy) that can host the new point with min-
            # imal density changes and minimal disruption of the hi-
            # erarchy monotonicity.
            host = None
            while node and host is None:
                node = self.get_parent(node)
                # Let N be the node being examined at current level.
                # The placement of a new point N J in the hierarchy
                # is performed according to the following rules:
                
                if d >= node.density_lower_limit() and\
                    d <= node.density_upper_limit():
                    # • if L L ≤ d ≤ U L then perform INS NODE
                    # (N, N J ) (see Figure 1a) where d is the distance
                    # from the new point N J to the nearest N ’s child
                    # node.

                    self.ins_node(node, Node(vec))
                    host = node
                elif d < node.density_lower_limit:
                    # • if N J forms a higher dense region on N , 
                    # and N J forms a lower dense region on at least one of N ‘s
                    # child nodes 
                    dist = None
                    for ch in node.children:
                        cdist = euclidean(vec, ch.center)
                        if cdist > ch.density_upper_limit():
                            if dist is None or cdist < dist:
                                dist = cdist
                                host = c
                    # then perform INS HIERARCHY (N I , N J ) 
                    # where N I is the child node of N closest to the new point N J .
                    if host:
                        found = True
                        self.ins_hierarchy(host, Node(vec))
                    # If none of the rules applies, the search proceeds to the
                    # next higher-level cluster.

            # If the search process reaches the top-level cluster,
            # a new cluster will be inserted at
            # the top level using the hierarchy insertion operator.
            if host is None:
                self.ins_hierarchy(self.root, Node(vec))

            # hierarchy restructuring
            self.restructure_hierarchy()


    def restructure_hierarchy(self, host_node):
        """Algorithm Hierarchy Restructuring
        """
        # 1. Let crntNode be the node that accepts the new point.
        # 2. While (crntN ode = ∅)
        # 3. Let parentNode ← Parent(crntNode)
        # 4. Recover the siblings of crntN ode that are misplaced.
        # 5. Maintain the homogeneity of crntNode.
        # 6. Let crntNode ← parentNode
        pass

    def get_closest_leaf(self, vec):
        """
            returns closest leaf node and the distance to it
        """
        # if distance matrix dimension is bigger than
        # current number of leaves, it means we preloaded
        # distance matrix for the points we are incorporating
        # (they are assumed to arrive in the same order)

        # otherwise, we must enlarge the distance matrix with
        # distances for the new point

    def get_parent(self, node):
        return self.nodes(node.parent_index)

    def fcluster(self, density_threshold=None):
        """
            Creates flat clusters by pruning all clusters
            with density higher than the given threshold
            and taking the leaves of the resulting hierarchy

            In case no density_threshold is given,
            we use the average density accross the entire hierarchy
        """
        pass

    def ins_node(self, ni, nj):
        pass

    def ins_hierarchy(self, ni, nj):
        pass

    def demote(self, ni, nj):
        pass

    def merge(self, ni, nj):
        pass

    def split(self, theta, nk):
        pass



class Node(object):
    def __init__(self, vec=None, children=[], parent_index=None):
        self.center = vec
        self.children = children
        if children:
            self.center = scipy.mean([c.center for c in children])
            # TODO get distances
            self.distances = []

        self.parent_index = parent_index
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

