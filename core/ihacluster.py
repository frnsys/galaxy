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

    def get_distance(self, i, j):
        if not (i, j) in self.distances:
            ni = self.nodes[i]
            nj = self.nodes[j]
            distances[(i, j)] = distance(ni.center, nj.center)

    def forms_lower_dense_region(self, a, c):
        """
        Let C be a homogenous cluster. 
        Given a new point A, let B be a C‘s cluster member that is the nearest
        neighbor to A. Let d be the distance from A to B. A (and B)
        is said to form a lower dense region in C if d > U_L
        """
        b = None
        pass

    def forms_higher_dense_region(self, a, c):
        """
        Let C be a homogenous cluster. Given a new
        point A, let B be a C‘s cluster member that is the nearest
        neighbor to A. Let d be the distance from A to B. A (and B)
        is said to form a higher dense region in C if d < L_L .
        """
        b = None
        pass



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
        current = host_node
        while current:
            parent = current.parent
            # Recover the siblings of current that are misplaced.
            # 
            # One of the most common problems is that a node is
            # stranded at an upper level cluster. In such a case,
            #  a node N J , which is supposed to be a child node of N I,
            #  is misplaced as N I ‘s sibling.
            #
            # A node N J , which is the sibling of N I , is said to be misplaced as
            # N I ’s sibling if and only if N J does not form a lower dense
            # region in N I .
            #
            # If such a problem is detected, we iteratively apply DEMOTE(N I , N J )
            siblings = parent.children
            misplaced = [s for s in siblings if not self.forms_lower_dense_region(s, current)]
            for node in misplaced:
                self.demote(current, node)

            # 5. Maintain the homogeneity of crntNode.
            self.repair_homogeneity(current)
            current = parent
    
    def repair_homogeneity(self, node):
        # Algorithm Homogeneity Maintenance(N )
        # 1. Let an input N be the node that is being examined.
        # 2. Repeat
        # 3. Let N I and N J be the pair of neighbors among N ‘s
        # child nodes with the smallest nearest distance.
        # 4. If N I and N J form a higher dense region,
        # 5. Then MERGE (N I , N J ) (see Figure 1d)
        # 6. Until there is no higher dense region found in N during
        # the last iteration.
        # 7. Let M I and M J be the pair of neighbors among N ‘s
        # child nodes with the largest nearest distance.
        # 8. If M I and M J form a lower dense region in N ,
        # 9. Then Let (N I , N J ) = SPLIT (Θ, N ). (see Figure 1e)
        # 10. Call Homogeneity Maintenance(N I ).
        # 11. Call Homogeneity Maintenance(N J ).
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
            # TODO: get distances
            # a list of tuples (index of nearest sibling, distance)
            self.distances = []

        self.index = None
        self.parent_index = parent_index
        self.mu = None
        self.sigma = None
         

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

