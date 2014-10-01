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
        self.hierarchy = Hierarchy()

    def fcluster(self, density_threshold=None):
        """
            Creates flat clusters by pruning all clusters
            with density higher than the given threshold
            and taking the leaves of the resulting hierarchy

            In case no density_threshold is given,
            we use the average density accross the entire hierarchy
        """
        pass


class Hierarchy(object):
    def __init__(self, size):
        """
            Size is the number of points we plan to cluster
        """
        self.size
        m_nodes = 2 * size - 1 # maximum number of nodes in the hierarchy
        self.nodes = [None] * m_nodes  # a list that will hold all the nodes created
        self.distances = -1 * np.ones((, 2 * size)) # A matrix holding cluster center distances for pairs of  node indices
        self.leaves = set() # Indices of leaf nodes
        self.root = None

    def resize(self):
        """
            resize data structures to hold process
            a new batch of points
        """
        pass

    def incorporate(self, vec):
        new_node = self.hierarchy.create_node(vec)
        if len(self.hierarchy.leaves) > 1:
            leaf, d = self.get_closest_leaf(new_node) 
            host = None
            node = leaf.parent
            while node and host is None:
                nchild, d = node.get_nearest_child(new_node)
                if d >= node.lower_limit() and d <= node.upper_limit():
                    self.ins_node(node, new_node)
                    host = node
                elif d < node.lower_limit:
                    # • if N J forms a higher dense region on N , 
                    # and N J forms a lower dense region on at least one of N ‘s
                    # child nodes 
                    dist = None
                    for ch in node.children:
                        if self.forms_lower_dense_region(new_node, ch):
                            break
                            if dist is None or cdist < dist:
                                dist = cdist
                                host = ch
                    # then perform INS HIERARCHY (N I , N J ) 
                    # where N I is the child node of N closest to the new point N J .

                    # QUESTION: is N I supposed to be chosen among those child nodes
                    # for which N J forms a lower dense region?
                    if host:
                        self.ins_hierarchy(host, new_node)

            if host is None:
                self.ins_hierarchy(self.root, new_node)

            self.restructure_hierarchy()

    def restructure_hierarchy(self, host_node):
        """Algorithm Hierarchy Restructuring
            starting on host_node, we traverse ancestors doing
            the following:
            1. Recover the siblings of current that are misplaced.
                (A node N J is misplaced as N I ’s sibling iff
                N J does not form a lower dense region in N I )
                In such case we apply DEMOTE(N I , N J )

            2. Maintain the homogeneity of crntNode.
        """
        current = host_node
        while current:
            parent = current.parent
            siblings = parent.children
            misplaced = [s for s in siblings if not self.forms_lower_dense_region(s, current)]
            for node in misplaced:
                self.demote(current, node)

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

    #
    # Distance functions
    #
    def get_distance(self, ni, nj):
        if self.distances[i, j] < 0:
            self.distances[ni.id, nj.id] = distance(ni.center, nj.center)
        return self.distances[i, j]

    def get_closest_leaf(self, node):
        """
            returns closest leaf node and the distance to it
        """
        mdist = np.inf
        cleaf = None
        for i in self.leaves:
            leaf = self.nodes[i]
            dist = self.get_distance(leaf, node)
            if dist < mdist:
                mdist = dist
                cleaf = leaf

        return leaf, mdist

    def get_nearest_child(self, parent, node):
        return nchild, d

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



    #
    # Restructuring operators
    #
    def ins_node(self, ni, nj):
        # Mariano
        pass

    def ins_hierarchy(self, ni, nj):
        # Mariano
        pass

    def demote(self, ni, nj):
        pass

    def merge(self, ni, nj):
        pass

    def split(self, theta, nk):
        pass


class Node(object):
    def __init__(self, vec=None, parent=None):
        self.center = vec
        self.children = children
        if children:
            self.center = scipy.mean([c.center for c in children])
            # TODO: get distances
            # a list of tuples (index of nearest sibling, distance)
            self.distances = []
        self.id = None
        self.parent = parent
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

    def lower_limit(self):
        n = len(self.children)
        if n > 2:
            return mu - sigma
        else:
            return (2.0 / 3) * self.distances[0][0]

    def upper_limit(self):
        n = len(self.children)
        if n > 2:
            return mu + sigma
        else:
            return 1.5 * self.distances[0][0]
