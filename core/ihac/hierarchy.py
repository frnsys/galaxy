import numpy as np
from scipy.spatial.distance import euclidean

from .node import LeafNode, ClusterNode
from .util import mirror_upper, triu_index

def distance(i, j):
    return euclidean(i, j)

class Hierarchy():
    """
    A hierarchy manages many nodes.
    It has a master distance matrix which keeps track
    of the pairwise distances between all its nodes, and manages
    the assignment and recycling of ids across its nodes.
    """
    def __init__(self, vector):
        """
        A hierarchy must be initialized with one vector.
        """
        node = LeafNode(id=0, vec=vector, parent=None)
        self.root = node
        self.nodes = [node]
        self.leaves = [node]
        self.dists = np.array([[0]], order='C')

    def add_vector(self, vec):
        """
        Add a new vector into the hierarchy.
        """
        n = self.create_node(LeafNode, vec=vec, parent=None)
        n_c, d = self.get_closest_leaf(n)
        n_cp = closest_leaf.parent

        # Try to find a parent for the new node, n.
        # If the distance is w/in the limits of the candidate parent n_cp,
        # just insert n there.
        if d >= n_cp.lower_limit and d <= n_cp.upper_limit:
            self.ins_node(n_cp, n)

        # Otherwise, 
        elif d <= n_cp.lower_limit:
            for ch in n_cp.children:
                if n.forms_lower_dense_region(ch)

    def get_closest_leaf(self, n):
        # Build a view of the master distance matrix showing only
        # n's distances from the leaves of this hierarchy.
        cols = [l.id for l in self.leaves]
        dist_mat = self.dists[[[n.id]], cols][0]
        i = np.argmin(dist_mat)
        return self.leaves[i], dist_mat[i]

    def create_node(self, node_cls, **init_args):
        """
        New nodes in the hierarchy MUST be created this way
        in order to properly manage node ids.

        The id must correspond to the node's index in the distance matrix.
        """
        idx = self.dists.shape[0]

        # Resize the distance matrix.
        # We can't use `np.resize` because it flattens the array first,
        # which moves values all over the place.
        dm = self.dists
        dm = np.hstack([dm, np.zeros((dm.shape[0], 1))])
        self.dists = np.vstack([dm, np.zeros(dm.shape[1])])

        if node_cls == ClusterNode: init_args['hierarchy'] = self
        init_args['id'] = idx
        node = node_cls(**init_args)

        self.update_distances(node)
        self.nodes.append(node)

        return node

    def update_distances(self, node):
        """
        Update all distances against a node.
        """
        for node_ in self.nodes:
            row, col = triu_index(node.id, node_.id)
            self.dists[row, col] = distance(node.center, node_.center)

        # Symmetrize the distance matrix, based off the upper triangle.
        self.dists = mirror_upper(self.dists)
        np.fill_diagonal(self.dists, 0)

    def distance(self, n_i, n_j):
        """
        Returns the pre-calculated distance b/w n_i and n_j, if it exists.
        Otherwise, calculates it and then returns it.
        """
        i, j = n_i.id, n_j.id

        # Identical nodes have a distance of 0.
        if i == j: return 0

        # Check if a pre-calculated distance exists.
        d = self.dists[i, j]

        # If not, calculate it.
        if d == 0:
            d = distance(n_i.center, n_j.center)
            i, j = triu_index(i, j)
            self.dists[i, j] = d
            self.dists = mirror_upper(self.dists)

        return d

