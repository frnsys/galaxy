from itertools import chain

import numpy as np

from .util import split_dist_matrix

class Node():
    def __init__(self):
        raise NotImplementedError

    @property
    def leaves(self):
        raise NotImplementedError

    @property
    def siblings(self):
        if self.parent:
            return [ch for ch in self.parent.children if ch.id != self.id]
        return []

    def forms_lower_dense_region(self, C):
        """
        Let C be a homogenous cluster. 
        Given a new point A, let B be C's cluster member that is the nearest
        neighbor to A. Let d be the distance from A to B. A (and B)
        is said to form a lower dense region in C if d > U_L

        Note:
            A => self
            B => nearest_child
        """
        assert(type(C) is ClusterNode)
        nearest_child, d = C.get_nearest_child(self)
        return d > C.upper_limit

    def forms_higher_dense_region(self, C):
        """
        Let C be a homogenous cluster. 
        Given a new point A, let B be C's cluster member that is the nearest
        neighbor to A. Let d be the distance from A to B. A (and B)
        is said to form a higher dense region in C if d < L_L

        Note:
            A => self
            B => nearest_child
        """
        assert(type(C) is ClusterNode)
        nearest_child, d = C.get_nearest_child(self)
        return d < C.lower_limit


class LeafNode(Node):
    def __init__(self, id, vec, parent=None):
        self.id = id
        self.parent = parent
        self.center = vec

    @property
    def leaves(self):
        return [self]


class ClusterNode(Node):
    def __init__(self, id, children, parent, hierarchy):
        """
            A new cluster node is created by passing
            a list of children.
        """
        if len(children) < 2: raise Exception

        self.id = id
        self.parent = parent
        self.children = children

        # A reference to the hierarchy to which this node belongs.
        # We can access the master distance matrix this way.
        self.hierarchy = hierarchy

        for ch in self.children:
            ch.parent = self
        self.center = np.mean([c.center for c in self.children], axis=0)

        self._update_children_dists()

    @property
    def leaves(self):
        return list(chain.from_iterable([ch.leaves for ch in self.children]))

    def add_child(self, node):
        node.parent = self
        self.children.append(node)

        self.center = np.mean([c.center for c in self.children], axis=0)

        # (Re)calculate distances to this node's new center.
        self.hierarchy.update_distances(self)

        self._update_children_dists()

    def remove_child(self, child):
        child.parent = None

        i = self.children.index(child)
        del self.children[i]

        self.center = np.mean([c.center for c in self.children], axis=0)

        # (Re)calculate distances to this node's new center.
        self.hierarchy.update_distances(self)

        self._update_children_dists()

    def _update_children_dists(self):
        """
        Calculate distances between children to update
        the master distance matrix.
        """
        for i, ch in enumerate(self.children):
            for ch_ in self.children:
                self.hierarchy.distance(ch, ch_)

        # Update mean and std.
        # Since the distance matrix is symmetric, it doesn't matter
        # which axis we use for the min.
        dist_mat = self.cdm
        np.fill_diagonal(dist_mat, np.nan)
        min = np.nanmin(dist_mat, axis=0)
        self.nearest_dists_mean = np.mean(min)
        self.nearest_dists_std  = np.std(min)

    @property
    def mdm(self):
        """
        Convenience access to the master distance matrix of the hierarchy.
        """
        return self.hierarchy.dists

    @property
    def cdm(self):
        """
        Get a view of the master distance matrix representing this cluster node's children.
        Note that this is only a _view_ (i.e. a copy), thus any modifications you make are
        not propagated to the original matrix.
        """
        rows, cols = zip(*[([ch.id], ch.id) for ch in self.children])
        return self.mdm[rows, cols]

    # TO DO
    def split_children(self, m_i, m_j):
        """
        Splits the set of children of self
        in two groups containing m_i and m_j.

        We construct the minimum spanning tree (MST) out of this cluster node's
        distance matrix (that is, the distance matrix of its children).

        We split this MST by removing the edge connecting nodes m_i and m_j,
        where m_i and m_j's edge has the greatest weight in the MST.
        """
        # The children dist matrix is a copy, so we can overwrite it.
        c_i, c_j = split_dist_matrix(self.cdm, overwrite=True)

        c_i = self.h.create_node(ClusterNode, children=[self.children[i] for i in c_i])
        c_j = self.h.create_node(ClusterNode, children=[self.children[i] for i in c_j])
        return c_i, c_j


    def get_nearest_child(self, n, clusters_only=False):
        if clusters_only:
            cols = [ch.id for ch in self.children if type(ch) is ClusterNode]
        else:
            cols = [ch.id for ch in self.children]

        # Build a view of the master distance matrix showing only
        # n's distances with children of this cluster node.
        dist_mat = self.mdm[[[n.id]], cols][0]
        i = np.argmin(dist_mat)
        return self.children[i], dist_mat[i]


    @property
    def closest_children(self):
        """
            returns the pair of children having
            the shortest nearest distance
        """
        dist_mat = self.cdm

        # Fill the diagonal with nan. Otherwise, they are all 0,
        # since distance(n_i,n_i) = 0.
        np.fill_diagonal(dist_mat, np.nan)
        i, j = np.unravel_index(np.nanargmin(dist_mat), dist_mat.shape)
        n_i, n_j = self.children[i], self.children[j]
        d = dist_mat[i, j]
        return n_i, n_j, d

    @property
    def furthest_children(self):
        """
            returns the pair of children having
            the largest nearest distance
        """
        dist_mat = self.cdm
        i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
        n_i, n_j = self.children[i], self.children[j]
        d = dist_mat[i, j]
        return n_i, n_j, d

    @property
    def lower_limit(self):
        return self.nearest_dists_mean - self.nearest_dists_std

    @property
    def upper_limit(self):
        return self.nearest_dists_mean + self.nearest_dists_std

    def is_root(self):
        return self.parent is None
