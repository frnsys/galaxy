from itertools import chain

import numpy as np

from .util import split_dist_matrix

class Node():
    # For determining the lower and upper limits.
    lower_limit_scale = 0.1
    upper_limit_scale = 1.5

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

    def __repr__(self):
        return str('<Node {0} ({1})>'.format(self.center, self.id))

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
    def __init__(self, id, vec):
        self.id = id
        self.center = vec
        self.parent = None

    @property
    def leaves(self):
        return [self]

    @property
    def children(self):
        return []

    @property
    def is_root(self):
        return False

    def __repr__(self):
        return str('<LeafNode {0} ({1})>'.format(self.center, self.id))


class ClusterNode(Node):
    def __init__(self, id, children, hierarchy):
        """
            A new cluster node is created by passing
            a list of children.
        """
        self.id = id
        self.children = children
        self.parent = None

        # A reference to the hierarchy to which this node belongs.
        # We can access the master distance matrix this way.
        self.hierarchy = hierarchy

        for ch in self.children:
            ch.parent = self
        self.center = np.mean([c.center for c in self.children], axis=0)

        self._update_children_dists()

    def __repr__(self):
        return str('<ClusterNode {0}, ndm {1}, nds {2}, ll {3}, ul {4} ({5})>'.format(self.center, self.nearest_dists_mean, self.nearest_dists_std, self.lower_limit, self.upper_limit, self.id))

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

        self.children.remove(child)

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
        np.fill_diagonal(dist_mat, np.inf)
        self.nearest_dists = np.min(dist_mat, axis=0)
        self.nearest_dists_mean = np.mean(self.nearest_dists)
        self.nearest_dists_std  = np.std(self.nearest_dists)

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

    def split_children(self):
        """
        Splits the set of children of self into two nodes.

        We construct the minimum spanning tree (MST) out of this cluster node's
        distance matrix (that is, the distance matrix of its children).

        We split this MST by removing the edge connecting nodes m_i and m_j,
        where m_i and m_j's edge has the greatest weight in the MST.
        """
        # The children dist matrix is a copy, so we can overwrite it.
        c_i, c_j = split_dist_matrix(self.cdm, overwrite=True)

        nodes = []
        for c in [c_i, c_j]:
            children = [self.children[i] for i in c]
            if len(children) == 1:
                n = children[0]
                self.hierarchy.update_distances(n)
            else:
                n = self.hierarchy.create_node(ClusterNode, children=children)
            nodes.append(n)
        return nodes


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
    def nearest_children(self):
        """
        The pair of children having the shortest nearest distance.
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
    def furthest_nearest_children(self):
        """
        The pair of children having the largest nearest distance.
        """

        # Get the largest of the nearest distances.
        max = self.nearest_dists.max()

        # Replace anything greater than the largest nearest distance with 0.
        dist_mat = self.cdm
        dist_mat[dist_mat > max] = 0

        # Find the argmax.
        i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
        n_i, n_j = self.children[i], self.children[j]
        d = dist_mat[i, j]
        return n_i, n_j, d

    @property
    def lower_limit(self):
        """
        The lower limit, which just needs to be some function of
        the nearest distance mean and the nearest distance standard deviation.
        """
        return Node.lower_limit_scale * (self.nearest_dists_mean - self.nearest_dists_std)

    @property
    def upper_limit(self):
        """
        The upper limit, which just needs to be some function of
        the nearest distance mean and the nearest distance standard deviation.
        """
        return Node.upper_limit_scale * (self.nearest_dists_mean + self.nearest_dists_std)

    @property
    def is_root(self):
        return self.parent is None
