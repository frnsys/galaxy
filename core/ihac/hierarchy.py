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
        # TO DO a leaf node cannot be root, it has to be a cluster.
        node = LeafNode(id=0, vec=vector, parent=None)
        self.root = node
        self.nodes = [node]
        self.leaves = [node]
        self.dists = np.array([[0]], order='C')

    def incorporate(self, vec):
        """
        Incorporate a new vector into the hierarchy.
        """
        n = self.create_node(LeafNode, vec=vec, parent=None)
        n_c, d = self.get_closest_leaf(n)
        n_cp = closest_leaf.parent

        # Try to find a parent for the new node, n.
        # If the distance d is w/in the limits of the candidate parent n_cp,
        # just insert n there.
        while n_cp:
            if d >= n_cp.lower_limit and d <= n_cp.upper_limit:
                n_cp.add_child(n)
                break

            # Otherwise, n forms a higher dense region with n_cp
            # (i.e. if d < n_cp.lower_limit)...
            elif d <= n_cp.lower_limit:
                # And if n forms a lower dense region with at least one of n_cp's children...
                for ch in [ch for ch in n_cp.children if type(ch) is ClusterNode]:
                    if n.forms_lower_dense_region(ch):
                        self.ins_hierarchy(n_c, n)
                        break

            # If the node has still not been incorporated,
            # move up the hierarchy to the next cluster node.
            # If n_cp.parent is None, we've reached the top of the hierarchy and this loop stops.
            n_cp = n_cp.parent
            if n_cp is not None:
                n_c, d = n_cp.get_nearest_child(n, clusters_only=True)

        # If the node has still not been incorporated,
        # replace the root with a new cluster node containing both n_cp and n.
        if n.parent is None:
            self.ins_hierarchy(n_cp, n)
        else:
            self.restructure(n.parent)

        self.leaves.append(n)

    def restructure(self, n_h):
        """
        Algorithm Hierarchy Restructuring:

        Starting from the host node n_h, we traverse ancestors doing
        the following:

        1. Recover the siblings of the node that are misplaced.
           (A node n_j is misplaced as n_i's sibling iff
           n_j does not form a lower dense region in n_i)
           In such case we apply DEMOTE(n_i, n_j)

        2. Maintain the homogeneity of crntNode.
        """
        while n_h:
            n_hp = n_h.parent
            if not n_h.is_root:
                misplaced = [s for s in n_h.siblings if not s.forms_lower_dense_region(n_h)]
                for n in misplaced:
                    self.demote(n_h, n)

            self.repair_homogeneity(n_h)
            n_h = n_hp

    def repair_homogeneity(self, n):
        """
        Algorithm Homogeneity Maintenance

        1. Let an input n be the node that is being examined.
        2. Repeat
            3. Let n_i and n_j be the pair of neighbors among n's
               child nodes with the smallest nearest distance.
            4. If n_i and n_j form a higher dense region,
                5. Then MERGE(n_i, n_j)
        6. Until there is no higher dense region found_host in n during
           the last iteration.

        7. Let m_i and m_j be the pair of neighbors among n's
           child nodes with the largest nearest distance.
        8. If m_i and m_j form a lower dense region in n,
            9. Then Let (n_i , n_j) = SPLIT(Θ, n).
            10. Call Homogeneity Maintenance(n_i).
            11. Call Homogeneity Maintenance(n_j).
        """

        while len(n.children) > 2:
            n_i, n_j, d = n.nearest_children

            # If n_i and n_j form a higher dense region...
            if d < n.lower_limit:
                self.merge(n_i, n_j)
            else:
                break

        if len(n.children >= 3):
            m_i, m_j, d = n.furthest_nearest_children

            # If m_i and m_j form a lower dense region,
            # split the node.
            if d > n.upper_limit:
                n_i, n_j = self.split(n)
                for n in [n_i, n_j]:
                    if type(n) is ClusterNode: self.repair_homogeneity(n)

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

    def ins_hierarchy(self, n_i, n_j):
        if not n_i.is_root:
            # Remove n_i from its parent and replace it with a new cluster node
            # containing both n_i and n_j as children.
            n_p = n_i.parent
            n_p.remove_child(n_i)
            n_k = self.create_node(ClusterNode, children=[n_i, n_j])
            n_p.add_child(n_k)

        else:
            # If n_i is the root we just replace it with a new cluster node
            # containing both n_i and n_j as children.
            n_k = self.create_node(ClusterNode, children=[n_i, n_j])
            self.root = n_k

    def demote(self, n_i, n_j):
        """
        Demote n_i to a child of n_j.
        """
        n_p = n_i.parent
        n_p.remove_child(n_j)
        n_i.add_child(n_j)

    def merge(self, n_i, n_j):
        """
        Replace both n_i and n_j with a cluster node containing both of them
        as its children.
        """
        n_p = n_i.parent
        n_p.remove_child(n_i)
        n_p.remove_child(n_j)
        n_k = self.create_node(ClusterNode, children=[n_i, n_j])
        n_p.add_child(n_k)

    def split(self, n):
        """
        Split cluster node n by its largest nearest distance into two nodes,
        and replace it with those new cluster nodes.
        """
        n_i, n_j = n.split_children()
        if n.is_root:
            self.root = self.create_node(ClusterNode, children=[n_i, n_j])
        else:
            n_p = n.parent
            n_p.remove_child(n)
            n_p.add_child(n_i)
            n_p.add_child(n_j)
        # TO DO: need to delete the cluster node from the distance matrix and reclaim its id?
        # or could we just reuse one of the old cluster nodes?
        return n_i, n_j
