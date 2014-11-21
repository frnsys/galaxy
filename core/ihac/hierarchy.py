import sys
import logging
from itertools import chain

import numpy as np
from scipy.spatial.distance import euclidean, cosine

# For visualization.
import matplotlib.pyplot as plt
import networkx as nx
import subprocess as sp

from .node import LeafNode, ClusterNode
from .util import mirror_upper, triu_index

dist_funcs = {
    'euclidean': euclidean,
    'cosine':    cosine
}

class Hierarchy():
    """
    A hierarchy manages many nodes.
    It has a master distance matrix which keeps track
    of the pairwise distances between all its nodes, and manages
    the assignment and recycling of ids across its nodes.
    It is initialized with two vectors.

    A few notes:
    - a cluster should never have less than two children. Any restructuring operation
        which could potentially lead to a cluster with only one or less children needs
        to fix the node afterwards.
    """
    def __init__(self, vec_A, vec_B, metric='euclidean'):
        """
        A hierarchy must be initialized with two vectors.
        """
        # For keeping track of ids which can be re-used.
        self.available_ids = []

        # So we know how distances are calculated.
        self.metric = metric

        # Create the initial leaf node.
        # Initialize all the other properties.
        node_A = LeafNode(id=0, vec=vec_A)
        self.dists = np.array([[0.]], order='C')
        self.leaves = [node_A]
        self.nodes = [node_A]

        node_B = self.create_node(LeafNode, vec=vec_B)

        # Create an initial cluster node as the root.
        self.root = self.create_node(ClusterNode, children=[node_A, node_B])

    def display(self, vertical=True):
        return self.root.display(vertical)

    def incorporate(self, vec):
        """
        Incorporate a new vector into the hierarchy.
        """
        logging.debug('\n\nIncorporating...')

        n = self.create_node(LeafNode, vec=vec)
        n_c, d = self.get_closest_leaf(n)
        n_cp = n_c.parent

        # Try to find a parent for the new node, n.
        # If the distance d is w/in the limits of the candidate parent n_cp,
        # just insert n there.
        while n_cp:
            # Note: after the first iteration, n_c could be a leaf *or* a cluster node.

            if d >= n_cp.lower_limit and d <= n_cp.upper_limit:
                n_cp.add_child(n)
                break

            # Otherwise, n forms a higher dense region with n_cp
            # (i.e. if d > n_cp.lower_limit)...
            elif d < n_cp.lower_limit:
                self.ins_hierarchy(n_c, n)
                break

            # Or if n forms a lower dense region with at least one of n_cp's children...
            # I think this only makes sense on the first interation, where n_c is a leaf,
            # so we also check cluster nodes. I'm not even sure that's true - this might be redundant.
            for ch in [ch for ch in n_cp.children if type(ch) is ClusterNode]:
                if n.forms_lower_dense_region(ch):
                    self.ins_hierarchy(ch, n)
                    break

            # If the node has still not been incorporated,
            # move up the hierarchy to the next cluster node.
            # If n_cp.parent is None, we've reached the top of the hierarchy and this loop stops.
            n_cp = n_cp.parent
            if n_cp is not None:
                # don't think this has to only be clusters.
                #n_c, d = n_cp.get_nearest_child(n, clusters_only=True)
                n_c, d = n_cp.get_nearest_child(n)

        # If the node has still not been incorporated,
        # replace the root with a new cluster node containing both n_cp and n.
        if n.parent is None:
            self.ins_hierarchy(self.root, n)

        self.restructure(n.parent)

        self.leaves.append(n)
        return n

    def restructure(self, n_i):
        """
        Algorithm Hierarchy Restructuring:

        Starting from the host node n, we traverse ancestors doing
        the following:

        1. Recover the siblings of the node that are misplaced.
           (A node n_j is misplaced as n_i's sibling iff
           n_j does not form a lower dense region in n_i)
           In such case we apply DEMOTE(n_i, n_j)

        2. Maintain the homogeneity of crntNode.
        """
        logging.debug('[RESTRUCTURE]\t Restructuring {0}'.format(n_i.id))
        while n_i:
            if not n_i.is_root:
                misplaced = [s for s in n_i.siblings if not s.forms_lower_dense_region(n_i)]
                for n_j in misplaced:
                    self.demote(n_i, n_j)

            self.repair_homogeneity(n_i)
            n_i = n_i.parent

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

        logging.debug('[REPAIR]\t\t Repairing homogeneity for {0}'.format(n.id))
        while len(n.children) > 2:
            n_i, n_j, d = n.nearest_children

            # If n_i and n_j form a higher dense region...
            if d < n.lower_limit:
                self.merge(n_i, n_j)
            else:
                break

        if len(n.children) >= 3:
            m_i, m_j, d = n.furthest_nearest_children

            # If m_i and m_j form a lower dense region,
            # split the node.
            if d > n.upper_limit:
                n_i, n_j = self.split(n)
                for n in [n_i, n_j]:
                    if type(n) is ClusterNode: self.repair_homogeneity(n)

    def get_closest_leaf(self, n):
        """
        Find the closest leaf node to a given node n.

        The assumption is that node n is not (yet) among the hierarchy's leaves.
        """
        # Build a view of the master distance matrix showing only
        # n's distances from the leaves of this hierarchy,
        # since we're not looking at cluster nodes.
        cols = [l.id for l in self.leaves]
        dist_mat = self.dists[[[n.id]], cols][0]
        i = np.argmin(dist_mat)
        return self.leaves[i], dist_mat[i]

    def create_node(self, node_cls, **init_args):
        """
        New nodes in the hierarchy MUST be created this way
        in order to properly manage node ids.

        The id must correspond to the node's index in the distance matrix.

        This creates a node but does NOT insert into the hierarchy,
        though it is tracked as one of the hierarchy's nodes (it is added to self.nodes).
        If you want to insert a node into the hierarchy, use the `incorporate` method.
        """

        # Reuse an id if we have one.
        if self.available_ids:
            id = self.available_ids.pop()

            # Now that we are using the distance row and col for this id,
            # reset to 0 (instead of inf).
            self.dists[id] = 0
            self.dists[:,id] = 0

        # Otherwise we need to expand the distance matrix and use a new id.
        else:
            # The next id is the length of dimension of the distance matrix.
            id = self.dists.shape[0]

            # Resize the distance matrix.
            # We can't use `np.resize` because it flattens the array first,
            # which moves values all over the place.
            dm = self.dists
            dm = np.hstack([dm, np.zeros((dm.shape[0], 1))])
            self.dists = np.vstack([dm, np.zeros(dm.shape[1])])

        logging.debug('[CREATE]\t\t Creating {0} {1}...'.format(node_cls.__name__, id))

        if node_cls == ClusterNode: init_args['hierarchy'] = self
        init_args['id'] = id
        node = node_cls(**init_args)

        self.update_distances(node)
        self.nodes.append(node)

        return node

    def delete_node(self, n):
        """
        Delete a node n and free up its id for reuse.

        This will delete ALL nodes in the subtree of n.
        """
        logging.debug('[DELETE]\t\t Deleting node {0}...'.format(n.id))

        i = n.id
        self.nodes.remove(n)

        if type(n) is LeafNode:
            self.leaves.remove(n)

        # Take it out of the hierarchy.
        # This only happens for the first node `delete_node` is called on,
        # since children nodes are disconnected from their parent (see the note below).
        if n.parent:
            n.parent.remove_child(n)
            n.parent = None

        # Make all distances at the node's former index infinity.
        self.dists[i]   = np.inf  # row
        self.dists[:,i] = np.inf  # col

        # Recycle the id.
        n.id = None
        self.available_ids.append(i)

        # Delete its subtree.
        for ch in n.children:
            # Remove the connection to the parent.
            # We don't actually have to explicitly remove n's children from n,
            # since these nodes are detached from the hierarchy (it doesn't matter).
            # Plus, calling `remove_child` on the parent does extraneous cleanup stuff
            # that is a waste if we are simply destroying the node.
            ch.parent = None
            self.delete_node(ch)

    def distance_function(self, a, b):
        return dist_funcs[self.metric](a.center, b.center)

    def update_distances(self, node):
        """
        Update all distances against a node.
        """
        for node_ in [n for n in self.nodes if n.id is not None]:
            # It's possible for cluster nodes without children to be encountered.
            # This is (should be) because they have not yet been deleted,
            # but the hierarchy does delete them eventually.
            if type(node_) is ClusterNode and node_.children == []:
                continue
            row, col = triu_index(node.id, node_.id)
            self.dists[row, col] = self.distance_function(node, node_)

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
            d = self.distance_function(n_i, n_j)
            i, j = triu_index(i, j)
            self.dists[i, j] = d
            self.dists = mirror_upper(self.dists)

        return d

    def ins_hierarchy(self, n_i, n_j):
        """
        This adds a node n_j which is not yet in the hierarchy by creating a new cluster
        node, n_k, which has n_j and n_i as children.

        The difference between merge and ins_hierarchy is that ins_hierarchy incorporates
        a node that is new (n_j) to the hierarchy.
        """
        logging.debug('[INS_HIERARCHY]\t Inserting {0} into the hierarchy alongside {1}...'.format(n_j.id, n_i.id))

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

    def fix_node(self, n):
        """
        This replaces a ClusterNode with its child if that child is an only child.
        """
        if len(n.children) == 1:
            logging.debug('[FIX]\t\t Fixing node {0}...'.format(n.id))

            n_c = n.children[0]

            if not n.is_root:
                # Replace the node n with node n_c.
                n_p = n.parent
                n_p.remove_child(n)
                n_p.add_child(n_c)

            else:
                assert(type(n_c) is ClusterNode)
                n_c.parent = None
                self.root = n_c

            # Clear out n's children so they aren't also deleted.
            n.children = []

            self.delete_node(n)

    def demote(self, n_i, n_j):
        """
        Demote n_j to a child of n_i.

        n_i must be a cluster node.
        """
        logging.debug('[DEMOTE]\t\t Demoting {0} to under {1}...'.format(n_j.id, n_i.id))

        n_p = n_i.parent
        n_p.remove_child(n_j)
        n_i.add_child(n_j)

        # It's possible that n_p now only has one child,
        # in which case it must be replaced with its only child.
        self.fix_node(n_p)

    def merge(self, n_i, n_j):
        """
        Replace both n_i and n_j with a cluster node containing both of them
        as its children.

        The difference between merge and ins_hierarchy is that merge works
        on nodes _already_ in the hierarchy.
        """
        logging.debug('[MERGE]\t\t Merging {0} and {1}...'.format(n_i.id, n_j.id))

        n_p = n_i.parent
        n_p.remove_children([n_i, n_j])
        n_k = self.create_node(ClusterNode, children=[n_i, n_j])
        n_p.add_child(n_k)

        # If n_i and n_j were n_p's only children, then it has only one child (n_k),
        # in which case it must be replaced by n_k.
        self.fix_node(n_p)

    def split(self, n):
        """
        Split cluster node n by its largest nearest distance into two cluster nodes,
        and replace it with those new nodes.
        """
        logging.debug('[SPLIT]\t\t Splitting {0}...'.format(n.id))
        n_i, n_j = n.split_children()

        if n.is_root:
            self.root = self.create_node(ClusterNode, children=[n_i, n_j])
        else:
            n_p = n.parent
            n_p.remove_child(n)
            n_p.add_children([n_i, n_j])

        # Delete the original cluster (and then its id will be reused).
        # But first we have to separate it from its children.
        n.children = []
        self.delete_node(n)
        return n_i, n_j

    def fcluster(self, distance_threshold=None):
        """
        Creates flat clusters by pruning all clusters
        with density higher than the given threshold
        and taking the leaves of the resulting hierarchy

        In case no distance_threshold is given,
        we use the average density accross the entire hierarchy
        (average of averages per level)
        """
        if distance_threshold is None:
            distance_threshold = self.avg_density

        clusters = [clus for clus in self._snip([self.root], distance_threshold)]
        return clusters

    def _snip(self, nodes, distance_threshold):
        """
        Yields clusters that satisfy a distance threshold.
        """
        for n in nodes:
            # Reached the end of the branch,
            # just take the leaf node as a cluster.
            if type(n) is LeafNode:
                yield [n]
                continue

            elif type(n) is ClusterNode:
                # If this cluster satisfies the distance threshold,
                # stop this branch here.
                if n.nearest_dists_mean <= distance_threshold:
                    yield n.leaves
                    continue

            # Otherwise, keep going down the branch.
            yield from self._snip(n.children, distance_threshold)

    @property
    def avg_density(self):
        level_averages = []

        current_level = [self.root]
        while current_level:
            # Get the average density for this level.
            level_averages.append(np.mean([n.nearest_dists_mean for n in current_level]))

            # Build out the next level.
            children = list(chain.from_iterable([n.children for n in current_level]))
            current_level = [ch for ch in children if type(ch) is ClusterNode]
        return np.mean(level_averages)

    def visualize(self, savepath='hierarchy.png'):
        """
        Visualize the hierarchy.
        """
        G = nx.DiGraph()

        current_level = [self.root]
        while current_level:
            for n in current_level:
                G.add_node(n)
                if n.parent is not None:
                    G.add_edge(n.parent, n)
            current_level = list(chain.from_iterable([n.children for n in current_level]))

        nx.write_dot(G,'/tmp/hierarchy.dot')
        with open(savepath, 'wb') as out:
            sp.call(['dot', '-Tpng', '/tmp/hierarchy.dot'], stdout=out)
