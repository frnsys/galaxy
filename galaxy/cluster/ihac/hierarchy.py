import logging

import numpy as np
import tables as tb
from scipy.spatial.distance import cdist, euclidean, cosine

from .util import split_dist_matrix
from .graph import Graph
from . import persistence, visual

# We should not get any numpy warnings since all operations should work
# if the hierarchy is working properly. So if we get a warning, we want to treat it as an error.
np.seterr(invalid='raise')

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
    - a "node" here is really just the index of the node in the matrices. So a node's representation is split across these matrices.
    - any changes in a cluster's membership (e.g. via adding or removing children) must be followed up by a update_cluster to update the cluster's values, then by update_distances to update distances against that cluster (but this is done for you).
    - creating a cluster node will automatically remove its children from their existing parents, so you don't need to call `remove_child` on their parents beforehand.
    """
    @staticmethod
    def load(filepath):
        """
        Load an existing hierarchy.
        """
        h5f = tb.openFile(filepath, mode='a', title='Hierarchy')
        root = h5f.root

        h = Hierarchy()

        h.ids     = root.ids
        h.ndists  = root.ndists
        h.centers = root.centers

        h.dists   = persistence.load_dists(h5f)
        h.g       = Graph(persistence.load_graph(h5f))

        h.metric            = root._v_attrs.metric
        h.lower_limit_scale = root._v_attrs.lower_limit_scale
        h.upper_limit_scale = root._v_attrs.upper_limit_scale

        return h

    def __init__(self, metric='euclidean', lower_limit_scale=0.9, upper_limit_scale=1.2):
        # Parameters.
        self.metric = metric
        self.lower_limit_scale = lower_limit_scale
        self.upper_limit_scale = upper_limit_scale

    def save(self, filepath):
        h5f = tb.openFile(filepath, mode='a', title='Hierarchy')
        root = h5f.root

        # Create these arrays if necessary.
        # Otherwise they should save themselves as they are changed.
        for name, shape in [('ids', (0,1)), ('ndists', (0,2)), ('centers', (0,self.centers.shape[1]))]:
            if not hasattr(root, name):
                arr = h5f.create_earray(root, name, tb.Float64Atom(), shape=shape, expectedrows=1000000)
                arr.append(getattr(self, name))

        persistence.save_graph(h5f, self.g.mx)
        persistence.save_dists(h5f, self.dists)

        # Hierarchy metadata.
        root._v_attrs.metric            = self.metric
        root._v_attrs.lower_limit_scale = self.lower_limit_scale
        root._v_attrs.upper_limit_scale = self.upper_limit_scale

        h5f.close()

    def initialize(self, vec_A, vec_B):
        """
        Initialize a *new* hierarchy.
        A hierarchy must be initialized with two vectors.
        """
        # For keeping track of ids which can be re-used.
        self.available_ids = []

        # n = number of nodes
        # m = number of features

        # Distance matrix (nxn).
        self.dists = np.array([[0.]], order='C', dtype=np.float64)

        # Nearest distance data (nx2).
        # The actual nearest distances are of variable length (size would be 1xnum_children).
        # So we only store the nearest distances mean and std.
        self.ndists = np.array([[0., 0.]], order='C', dtype=np.float64)

        # Leaf node ID matrix (nx1)
        # This maps internal ids of leaf nodes (which are resued) to a universally unique one.
        # This is used for preserving leaf input order for returning labeled clusters.
        self.ids = np.array([[0]], order='C', dtype=np.uint32)

        # Adjacency matrix (nxn).
        self.g = Graph()

        # Create initial leaves and root.
        # Centers matrix (mxn).
        # It is initialized with the first vector.
        self.centers = np.array([vec_A], order='C', dtype=np.float64) # node 0
        self.create_node(vec=vec_B)                                   # node 1
        self.create_node(children=[0,1])                              # root 2, no parents

        return [0, 1]

    def fit(self, vecs):
        # The uuids for each incorporated vector.
        uuids = []
        if not hasattr(self, 'dists'):
            if len(vecs) < 2:
                raise Exception('You must initialize the hierarchy with at least two vectors.')
            uuids += self.initialize(*vecs[:2])
            vecs = vecs[2:]
        for vec in vecs:
            uuids.append(self.incorporate(vec))
        return uuids

    def visualize(self, dir='vertical'):
        func = visual.render_node_vertical if dir is 'vertical' else visual.render_node_horizontal
        return func(self.g.root, child_iter=lambda n: self.g.get_children(n))

    def incorporate(self, vec):
        """
        Incorporate a new vector (leaf) into the hierarchy.
        """
        n = self.create_node(vec=vec)
        n_c, d = self.get_closest_leaf(n)
        n_cp = self.g.get_parent(n_c)

        # Try to find a parent for the new node, n.
        # If the distance d is w/in the limits of the candidate parent, n_cp,
        # just insert n there.
        while n_cp:
            # Note: after the first iteration, n_c could be a leaf *or* a cluster node.
            if self._incorporate(n, n_c, n_cp, d):
                break
            else:
                # If the node has still not been incorporated,
                # move up the hierarchy to the next cluster node.
                # If n_cp's parent is None, we've reached the top of the hierarchy and this loop stops.
                n_cp = self.g.get_parent(n_cp)
                if n_cp is not None:
                    n_c, d = self.get_nearest_child(n_cp, n)

        # If the node has still not been incorporated,
        # replace the root with a new cluster node containing both c and n.
        parent = self.g.get_parent(n)
        if parent is None:
            self.ins_hierarchy(self.g.root, n)

        self.restructure(parent)

        return self.ids[n][0]

    def _incorporate(self, n, n_c, n_cp, d):
        """
        Tries to find a way to integrate a new node n,
        where n_c is the nearest node (it may be a leaf or a cluster),
        n_cp is the parent of the nearest node, and d is the distance to n_c.
        """

        # If n fits within the limits of n_cp...
        if d >= self.lower_limit(n_cp) and d <= self.upper_limit(n_cp):
            self.g.add_child(n_cp, n)
            return True

        # Otherwise, if n forms a higher dense region with n_cp.
        elif d < self.lower_limit(n_cp):
            self.ins_hierarchy(n_c, n)
            return True

        # Or if n forms a lower dense region with at least one of n_cp's children...
        # I think this only makes sense on the first interation, where n_c is a leaf,
        # so we also check cluster nodes. I'm not even sure that's true - this might be redundant.
        for ch in [ch for ch in self.g.get_children(n_cp) if self.g.is_cluster(ch)]:
            if self.forms_lower_dense_region(n, ch):
                self.ins_hierarchy(ch, n)
                return True
        return False


    # =================================================================
    # NODE MGMT =======================================================
    # =================================================================

    def create_node(self, vec=None, children=[]):
        """
        New nodes in the hierarchy MUST be created this way
        in order to properly manage node ids.

        The id corresponds to the node's index in the matrices.

        This creates a node but does NOT insert into the hierarchy.
        If you want to insert a node into the hierarchy, use the `incorporate` method.
        """
        # If children are specified, we are making a cluster node.
        # If just a vector was specified, we are making a leaf node.
        assert(vec is not None or children)

        # Reuse an id if we have one.
        if self.available_ids:
            id = self.available_ids.pop()

            # Now that we are using the distance row and col for this id,
            # reset to 0 (instead of inf).
            self.dists[id] = 0
            self.dists[:,id] = 0

            # Reset relationships for this node id.
            self.g.reset_node(id)

        # Otherwise we need to expand the distance matrix and use a new id.
        else:
            # The next id is the length of dimension of the distance matrix.
            id = self.dists.shape[0]

            # Resize the distance, graph, centers, and ids matrices.
            # We can't use `np.resize` because it flattens the array first,
            # which moves values all over the place.
            self.dists   = self._expand_mat(self.dists, True, True)
            self.ndists  = self._expand_mat(self.ndists, False, True)
            self.centers = self._expand_mat(self.centers, False, True)
            self.ids     = self._expand_mat(self.ids, False, True)
            self.g.expand()


        # Add the children, if any were specified.
        if children:
            logging.debug('[CREATE]\t\t Creating cluster {0} with children {1}...'.format(id, children))
            self.g.set_parents(id, children)

            # Update cluster calculates the cluster's center
            # and updates distances against in,
            # which is why `update_distances` is unnecessary here.
            self.update_cluster(id)
        else:
            logging.debug('[CREATE]\t\t Creating leaf {0}...'.format(id))
            self.centers[id] = vec
            self.update_distances(id)

        # Assign an incremented id. These should be universally unique.
        self.ids[id] = np.max(self.ids) + 1

        return id

    def _expand_mat(self, m, expand_h, expand_v):
        # Add a column.
        if expand_h: m = np.hstack([m, np.zeros((m.shape[0], 1), dtype=m.dtype)])

        # Add a row.
        if expand_v: m = np.vstack([m, np.zeros(m.shape[1], dtype=m.dtype)])
        return m

    def delete_node(self, n):
        """
        Delete a node n and free up its id for reuse.

        This will delete ALL nodes in the subtree of n.
        """
        logging.debug('[DELETE]\t\t Deleting node {0}...'.format(n))

        children = self.g.get_children(n)

        # Remove the node from the hierarchy.
        self.g.reset_node(n)

        # Make all distances against the node infinity.
        self.dists[n]   = np.inf  # row
        self.dists[:,n] = np.inf  # col

        # Make the node's center infinity.
        self.centers[n] = np.inf  # row

        # Make the node's ndists 0.
        self.ndists[n]  = 0       # row

        # Delete its subtree.
        for ch_id in children:
            self.delete_node(ch_id)

        self.available_ids.append(n)

    def modify_children(self, p, remove=[], add=[]):
        """
        This batch-modifies a cluster node's children membership.
        """
        for ch in remove:
            self.g.remove_child(p, ch)

        for ch in add:
            self.g.add_child(p, ch)

        assert(self.g.get_children(p).size > 1)

        self.update_cluster(p)

    def update_cluster(self, p):
        """
        Updates a cluster's data:
        - center
        - nearest_dists
        - nearest_dists_mean
        - nearest_dists_std

        This needs to be called after any cluster has its children modified!
        If you are updating multiple clusters, start nested and move up.
        That is, if you have two clusters A and B, and A is a parent of B, first update B, then update A. That way A is working with the latest center for B.
        """
        children = self.g.get_children(p)
        self.centers[p] = np.mean([self.centers[c] for c in children], axis=0)

        # Since this node's center has changed,
        # distances to it must be updated.
        self.update_distances(p)

        ndists = self.get_nearest_distances(p)
        self.ndists[p] = [np.mean(ndists), np.std(ndists)]


    # =================================================================
    # HIERARCHY OPERATIONS ============================================
    # =================================================================

    def ins_hierarchy(self, i, j):
        """
        This adds a node j which is not yet in the hierarchy by creating a new cluster
        node, k, which has j and i as children.

        The difference between merge and ins_hierarchy is that ins_hierarchy incorporates
        a node that is new (j) to the hierarchy.
        """
        if not self.g.is_root(i):
            # Remove i from its parent and replace it with
            # a new cluster node containing both i and j.
            p = self.g.get_parent(i)

            logging.debug('[INS_HIERARCHY]\t Inserting {0} into the hierarchy alongside {1} under {2}...'.format(j, i, p))

            k = self.create_node(children=[i, j])
            self.modify_children(p, add=[k])
        else:
            logging.debug('[INS_HIERARCHY]\t Inserting {0} into the hierarchy alongside {1} under root...'.format(j, i))

            # Since the current root is added as a child,
            # this new node becomes the new root.
            k = self.create_node(children=[i, j])

    def fix(self, n):
        """
        This replaces a cluster node with its child if that child is an only child.
        """
        children = self.g.get_children(n)
        if children.size == 1:
            logging.debug('[FIX]\t\t Fixing node {0}...'.format(n))

            c = children[0]

            if not self.g.is_root(n):
                p = self.g.get_parent(n)
                self.modify_children(p, remove=[n], add=[c])
            else:
                assert(self.g.is_cluster(n))
                # Separate c from its parent.
                # It should be the root now (if n was the root).
                self.g.remove_parent(c)

            # Clear out n's children so they aren't also deleted.
            #self.clear_children(n) # this may be unnecessary
            self.delete_node(n)

            return True
        return False

    def demote(self, i, j):
        """
        Demote j to a child of i.
        i must be a cluster node.
        """
        logging.debug('[DEMOTE]\t\t Demoting {0} to under {1}...'.format(j, i))
        assert(self.g.is_cluster(i))
        p = self.g.get_parent(i)

        # Remove j from p.
        self.g.remove_child(p, j)

        # Add j to i.
        self.modify_children(i, add=[j])

        # It's possible that p now only has one child,
        # in which case it must be replaced with its only child.
        # If p gets fixed, it gets deleted, and we don't do anything more.
        # If p wasn't fixed (deleted), we need to update it.
        if not self.fix(p):
            self.update_cluster(p)

    def merge(self, i, j):
        """
        Replace both i and j with a cluster node containing both of them
        as its children.

        The difference between merge and ins_hierarchy is that merge works
        on nodes _already_ in the hierarchy.
        """
        logging.debug('[MERGE]\t\t Merging {0} and {1}...'.format(i, j))

        p = self.g.get_parent(i)
        k = self.create_node(children=[i, j])
        self.g.add_child(p, k)

        # If i and j were p's only children, then it has only
        # one child (k), in which case it must be replaced by k.
        # If p gets fixed, it gets deleted, and we don't do anything more.
        # If p wasn't fixed (deleted), we need to update it.
        if not self.fix(p):
            self.update_cluster(p)

    def split(self, n):
        """
        Split cluster node n by its largest nearest distance
        into two cluster nodes, and replace it with those new nodes.
        """
        logging.debug('[SPLIT]\t\t Splitting {0}...'.format(n))

        # The children dist matrix is a copy, so we can overwrite it.
        c_i, c_j = split_dist_matrix(self.cdm(n), overwrite=True)
        children = self.g.get_children(n)
        is_root  = self.g.is_root(n)

        # Create the new nodes out of the split children.
        new_nodes = []
        for c in [c_i, c_j]:
            ch = [children[i] for i in c]

            # If this cluster has only one child,
            # just use the child instead of creating a new cluster node.
            if len(ch) == 1:
                k = ch[0]
            else:
                k = self.create_node(children=ch)
            new_nodes.append(k)

        # If n is the root node, just create a parentless cluster
        # node, which will become the new root when n is deleted.
        if is_root:
            self.create_node(children=new_nodes)

        # Otherwise, replace n with the two new cluster nodes.
        else:
            p = self.g.get_parent(n)
            self.modify_children(p, remove=[n], add=new_nodes)

        # Separate n from its children before deleting it
        # (otherwise its children will also get deleted).
        #self.clear_children(n) # this might not be necessary
        self.delete_node(n)

        return new_nodes

    def restructure(self, n):
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
        logging.debug('[RESTRUCTURE]\t Restructuring {0}'.format(n))
        while n:
            if self.g.is_root(n):
                misplaced = [s for s in self.g.get_siblings(n) if self.forms_lower_dense_region(s, i)]
                for m in misplaced:
                    self.demote(n, m)
            self.repair_homogeneity(n)
            n = self.g.get_parent(n)

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
        logging.debug('[REPAIR]\t\t Repairing homogeneity for {0}'.format(n))

        while self.g.get_children(n).size > 2:
            i, j, d = self.get_nearest_children(n)

            # If n_i and n_j form a higher dense region...
            if d < self.lower_limit(n):
                self.merge(i, j)
            else:
                break

        if self.g.get_children(n).size >= 3:
            i, j, d = self.get_furthest_nearest_children(n)

            # If m_i and m_j form a lower dense region,
            # split the node.
            if d > self.upper_limit(n):
                k, g = self.split(n)
                for l in [k, g]:
                    if self.g.is_cluster(l): self.repair_homogeneity(l)


    # =================================================================
    # DISTANCES =======================================================
    # =================================================================

    def get_distance(self, i, j):
        if i == j: return 0

        # We only calculate distances for the lower triangle,
        # so adjust indices accordingly.
        idx = (j,i) if i < j else (i,j)
        d = self.dists[idx]

        # Calculate if necessary.
        if d == 0:
            d = dist_funcs[self.metric](self.centers[i], self.centers[j])
            self.dists[idx] = d
        return d

    def update_distances(self, n):
        """
        Update the distances of the node n against all other nodes.
        """
        # Update the row with the new distances.
        self.dists[n] = cdist([self.centers[n]], self.centers, metric=self.metric)
        # Also update the column.
        self.dists[:,n] = self.dists[n].T


    # =================================================================
    # NODE PROPS ======================================================
    # =================================================================

    def cdm(self, i):
        """
        Get a view of the master distance matrix representing this cluster node's children.
        Note that this is only a _view_ (i.e. a copy), thus any modifications you make are
        not propagated to the original matrix.
        """
        children = self.g.get_children(i)
        rows, cols = zip(*[([ch], ch) for ch in children])
        return self.dists[rows, cols]

    def get_closest_leaf(self, n):
        """
        Find the closest leaf node to a given node n.
        The assumption is that node n is not (yet) among the hierarchy's leaves.
        """
        # Build a view of the master distance matrix showing only
        # n's distances from the leaves of this hierarchy,
        # since we're not looking at cluster nodes.
        leaves = self.g.leaves
        dist_mat = self.dists[[[n]], leaves[leaves != n]][0]
        i = np.argmin(dist_mat)
        return leaves[i], dist_mat[i]

    def get_nearest_child(self, p, n):
        """
        Get the nearest child to n in the cluster node p.
        """
        children = self.g.get_children(p)

        # Build a view of the master distance matrix showing only
        # n's distances with children of this cluster node.
        dist_mat = self.dists[[[n]], children][0]
        i = np.argmin(self.dists)
        return children[i], dist_mat[i]

    def get_nearest_children(self, n):
        """
        Get the n's closest pair of children.
        """
        dist_mat = self.cdm(n)
        children = self.g.get_children(n)

        # Fill the diagonal with nan. Otherwise, they are all 0,
        # since distance(n_i,n_i) = 0.
        np.fill_diagonal(dist_mat, np.nan)

        i, j = np.unravel_index(np.nanargmin(dist_mat), dist_mat.shape)
        d = dist_mat[i, j]

        # i and j are indices local to the children,
        # so translate it to global indices n_i and n_j.
        n_i, n_j = children[i], children[j]

        return n_i, n_j, d

    def get_furthest_nearest_children(self, n):
        """
        The pair of children of n having the largest nearest distance.
        """
        children = self.g.get_children(n)

        # Get the largest of the nearest distances.
        max = self.get_nearest_distances(n).max()

        # Replace anything greater than the largest nearest distance with 0.
        dist_mat = self.cdm(n)
        dist_mat[dist_mat > max] = 0

        # Find the argmax.
        i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
        n_i, n_j = children[i], children[j]
        d = dist_mat[i, j]
        return n_i, n_j, d

    def get_representative(self, n):
        """
        Returns the most representative child of node n,
        Here we are just taking the child closest to the cluster's center.
        """
        children = self.g.get_children(n)
        dists = self.dists[n,children]
        i = np.argmax(dists)
        rep = children[i]
        if self.g.is_cluster(rep):
            return self.get_representative(rep)
        return rep

    @property
    def nodes(self):
        """
        Returns all nodes in the hierarchy.
        These are all nodes that do not have infinite distance.
        """
        return np.where(np.all(self.dists != np.inf, axis=1))[0].tolist()

    def upper_limit(self, n):
        """
        The upper limit is the maximum nearest distance for this cluster.
        Any nodes which have a nearest distance greater than this form a less dense region (i.e. they are too far apart).

        The upper limit just needs to be some function of
        the nearest distance mean and the nearest distance standard deviation.
        """
        return self.upper_limit_scale * (self.ndists[n][0] + self.ndists[n][1])

    def lower_limit(self, n):
        """
        The lower limit is the minimum nearest distance for this cluster.
        Any nodes which have a nearest distance smaller than this form a more dense region (i.e. they are too close together).

        The lower limit just needs to be some function of
        the nearest distance mean and the nearest distance standard deviation.
        """
        return self.lower_limit_scale * (self.ndists[n][0] - self.ndists[n][1])

    def forms_lower_dense_region(self, A, C):
        """
        Let C be a homogenous cluster.
        Given a new point A, let B be C's cluster member that is the nearest
        neighbor to A. Let d be the distance from A to B. A (and B)
        is said to form a lower dense region in C if d > U_L

        Note:
            A => new node
            B => nearest_child
        """
        assert(self.g.is_cluster(C))
        nearest_child, d = self.get_nearest_child(C, A)
        return d > self.upper_limit(C)

    def get_nearest_distances(self, n):
        dist_mat = self.cdm(n)
        np.fill_diagonal(dist_mat, np.inf)
        return np.min(dist_mat, axis=0)


    # =================================================================
    # CLUSTERING ======================================================
    # =================================================================

    def clusters(self, distance_threshold, with_labels=True):
        """
        Creates flat clusters by pruning all clusters
        with density higher than the given threshold
        and taking the leaves of the resulting hierarchy.
        Returns clusters of nodes as represented by their uiids.
        """
        # Clusters with internal node ids.
        clusters_i = [clus for clus in self.snip([self.g.root], distance_threshold)]

        # Build clusters with node uuids.
        clusters = []
        for clus in clusters_i:
            clusters.append([self.ids[id][0] for id in clus])

        # Return labels in the order that the vectors were inputted,
        # which is the same as the order of nodes by their uuids,
        # which are assigned according to when they were created.
        if with_labels:
            label_map = {}
            for i, clus in enumerate(clusters_i):
                for leaf in clus:
                    label_map[self.ids[leaf][0]] = i
            labels = [label_map[id] for id in sorted(label_map)]
            return clusters, labels
        return clusters

    def snip(self, nodes, distance_threshold):
        """
        Yields clusters that satisfy a distance threshold.
        """
        for n in nodes:
            # Reached the end of the branch,
            # just take the leaf node as a cluster.
            if not self.g.is_cluster(n):
                yield [n]
                continue

            # If this cluster satisfies the distance threshold,
            # stop this branch here.
            else:
                ndists = self.get_nearest_distances(n)
                if np.mean(ndists) <= distance_threshold:
                    yield self.g.get_leaves(n)
                    continue

            # Otherwise, keep going down the branch.
            yield from self.snip(self.g.get_children(n), distance_threshold)
