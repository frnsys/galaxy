import unittest
import math

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.metrics.pairwise import pairwise_distances

from galaxy.cluster.ihac import Hierarchy

# We should not get any numpy warnings since all operations should work
# if the hierarchy is working properly. So if we get a warning, we want to treat it as an error.
np.seterr(invalid='raise')

class ClusteringTest(unittest.TestCase):
    def setUp(self):
        self.h = Hierarchy()

    def test_labels(self):
        points_1 = [np.array([p]) for p in np.arange(0.1, 1.0, 0.1)]
        points_2 = [np.array([p]) for p in np.arange(20.0, 21.0, 0.1)]
        points_3 = [np.array([p]) for p in np.arange(0.1, 1.0, 0.1)]
        self.h.fit(points_1)
        self.h.fit(points_2)
        self.h.fit(points_3)

        clusters, labels = self.h.clusters(distance_threshold=0.5)

        # Expect that the labels preserve the input order.
        num_1 = len(points_1)
        labels_1 = labels[:num_1]

        num_2 = len(points_2)
        labels_2 = labels[num_1:num_1+num_2]

        labels_3 = labels[num_1+num_2:]

        # labels_1 and labels_3 are operating off the same data (points) so they should be equivalent.
        self.assertEqual(labels_1, labels_3)
        self.assertNotEqual(labels_1, labels_2)

    def test_no_cluster_nodes_with_single_cluster_child(self):
        points = [0.30, 0.40, 0.80, 2.70, 0.20, 2.40]
        points = [np.array([p]) for p in points]
        points_1, points_2 = points[:4], points[4:]

        self.h.fit(points_1)
        bad_nodes = [n for n in self.h.nodes if self.h.g.is_cluster(n) and self.h.g.get_children(n).size <= 1]
        self.assertFalse(bad_nodes)

        self.h.fit(points_2)
        bad_nodes = [n for n in self.h.nodes if self.h.g.is_cluster(n) and self.h.g.get_children(n).size <= 1]
        self.assertFalse(bad_nodes)

    def test_many_points(self):
        """
        Test clustering with 160 points.
        This should just execute without error.
        """
        cluster_a1 = np.arange(0,0.4,0.01)
        cluster_a2 = np.arange(0.6,1,0.01)
        cluster_b1 = np.arange(2,2.4,0.01)
        cluster_b2 = np.arange(2.6,3,0.01)
        points = np.append(cluster_a1, cluster_a2)
        points = np.append(points, cluster_b1)
        points = np.append(points, cluster_b2)
        np.random.shuffle(points)

        points = [np.array([p]) for p in points]
        self.h.fit(points)

class HierarchyTest(unittest.TestCase):
    def setUp(self):
        self.initial_vecs = [[10], [30]]
        self.h = Hierarchy(metric='euclidean',
                           lower_limit_scale=0.1,
                           upper_limit_scale=1.5)
        self.h.fit(self.initial_vecs)
        self.initial_leaves = [0,1]
        self.initial_clus   = 2

    def _build_cluster_node(self, num_children=2):
        """
        Just builds a cluster node with two leaf node children.
        """
        children = [self.h.create_node(vec=[i*20]) for i in range(num_children)]
        return self.h.create_node(children=children)

    def test_init(self):
        # The dist and graph matrices are square (nxn).
        self.assertEqual(self.h.dists.shape, (3,3))
        self.assertEqual(self.h.g.mx.shape, (3,3))

        # The centers matrix is nxm.
        self.assertEqual(self.h.centers.shape, (3,1))

    def test_fit_returns_uuids(self):
        vecs = [[20], [30], [40]]
        new_uuids = self.h.fit(vecs)
        self.assertEqual(new_uuids, [3,4,6])

    def test_save_and_load(self):
        ids     = self.h.ids
        graph   = self.h.g.mx
        dists   = self.h.dists
        ndists  = self.h.ndists
        centers = self.h.centers
        avail   = self.h.available_ids

        self.h.save('/tmp/hierarchy.ihac')

        h = Hierarchy.load('/tmp/hierarchy.ihac')
        assert_array_equal(graph,   h.g.mx)
        assert_array_equal(dists,   h.dists)
        assert_array_equal(ids,     h.ids.read())
        assert_array_equal(ndists,  h.ndists.read())
        assert_array_equal(centers, h.centers.read())
        assert_array_equal(avail,   h.available_ids.read())

    def test_create_node(self):
        node = self.h.create_node(vec=[20])

        expected_dists = np.array([[  0., 20., 10., 10.],
                                   [ 20.,  0., 10., 10.],
                                   [ 10., 10.,  0.,  0.],
                                   [ 10., 10.,  0.,  0.]])

        # Distance matrix should be reshaped.
        self.assertEqual(self.h.dists.shape, (4,4))
        self.assertTrue((self.h.dists == expected_dists).all())
        self.assertEqual(self.h.nodes, self.initial_leaves + [self.initial_clus, node])

        # Id should properly be assigned.
        self.assertEqual(node, 3)

        # Params should be passed through.
        self.assertEqual(self.h.centers[node], [20])

    def test_delete_node(self):
        # Create a simple hierarchy to test.
        nodes = [self.h.create_node(vec=[i*10]) for i in range(5)]
        children = nodes[:3]
        siblings = nodes[3:]

        n = self.h.create_node(children=children)
        parent = self.h.create_node(children=siblings + [n])

        assert_array_equal(self.h.g.get_siblings(n), siblings)
        assert_array_equal(self.h.g.get_parent(n), parent)
        assert_array_equal(self.h.g.get_children(n), children)

        old_ids = [n] + [c for c in self.h.g.get_children(n)]

        self.h.delete_node(n)

        # Node should be gone from the hierarchy.
        assert_array_equal(self.h.g.get_siblings(n), [])
        self.assertEqual(self.h.g.get_parent(n), None)
        self.assertTrue(n not in self.h.g.get_children(parent))
        for s in siblings:
            self.assertTrue(n not in self.h.g.get_siblings(s))

        # The node and its children's ids should be available for reuse.
        self.assertEqual(set(self.h.available_ids), set(old_ids))

        # Its children should also be deleted.
        for c in children:
            self.assertEqual(self.h.g.get_siblings(c), [])
            self.assertEqual(self.h.g.get_parent(c), None)

        # The ids should be reused.
        new_nodes = [self.h.create_node(vec=[i*10]) for i in range(len(old_ids))]

        for nn in new_nodes:
            self.assertTrue(nn in old_ids)

    def test_demote(self):
        # If N_i and N_j are sibling nodes,
        # DEMOTE(N_i, N_j)
        # will set N_j as a child to N_i.

        # Build three sibling cluster nodes.
        n_i = self._build_cluster_node()
        n_j = self._build_cluster_node()
        n_k = self._build_cluster_node()
        parent = self.h.create_node(children=[n_i, n_j, n_k])

        self.h.demote(n_i, n_j)
        self.assertEqual(n_i, self.h.g.get_parent(n_j))

    def test_demote_omits_clusters_with_only_childs(self):
        # If demoting causes a cluster node to have only one child, that node
        # should be removed and replaced by its only child node.

        # Build two sibling cluster nodes.
        n_i = self._build_cluster_node()
        n_j = self._build_cluster_node()
        parent = self.h.create_node(children=[n_i, n_j])

        self.h.demote(n_i, n_j)

        # The parent should be removed.
        self.assertFalse(parent in self.h.nodes)

        # n_i should be the root now.
        self.assertTrue(self.h.g.is_root(n_i))

        self.assertTrue(n_j in self.h.g.get_children(n_i))
        self.assertEqual(self.h.g.get_parent(n_j), n_i)

    def test_merge(self):
        # If N_i and N_j are sibling nodes under a parent N,
        # MERGE(N_i, N_j)
        # will create a new cluster node, N_k, with N_i and N_j
        # as its chidlren and N as its parent.

        parent = self._build_cluster_node(num_children=3)
        n_i, n_j, n_p = self.h.g.get_children(parent)

        # All three nodes are siblings.
        self.assertEqual(self.h.g.get_siblings(n_p), [n_i, n_j])

        self.h.merge(n_i, n_j)

        # The old parent should be replaced.
        n_k = self.h.g.get_parent(n_i)
        self.assertNotEqual(n_k, parent)

        # Now n_i and n_j should be in their own cluster.
        self.assertEqual(self.h.g.get_siblings(n_i), [n_j])

        # And that new cluster should be a sibling to the remaining node.
        self.assertEqual(self.h.g.get_siblings(n_p), [n_k])

    def test_split(self):
        # If N_k is a cluster node with a set of children S_k,
        # SPLIT(θ, N_k)
        # will split N_k into two new nodes, N_i and N_j, each with a different
        # subset of S_k (S_i and S_j, respectively). S_k is split by disconnecting an
        # edge in N_k's minimum spanning tree (MST).

        n_i = self.h.to_iid(self.h.incorporate([10.05]))
        n_j = self.h.to_iid(self.h.incorporate([10.08]))
        n_k = self.h.to_iid(self.h.incorporate([10.10]))

        sibs = self.h.g.get_siblings(self.initial_leaves[0])
        parent = self.h.g.get_parent(self.initial_leaves[0])
        self.assertEqual(sibs, [n_i, n_j, n_k])
        self.h.split(parent)

        # n_i, n_j, and n_k are all closer to each other than they are to the initial leaf,
        # so we expect them to be in their own cluster.
        self.assertEqual(self.h.g.get_siblings(n_i), [n_j, n_k])
        self.assertEqual(self.h.g.get_siblings(self.initial_leaves[0]), [self.initial_leaves[1], self.h.g.get_parent(n_i)])

    def test_restructure(self):
        # This isn't a comprehensive test, but a simple check.

        # As the only two nodes initially, these two are siblings.
        self.assertEqual(self.h.g.get_siblings(self.initial_leaves[0]), [self.initial_leaves[1]])

        # Add some new nodes very close to the first leaf node ([10]).
        n_i = self.h.to_iid(self.h.incorporate([10.05]))
        n_j = self.h.to_iid(self.h.incorporate([10.08]))
        n_k = self.h.to_iid(self.h.incorporate([10.10]))

        # The second leaf node ([30]) should be different enough that it should
        # have moved to its own cluster.
        self.assertNotEqual(self.h.g.get_siblings(self.initial_leaves[0]), [self.initial_leaves[1]])

    def test_ins_hierarchy(self):
        # If N_i is a node in the hierarchy, child to node N,
        # and N_j is a node not yet in the hierarchy,
        # INS_HIERARCHY(N_i, N_j)
        # creates a new node N_k with children N_i and N_j and N as its parent.

        # Build a node with a parent and another node.
        n_i = self.h.g.get_children(self._build_cluster_node())[0]
        n_j = self.h.create_node(vec=[20])

        self.h.ins_hierarchy(n_i, n_j)

        # The nodes should be siblings now.
        self.assertEqual(self.h.g.get_siblings(n_i), [n_j])

    def test_incorporate_adds_to_existing_cluster_node(self):
        # A node this close should be added as a sibling.
        node_i = self.h.to_iid(self.h.incorporate([11]))
        self.assertEqual(self.h.g.get_siblings(self.initial_leaves[0]), [node_i])

    def test_incorporate_creates_new_cluster_node(self):
        # The cluster node and the new node should be siblings.
        node_i = self.h.to_iid(self.h.incorporate([90]))
        self.assertEqual(self.h.g.get_siblings(self.initial_clus), [node_i])

    def test_prune(self):
        self.h.fit([[20], [30]])

        self.assertEqual(self.h.available_ids, [])
        assert_array_equal(self.h.g.leaves, [0,1,3,4])

        self.h.prune([5])

        self.assertEqual(self.h.available_ids, [1,4,5])
        assert_array_equal(self.h.g.leaves, [0,3])
        assert_array_equal(self.h.nodes, [0,2,3])

    def test_clusters(self):
        node_i = self.h.to_iid(self.h.incorporate([90]))
        node_j = self.h.to_iid(self.h.incorporate([40]))

        clusters = self.h.clusters(distance_threshold=14.0, with_labels=False)
        self.assertEqual(clusters, [self.initial_leaves + [node_j], [node_i]])

        clusters = self.h.clusters(distance_threshold=71.0, with_labels=False)
        self.assertEqual(clusters, [self.initial_leaves + [node_j, node_i]])

        clusters = self.h.clusters(distance_threshold=0.0, with_labels=False)
        self.assertEqual(clusters, [[self.initial_leaves[0]], [self.initial_leaves[1]], [node_j], [node_i]])


class DistancesTest(unittest.TestCase):
    """
    Tests the management of distances (in the dists matrix).
    """
    def setUp(self):
        self.vecs = [[10], [20], [0], [20]]
        self.initial_vecs =  self.vecs[:2]
        self.h = Hierarchy(metric='euclidean',
                           lower_limit_scale=0.1,
                           upper_limit_scale=1.5)
        self.h.fit(self.initial_vecs)

        children = [self.h.create_node(vec=vec) for vec in self.vecs[2:]]
        n = self.h.create_node(children=children)
        self.h.g.add_child(2, n)

        self.leaves   = [0,1,3,4]
        self.clusters = [2,5]

    def test_distance(self):
        node_k = self.h.create_node(vec=[20])

        # Distances should be symmetric.
        for n in self.leaves:
            d = self.h.get_distance(n, node_k)
            d_ = self.h.get_distance(node_k, n)
            self.assertEqual(d, d_)

    def test_update_distances(self):
        # Create some extra nodes.
        data = np.array([[1],[2],[4],[8],[12]])
        nodes = [self.h.create_node(vec=center) for center in data]

        # Calculate a distance matrix independently to compare to.
        # We include the vector which initialized the hierarchy
        # and the center of the initial cluster node.
        old_data = self.initial_vecs + [self.h.centers[self.clusters[0]]] + self.vecs[2:] + [self.h.centers[self.clusters[1]]]
        data = np.insert(data, 0, old_data, axis=0)
        dist_mat = pairwise_distances(data, metric='euclidean')

        self.assertTrue((dist_mat == self.h.dists).all())

    def test_cdm(self):
        # Expecting the matrix to have rows and columns 0,1,n (n=5)
        # since those are the child nodes.
        expected = [[ 0., 10.,  0.],
                    [10.,  0., 10.],
                    [ 0., 10.,  0.]]
        assert_array_equal(expected, self.h.cdm(2))

    def test_get_closest_leaf(self):
        node_k = self.h.create_node(vec=[11])
        result, dist = self.h.get_closest_leaf(node_k)
        self.assertEqual(result, self.leaves[0])
        self.assertEqual(dist, 1)

    def test_get_nearest_distances(self):
        d = self.h.get_nearest_distances(2)
        expected = [ 0., 10.,  0.]
        assert_array_equal(expected, d)

    def test_get_nearest_children(self):
        i, j, d = self.h.get_nearest_children(2)
        self.assertEqual(i, 0)
        self.assertEqual(j, 5)
        self.assertEqual(d, 0)

    def test_get_furthest_nearest_children(self):
        i, j, d = self.h.get_furthest_nearest_children(2)
        self.assertEqual(i, 0)
        self.assertEqual(j, 1)
        self.assertEqual(d, 10)

    def test_get_representative(self):
        r = self.h.get_representative(2)
        self.assertEqual(r, 0)

    def test_most_representative(self):
        # Incorporating these vectors puts the center of all nodes around ~22
        new_vecs = [[30], [40], [40]]
        self.h.fit(new_vecs)

        nodes = self.h.nodes
        rep = self.h.most_representative(nodes)

        # Expecting that the representative node is 1, w/ a center of [20]
        self.assertEqual(rep, 1)


class ClusterNodeTest(unittest.TestCase):
    def setUp(self):
        """
        Keep it simple: 5 1-dimensional datapoints::

            [[1],
             [2],
             [4],
             [8],
             [12]]

        The child distance matrix will look like::

            [[  0.   1.   3.   7.  11.]
             [  1.   0.   2.   6.  10.]
             [  3.   2.   0.   4.   8.]
             [  7.   6.   4.   0.   4.]
             [ 11.  10.   8.   4.   0.]]
        """
        self.data = np.array([[1],[2],[4],[8],[12]])

        # Initialize the hierarchy with the first two datapoints.
        self.h = Hierarchy()
        self.h.fit(self.data[:2])

        # Create (leaf) nodes for each other datapoint.
        self.nodes = self.h.nodes[:2] + [self.h.create_node(vec=vec) for vec in self.data[2:]]

        # Create the cluster node to test with.
        self.c = self.h.create_node(children=self.nodes)

    def test_init(self):
        c = self.c
        self.assertTrue(self.h.g.is_cluster(c))

        # The center should be the mean of the datapoints.
        expected_center = (1+2+4+8+12)/5
        self.assertEquals(self.h.centers[c], expected_center)

        # The mean of the nearest distances.
        mins = [1,1,4,2,4]
        expected_nearest_distance_mean = sum(mins)/len(mins)
        self.assertEqual(np.mean(self.h.get_nearest_distances(c)), expected_nearest_distance_mean)

        # The std of the nearest distances.
        expected_nearest_distance_std = math.sqrt(sum([(min - expected_nearest_distance_mean)**2 for min in mins])/len(mins))
        self.assertAlmostEqual(np.std(self.h.get_nearest_distances(c)), expected_nearest_distance_std)

    def test_split_children(self):
        """
        The MST for self.c looks like::

            [[ 0.  1.  0.  0.  0.]
             [ 0.  0.  2.  0.  0.]
             [ 0.  0.  0.  4.  0.]
             [ 0.  0.  0.  0.  4.]
             [ 0.  0.  0.  0.  0.]]

        Visually, the MST looks something like::

            (0)--1--(1)--2--(2)--4--(3)--3--(4)

        Where::

            (A)--C--(B)

        Means A and B are connected by an edge with weight C.

        So splitting at the greatest edge, we get::

            (0)--1--(1)--2--(2)   (3)--3--(4)
        """

        children = self.h.g.get_children(self.c).copy()
        c_i, c_j = self.h.split(self.c)

        expected_i_children = [children[i] for i in [0,1,2]]
        expected_j_children = [children[i] for i in [3,4]]

        assert_array_equal(self.h.g.get_children(c_i), expected_i_children)
        assert_array_equal(self.h.g.get_children(c_j), expected_j_children)


class GraphTest(unittest.TestCase):
    """
    Test the managing of the adjacency matrix.
    """

    def setUp(self):
        self.initial_vecs = [[10], [20]]
        self.h = Hierarchy(metric='euclidean',
                           lower_limit_scale=0.1,
                           upper_limit_scale=1.5)
        self.h.fit(self.initial_vecs)
        self.g = self.h.g

        self.extra_vecs = [[0], [20]]
        children = [self.h.create_node(vec=vec) for vec in self.extra_vecs]
        n = self.h.create_node(children=children)
        self.g.add_child(2, n)

        self.leaves   = [0,1,3,4]
        self.clusters = [2,5]

    def test_is_cluster(self):
        for clus in self.clusters:
            self.assertTrue(self.g.is_cluster(clus))

        for l in self.leaves:
            self.assertFalse(self.g.is_cluster(l))

    def test_is_root(self):
        clus = self.clusters[0]
        self.assertTrue(self.g.is_root(clus))
        self.assertFalse(self.g.is_root(self.clusters[1]))

        for l in self.leaves:
            self.assertFalse(self.g.is_root(l))

        self.assertEqual(clus, self.g.root)

        # Try making a new root.
        # If the existing root becomes a child, its parent
        # is the new root.
        new_root = self.h.create_node(children=[clus, 4])

        self.assertTrue(self.g.is_root(new_root))
        self.assertFalse(self.g.is_root(clus))
        self.assertEqual(new_root, self.g.root)

    def test_leaves(self):
        assert_array_equal(self.g.leaves, [0,1,3,4])

    def test_nodes(self):
        assert_array_equal(self.h.nodes, [0,1,2,3,4,5])

    def test_get_parent(self):
        for n in [0,1,5]:
            assert_array_equal(self.g.get_parent(n), 2)
        for n in [3,4]:
            assert_array_equal(self.g.get_parent(n), 5)
        assert_array_equal(self.g.get_parent(2), None)

    def test_get_children(self):
        assert_array_equal(self.g.get_children(2), [0,1,5])
        assert_array_equal(self.g.get_children(5), [3,4])

        for l in self.leaves:
            assert_array_equal(self.g.get_children(l), [])

    def test_reset_node(self):
        self.g.reset_node(2)
        assert_array_equal(self.g.get_children(2), [])

    def test_get_siblings(self):
        assert_array_equal(self.g.get_siblings(0), [1,5])
        assert_array_equal(self.g.get_siblings(1), [0,5])
        assert_array_equal(self.g.get_siblings(5), [0,1])
        assert_array_equal(self.g.get_siblings(2), [])
        assert_array_equal(self.g.get_siblings(4), [3])
        assert_array_equal(self.g.get_siblings(3), [4])

    def test_get_leaves(self):
        assert_array_equal(self.g.get_leaves(0), [0])
        assert_array_equal(self.g.get_leaves(1), [1])
        assert_array_equal(self.g.get_leaves(2), [0,1,3,4])
        assert_array_equal(self.g.get_leaves(5), [3,4])

    def test_add_child(self):
        n = self.h.create_node(vec=[80])
        self.g.add_child(2, n)

        self.assertEqual(self.g.get_parent(n), 2)
        self.assertTrue(n in self.g.get_children(2))

        # Changing children should automatically remove it from its previous parent.
        self.g.add_child(5, n)
        self.assertNotEqual(self.g.get_parent(n), 2)
        self.assertEqual(self.g.get_parent(n), 5)
        self.assertFalse(n in self.g.get_children(2))
        self.assertTrue(n in self.g.get_children(5))

    def test_remove_child(self):
        n = self.h.create_node(vec=[80])
        self.g.add_child(2, n)

        self.assertEqual(self.g.get_parent(n), 2)
        self.assertTrue(n in self.g.get_children(2))
        self.g.remove_child(2, n)

        self.assertNotEqual(self.g.get_parent(n), 2)
        self.assertFalse(n in self.g.get_children(2))
