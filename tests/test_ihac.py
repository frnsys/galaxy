import unittest
import math

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import euclidean

from core.ihac.node import LeafNode, ClusterNode
from core.ihac.hierarchy import Hierarchy
from core.ihac import IHAC


class IHACTest(unittest.TestCase):
    def setUp(self):
        self.ihac = IHAC()

    def test_labels(self):
        points_1 = [np.array([p]) for p in np.arange(0.1, 1.0, 0.1)]
        points_2 = [np.array([p]) for p in np.arange(20.0, 21.0, 0.1)]
        points_3 = [np.array([p]) for p in np.arange(0.1, 1.0, 0.1)]
        self.ihac.fit(points_1)
        self.ihac.fit(points_2)
        self.ihac.fit(points_3)

        clusters, labels = self.ihac.clusters(distance_threshold=0.5)

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

        self.ihac.fit(points_1)
        bad_nodes = [n for n in self.ihac.hierarchy.nodes if type(n) is ClusterNode and len(n.children) <= 1]
        self.assertFalse(bad_nodes)

        self.ihac.fit(points_2)
        bad_nodes = [n for n in self.ihac.hierarchy.nodes if type(n) is ClusterNode and len(n.children) <= 1]
        self.assertFalse(bad_nodes)

    def test_load_and_save(self):
        tmp_path = '/tmp/test_hierarchy.pickle'
        points = [0.30, 0.40, 0.80, 2.70, 0.20, 2.40]
        points = [np.array([p]) for p in points]

        self.ihac.fit(points)
        self.ihac.save(tmp_path)

        ihac = IHAC()
        ihac.load(tmp_path)

        _, labels_1 = self.ihac.clusters()
        _, labels_2 = ihac.clusters()

        self.assertEqual(labels_1, labels_2)

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

        self.ihac.fit(points)

class HierarchyTest(unittest.TestCase):
    def setUp(self):
        self.initial_vecs = [[10], [30]]
        self.h = Hierarchy(*self.initial_vecs)

        # The hierarchy initially creates
        # two leaf nodes and a cluster node.
        self.initial_leaves = self.h.nodes[:2]
        self.initial_clus   = self.h.nodes[2]

    def _build_cluster_node(self, num_children=2):
        """
        Just builds a cluster node with two leaf node children.
        """
        children = [self.h.create_node(LeafNode, vec=[i*20]) for i in range(num_children)]
        return self.h.create_node(ClusterNode, children=children)

    def test_init(self):
        self.assertEqual(self.h.dists.shape, (3,3))

        for i, n in enumerate(self.initial_leaves):
            self.assertEqual(n.id, i)
            self.assertEqual(type(n), LeafNode)
        self.assertEqual(type(self.initial_clus), ClusterNode)

    def test_create_node(self):
        node = self.h.create_node(LeafNode, vec=[20])

        expected_dists = np.array([[  0., 20., 10., 10.],
                                   [ 20.,  0., 10., 10.],
                                   [ 10., 10.,  0.,  0.],
                                   [ 10., 10.,  0.,  0.]])

        # Distance matrix should be reshaped.
        self.assertEqual(self.h.dists.shape, (4,4))
        self.assertTrue((self.h.dists == expected_dists).all())
        self.assertEqual(self.h.nodes, self.initial_leaves + [self.initial_clus, node])

        # Id should properly be assigned.
        self.assertEqual(node.id, 3)

        # Params should be passed through.
        self.assertEqual(node.center, [20])

    def test_delete_node(self):
        # Create a simple hierarchy to test.
        nodes = [self.h.create_node(LeafNode, vec=[i*10]) for i in range(5)]
        children = nodes[:3]
        siblings = nodes[3:]
        self.h.leaves = nodes

        n = self.h.create_node(ClusterNode, children=children)
        parent = self.h.create_node(ClusterNode, children=siblings + [n])

        self.assertEqual(n.siblings, siblings)
        self.assertEqual(n.parent, parent)
        self.assertEqual(n.children, children)

        old_ids = [n.id] + [c.id for c in n.children]

        self.h.delete_node(n)

        # Node should be gone from the hierarchy.
        self.assertEqual(n.id, None)
        self.assertEqual(n.siblings, [])
        self.assertEqual(n.parent, None)
        self.assertTrue(n not in parent.children)
        for s in siblings:
            self.assertTrue(n not in s.siblings)

        # The node and its children's ids should be available for reuse.
        self.assertEqual(self.h.available_ids, old_ids)

        # Its children should also be deleted.
        for c in children:
            self.assertEqual(c.id, None)
            self.assertEqual(c.siblings, [])
            self.assertEqual(c.parent, None)

        # The ids should be reused.
        new_nodes = [self.h.create_node(LeafNode, vec=[i*10]) for i in range(len(old_ids))]

        for nn in new_nodes:
            self.assertTrue(nn.id in old_ids)

    def test_distance(self):
        node_k = self.h.create_node(LeafNode, vec=[20])

        # Distances should be symmetric.
        for n in self.initial_leaves:
            d = self.h.distance(n, node_k)
            d_ = self.h.distance(node_k, n)
            self.assertEqual(d, 10)
            self.assertEqual(d_, 10)

    def test_update_distances(self):
        # Create some extra nodes.
        data = np.array([[1],[2],[4],[8],[12]])
        nodes = [self.h.create_node(LeafNode, vec=center) for center in data]

        # Calculate a distance matrix independently to compare to.
        # We include the vector which initialized the hierarchy
        # and the center of the initial cluster node.
        data = np.insert(data, 0, self.initial_vecs + [self.initial_clus.center], axis=0)
        dist_mat = pairwise_distances(data, metric='euclidean')

        self.assertTrue((dist_mat == self.h.dists).all())

    def test_get_closest_leaf(self):
        node_i = self.h.incorporate([20])
        node_j = self.h.incorporate([40])
        node_k = self.h.create_node(LeafNode, vec=[21])
        result, dist = self.h.get_closest_leaf(node_k)

        self.assertEqual(result, node_i)
        self.assertEqual(dist, 1)

    def test_demote(self):
        # If N_i and N_j are sibling nodes,
        # DEMOTE(N_i, N_j)
        # will set N_j as a child to N_i.

        # Build three sibling cluster nodes.
        n_i = self._build_cluster_node()
        n_j = self._build_cluster_node()
        n_k = self._build_cluster_node()
        parent = self.h.create_node(ClusterNode, children=[n_i, n_j, n_k])

        self.h.demote(n_i, n_j)
        self.assertEqual(n_i, n_j.parent)

    def test_demote_omits_clusters_with_only_childs(self):
        # If demoting causes a ClusterNode to have only one child, that node
        # should be removed and replaced by its only child node.

        # Build two sibling cluster nodes.
        n_i = self._build_cluster_node()
        n_j = self._build_cluster_node()
        parent = self.h.create_node(ClusterNode, children=[n_i, n_j])

        self.h.demote(n_i, n_j)

        # The parent should be removed.
        self.assertEqual(parent.id, None)

        # n_i should be the root now.
        self.assertEqual(n_i.parent, None)

        self.assertTrue(n_j in n_i.children)
        self.assertEqual(n_j.parent, n_i)

    def test_merge(self):
        # If N_i and N_j are sibling nodes under a parent N,
        # MERGE(N_i, N_j)
        # will create a new cluster node, N_k, with N_i and N_j
        # as its chidlren and N as its parent.

        parent = self._build_cluster_node(num_children=3)
        n_i, n_j, n_p = parent.children

        # All three nodes are siblings.
        self.assertEqual(n_p.siblings, [n_i, n_j])

        self.h.merge(n_i, n_j)

        # The old parent should be replaced.
        n_k = n_i.parent
        self.assertNotEqual(n_k, parent)

        # Now n_i and n_j should be in their own cluster.
        self.assertEqual(n_i.siblings, [n_j])

        # And that new cluster should be a sibling to the remaining node.
        self.assertEqual(n_p.siblings, [n_k])

    def test_split(self):
        # If N_k is a cluster node with a set of children S_k,
        # SPLIT(θ, N_k)
        # will split N_k into two new nodes, N_i and N_j, each with a different
        # subset of S_k (S_i and S_j, respectively). S_k is split by disconnecting an
        # edge in N_k's minimum spanning tree (MST).

        n_i = self.h.incorporate([10.05])
        n_j = self.h.incorporate([10.08])
        n_k = self.h.incorporate([10.1])

        self.assertEqual(self.initial_leaves[0].siblings, [n_i, n_j, n_k])
        self.h.split(self.initial_leaves[0].parent)

        # n_i, n_j, and n_k are all closer to each other than they are to the initial leaf,
        # so we expect them to be in their own cluster.
        self.assertEqual(n_i.siblings, [n_j, n_k])
        self.assertEqual(self.initial_leaves[0].siblings, [self.initial_leaves[1], n_i.parent])

    def test_restructure(self):
        # This isn't a comprehensive test, but a simple check.

        # As the only two nodes initially, these two are siblings.
        self.assertEqual(self.initial_leaves[0].siblings, [self.initial_leaves[1]])

        # Add some new nodes very close to the first leaf node ([10]).
        n_i = self.h.incorporate([10.05])
        n_j = self.h.incorporate([10.08])
        n_k = self.h.incorporate([10.1])

        # The second leaf node ([30]) should be different enough that it should
        # have moved to its own cluster.
        self.assertNotEqual(self.initial_leaves[0].siblings, [self.initial_leaves[1]])

    def test_ins_hierarchy(self):
        # If N_i is a node in the hierarchy, child to node N,
        # and N_j is a node not yet in the hierarchy,
        # INS_HIERARCHY(N_i, N_j)
        # creates a new node N_k with children N_i and N_j and N as its parent.

        # Build a node with a parent and another node.
        n_i = self._build_cluster_node().children[0]
        n_j = self.h.create_node(LeafNode, vec=[20])

        self.h.ins_hierarchy(n_i, n_j)

        # The nodes should be siblings now.
        self.assertEqual(n_i.siblings, [n_j])

    def test_incorporate_adds_to_existing_cluster_node(self):
        # A node this close should be added as a sibling.
        node_i = self.h.incorporate([11])
        self.assertEqual(self.initial_leaves[0].siblings, [node_i])

    def test_incorporate_creates_new_cluster_node(self):
        # The cluster node and the new node should be siblings.
        node_i = self.h.incorporate([60])
        self.assertEqual(self.initial_clus.siblings, [node_i])

    def test_fcluster(self):
        node_i = self.h.incorporate([20])
        node_j = self.h.incorporate([40])

        clusters = self.h.fcluster(distance_threshold=12.0)
        self.assertEqual(clusters, [self.initial_leaves + [node_i], [node_j]])

        clusters = self.h.fcluster(distance_threshold=30.0)
        self.assertEqual(clusters, [self.initial_leaves + [node_i, node_j]])

        clusters = self.h.fcluster(distance_threshold=0.0)
        self.assertEqual(clusters, [[self.initial_leaves[0]], [self.initial_leaves[1]], [node_i], [node_j]])

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
        self.h = Hierarchy(*self.data[:2])

        # Create (leaf) nodes for each other datapoint.
        self.nodes = self.h.nodes[:2] + [self.h.create_node(LeafNode, vec=vec) for vec in self.data[2:]]

        # Create the cluster node to test with.
        self.c = self.h.create_node(ClusterNode, children=self.nodes)

    def test_init(self):
        c = self.c
        self.assertIsInstance(c, ClusterNode)

        # The center should be the mean of the datapoints.
        expected_center = (1+2+4+8+12)/5
        self.assertEquals(c.center, expected_center)

        # The mean of the nearest distances.
        mins = [1,1,4,2,4]
        expected_nearest_distance_mean = sum(mins)/len(mins)
        self.assertEqual(c.nearest_dists_mean, expected_nearest_distance_mean)

        # The std of the nearest distances.
        expected_nearest_distance_std = math.sqrt(sum([(min - expected_nearest_distance_mean)**2 for min in mins])/len(mins))
        self.assertAlmostEqual(c.nearest_dists_std, expected_nearest_distance_std)

    def test_nearest_children(self):
        n_i, n_j, d = self.c.nearest_children
        self.assertEqual(n_i, self.nodes[0])
        self.assertEqual(n_j, self.nodes[1])
        self.assertEqual(d, 1.0)

    def test_furthest_nearest_children(self):
        n_i, n_j, d = self.c.furthest_nearest_children
        self.assertEqual(n_i, self.nodes[2])
        self.assertEqual(n_j, self.nodes[3])
        self.assertEqual(d, 4.0)

    def test_leaves(self):
        self.assertEqual(self.c.leaves, self.nodes)

    def test_siblings(self):
        idx = 2
        child = self.c.children[idx]
        siblings = self.c.children[:idx] + self.c.children[(idx + 1):]
        self.assertEqual(child.siblings, siblings)

    def test_child_dist_matrix(self):
        # Independently calculate the child distance matrix.
        c_dist_mat = pairwise_distances(self.data, metric='euclidean')

        # Sanity check that the master distance matrix and the child distance matrix
        # are NOT the same.
        self.assertFalse(self.c.mdm == c_dist_mat)

        # Assert that the child distance matrix equals the independently calculated one.
        self.assertTrue((self.c.cdm == c_dist_mat).any())

    def test_nearest_child(self):
        node = self.h.create_node(LeafNode, vec=[5])

        child, dist = self.c.get_nearest_child(node)

        # Expecting the nearest child to be the node with the center of [4].
        self.assertEqual(child, self.nodes[2])

        # It should have a distance of 1.
        self.assertEqual(dist, 1)

    def test_add_child(self):
        node = self.h.create_node(LeafNode, vec=[5])

        old_dist_mat = self.h.dists.copy()

        self.c.add_child(node)

        # The center should have been updated.
        expected_center = (1+2+4+8+12+5)/6
        self.assertEqual(self.c.center, expected_center)

        # The distance matrix shape should remain the same,
        # but the values should not (they have been updated).
        self.assertEqual(old_dist_mat.shape, self.h.dists.shape)
        self.assertFalse((old_dist_mat == self.h.dists).all())

        # Independently calculate the child distance matrix and compare.
        data = np.concatenate((self.data, [[5]]))
        c_dist_mat = pairwise_distances(data, metric='euclidean')
        self.assertTrue((c_dist_mat == self.c.cdm).all())

    def test_remove_child(self):
        old_dist_mat = self.h.dists.copy()

        child = self.c.children[2]
        self.c.remove_child(child)

        # The center should have been updated.
        expected_center = (1+2+8+12)/4
        self.assertEqual(self.c.center, expected_center)

        # The distance matrix shape should remain the same,
        # but the values should not (they have been updated).
        self.assertEqual(old_dist_mat.shape, self.h.dists.shape)
        self.assertFalse((old_dist_mat == self.h.dists).all())

        # Independently calculate the child distance matrix and compare.
        data = np.delete(self.data, 2, 0)
        c_dist_mat = pairwise_distances(data, metric='euclidean')
        self.assertTrue((c_dist_mat == self.c.cdm).all())

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

        children = self.c.children.copy()
        c_i, c_j = self.c.split_children()

        expected_i_children = [children[i] for i in [0,1,2]]
        expected_j_children = [children[i] for i in [3,4]]

        self.assertEqual(c_i.children, expected_i_children)
        self.assertEqual(c_j.children, expected_j_children)
