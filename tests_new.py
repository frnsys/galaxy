import unittest
import math

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import euclidean

from core.ihac.node import LeafNode, ClusterNode
from core.ihac.hierarchy import Hierarchy

class HierarchyTest(unittest.TestCase):
    def setUp(self):
        self.initial_vec = [10]
        self.h = Hierarchy(self.initial_vec)
        self.initial_node = self.h.nodes[0]

    def test_init(self):
        self.assertEqual(self.h.dists.shape, (1,1))
        self.assertEqual(self.initial_node.id, 0)

    def test_create_node(self):
        node = self.h.create_node(LeafNode, vec=[20], parent=None)

        # Distance matrix should be reshaped.
        self.assertEqual(self.h.dists.shape, (2,2))
        self.assertTrue((self.h.dists == [[0, 10], [10, 0]]).all())
        self.assertEqual(self.h.nodes, [self.initial_node, node])

        # Id should properly be assigned.
        self.assertEqual(node.id, 1)

        # Params should be passed through.
        self.assertEqual(node.center, [20])
        self.assertEqual(node.parent, None)

    def test_distance(self):
        node_i = self.initial_node
        node_j = self.h.create_node(LeafNode, vec=[20], parent=None)

        # Distances should be symmetric.
        d = self.h.distance(node_i, node_j)
        d_ = self.h.distance(node_j, node_i)
        self.assertEqual(d, 10)
        self.assertEqual(d_, 10)

    def test_update_distance(self):
        # Create some extra nodes.
        data = np.array([[1],[2],[4],[8],[12]])
        nodes = [self.h.create_node(LeafNode, vec=center, parent=None) for center in data]

        # Calculate a distance matrix independently to compare to.
        # We include the vector which initialized the hierarchy.
        data = np.insert(data, 0, self.initial_vec, axis=0)
        dist_mat = pairwise_distances(data, metric='euclidean')

        self.assertTrue((dist_mat == self.h.dists).all())


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

        # Initialize the hierarchy with the first datapoint.
        self.h = Hierarchy(self.data[0])

        # Create (leaf) nodes for each other datapoint.
        self.nodes = [self.h.nodes[0]] + [self.h.create_node(LeafNode, vec=vec, parent=None) for vec in self.data[1:]]

        # Create the cluster node to test with.
        self.c = self.h.create_node(ClusterNode, children=self.nodes, parent=None)

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
        node = self.h.create_node(LeafNode, vec=[5], parent=None)

        child, dist = self.c.get_nearest_child(node)

        # Expecting the nearest child to be the node with the center of [4].
        self.assertEqual(child, self.nodes[2])

        # It should have a distance of 1.
        self.assertEqual(dist, 1)

    def test_add_child(self):
        node = self.h.create_node(LeafNode, vec=[5], parent=None)

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

    # TO DO
    def test_split_children(self):
        pass
