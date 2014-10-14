#!/usr/bin/env python # -*- coding: utf-8
import unittest
import numpy as np
from core.ihacluster import IHAClusterer


def generate_points(num_clusters=2):
    # Generate num_clusters 1d clusters.
    points = np.array([])
    for i in range(num_clusters):
        clus = np.arange(2 * i, 2 * i + 1, 0.1)
        points = np.append(points, clus)
    np.random.shuffle(points)

    return [np.array([p]) for p in points]


class IHACTest(unittest.TestCase):
    # def setUp(self):
    #     pass

    # def tearDown(self):
    #     pass

    def test_simple_clustering(self):
        points = generate_points()
        clusterer = IHAClusterer()
        clusterer.fit(points)
        self.assertTrue(True)

    def test_many_clusters(self):
        pass
        points = generate_points(num_clusters=10)
        clusterer = IHAClusterer()
        clusterer.fit(points)
        self.assertTrue(True)

    def test_load_and_save(self):
        clusterer = IHAClusterer()
        points = generate_points(num_clusters=5)
        clusterer.fit(points)

        half = int(len(points)/2)
        points_1, points_2 = points[:half], points[half:]

        first_clusterer = IHAClusterer()
        first_clusterer.fit(points_1)
        first_clusterer.save("./")

        second_clusterer = IHAClusterer()
        second_clusterer.load("./")
        second_clusterer.fit_more(points_2)

        labels_1 = clusterer.get_labels()
        labels_2 = second_clusterer.get_labels()
        self.assertListEqual(labels_1, labels_2)

if __name__ == '__main__':
    unittest.main()
#     iht = IHACTest()
#     iht.setUp()
#     iht.test_load_and_save()
