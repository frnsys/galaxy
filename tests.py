import unittest

import numpy as np

from core.ihacluster import IHAClusterer

class IHACTest(unittest.TestCase):
    def setUp(self):
        self.clusterer = IHAClusterer()

    def tearDown(self):
        pass

    def test_simple_clustering(self):
        points = generate_points()
        self.clusterer.fit(points)

    def test_many_clusters(self):
        points = generate_points(num_clusters=200)
        self.clusterer.fit(points)


def generate_points(num_clusters=2):
    # Generate num_clusters 1d clusters.
    points = np.array([])
    for i in range(num_clusters):
        clus = np.arange(i, i+1, 0.1)
        points = np.append(points, clus)
    np.random.shuffle(points)

    return [np.array([p]) for p in points]
