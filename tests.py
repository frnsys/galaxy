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

    def test_load_and_save(self):
        points = generate_points(num_clusters=10)
        self.clusterer.fit(points)

        half = int(len(points)/2)
        points_1, points_2 = points[:half], points[half:]

        first_clusterer = IHAClusterer()
        first_clusterer.fit(points_1)
        first_clusterer.save("./")

        second_clusterer = IHAClusterer()
        second_clusterer.load("./")
        second_clusterer.fit_more(points_2)

        print(self.clusterer.labels_)
        print(second_clusterer.labels_)

        # import ipdb; ipdb.set_trace()


def generate_points(num_clusters=2):
    # Generate num_clusters 1d clusters.
    points = np.array([])
    for i in range(num_clusters):
        clus = np.arange(2 * i, 2 * i + 1, 0.1)
        points = np.append(points, clus)
    np.random.shuffle(points)

    return [np.array([p]) for p in points]

if __name__ == '__main__':
    iht = IHACTest()
    iht.setUp()
    iht.test_load_and_save()
