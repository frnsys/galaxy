from .hierarchy import Hierarchy

class IHAC():
    def __init__(self):
        self.hierarchy = None

    def fit(self, vecs):
        if self.hierarchy is None:
            self.hierarchy = Hierarchy(vecs[0])
            vecs = vecs[1:]

        for vec in vecs:
            self.hierarchy.incorporate(vec)

    def clusters(self, distance_threshold=None, with_labels=True):
        clusters = self.hierarchy.fcluster(distance_threshold)

        # Return labels in the order that the vectors were inputted,
        # which is the same as the order of nodes by their ids.
        if with_labels:
            label_map = {}
            for i, clus in enumerate(clusters):
                for leaf in clus:
                    label_map[leaf.id] = i
            labels = [label_map[id] for id in sorted(label_map)]
            return clusters, labels
        return clusters
