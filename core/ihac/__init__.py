from sklearn.externals import joblib

from .hierarchy import Hierarchy


class IHAC():
    def __init__(self):
        self.hierarchy = None

    def fit(self, vecs):
        """
        Incorporate vectors into the hierarchy.
        """
        if self.hierarchy is None:
            if len(vecs) < 2:
                raise Exception('You must initialize the model with at least two vectors.')
            self.hierarchy = Hierarchy(*vecs[:2])
            vecs = vecs[2:]

        for vec in vecs:
            self.hierarchy.incorporate(vec)

    def clusters(self, distance_threshold=None, with_labels=True):
        """
        Return the vectors in clusters defined according to some distance threshold.
        """
        clusters = self.hierarchy.fcluster(distance_threshold)

        # Return labels in the order that the vectors were inputted,
        # which is the same as the order of nodes by their uuids,
        # which are assigned according to when they were created.
        if with_labels:
            label_map = {}
            for i, clus in enumerate(clusters):
                for leaf in clus:
                    label_map[leaf.uuid] = i
            labels = [label_map[id] for id in sorted(label_map)]
            return clusters, labels
        return clusters

    def save(self, savepath='ihac.pickle'):
        """
        Saves the hierarchy to file.
        Using joblib for now since it's better than pickle
        and easy to use.
        http://scikit-learn.org/stable/modules/model_persistence.html#model-persistence
        """
        joblib.dump(self.hierarchy, savepath)

    def load(self, savepath='ihac.pickle'):
        """
        Loads the hierarchy from a file.
        """
        self.hierarchy = joblib.load(savepath)
