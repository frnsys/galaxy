"""
    Similarity Histogram based Clustering
    as described on

    "Efficient Phrase-Based Document Indexing for Web Document Clustering"
"""
from .dig import DocumentIndexGraph
import numpy as np
import scipy


def ndists(npoints):
    return (npoints * (npoints - 1)) / 2

class SimilarityHistogramClusterer(DocumentIndexGraph):
    """docstring for ClassName"""
    def __init__(self, alpha=0.7, threshold=0.2, epsilon=0.01, hr_min=0.4):
        super(SimilarityHistogramClusterer, self).__init__(alpha=alpha)
        self.threshold = threshold
        self.epsilon = epsilon
        self.hr_min = hr_min
        self.formed_clusters = []
        # To keep counts of similarities above threshold
        # for each cluster (used in Histogram Rate calculation)
        self.high_sim_counts = {}

    def get_histogram_ratio(self, cluster):
        if cluster.size() > 1:
            return self.high_sim_counts[cluster.id] * 1.0 / ndists(cluster.size())
        else:
            return 0

    def fit(self, plain_text_doc):
        document = self.index_document(plain_text_doc)
        self.assign_clusters(document)

    def assign_clusters(self, document):
        found_clusters = False
        best_similarities = []

        # calculate similarities and add to similar clusters
        for cluster in self.formed_clusters:
            hr_old = self.get_histogram_ratio(cluster)

            # Calculation of potential histogram rate after insertion
            new_sims = np.array([self.get_sim_blend(document.id, doc_id) for doc_id in cluster.doc_ids])
            new_high_sim_count = (new_sims >= self.threshold).sum()
            hr_new_num = new_high_sim_count + self.high_sim_counts[cluster.id]
            hr_new_den = ndists(cluster.size() + 1)
            hr_new = hr_new_num * 1.0 / hr_new_den
            # import ipdb; ipdb.set_trace()
            if (hr_new >= hr_old) or (hr_new > self.hr_min and (hr_old - hr_new) < self.epsilon):
                self.add_doc_to_cluster(document, cluster)
                self.high_sim_counts[cluster.id] = hr_new
                found_clusters = True

        # if no similar cluster found, create new
        if not found_clusters:
            self.create_cluster(document)

    def create_cluster(self, first_doc):
        new_cluster_id = len(self.formed_clusters)
        self.formed_clusters.append(Cluster(id=new_cluster_id, first_doc=first_doc))
        self.high_sim_counts[new_cluster_id] = 0

    def add_doc_to_cluster(self, document, cluster):
        cluster.add_doc(document)

    def get_cluster_sim(self, cluster, doc):
        sims = np.array([self.get_sim_blend(doc.id, d_id) for d_id
                    in cluster.doc_ids if d_id != doc.id])
        return scipy.average(sims)

    def get_cluster(self, cluster_id):
        return self.formed_clusters[cluster_id]



class Cluster(object):
    """docstring for Cluster"""
    def __init__(self, id, first_doc):
        self.id = id
        self.doc_ids = [first_doc.id]

    def add_doc(self, doc):
        self.doc_ids.append(doc.id)

    def size(self):
        return len(self.doc_ids)

if __name__ == '__main__':
    docs = ["river rafting. mild river rafting. river rafting trips",
            "wild river adventures. river rafting vacation plan",
            "fishin trips. fishing vacation plan. booking fishing trips. river fishing"]

    shc = SimilarityHistogramClusterer()
    for doc in docs:
        shc.fit(doc)

    import ipdb; ipdb.set_trace()
    # print([dig.get_sim_blend(a, b) for (a, b) in [(0, 1), (1, 2), (0, 2)]])
