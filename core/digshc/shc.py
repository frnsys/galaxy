"""
    Similarity Histogram based Clustering
    as described on

    "Efficient Phrase-Based Document Indexing for Web Document Clustering"
"""
from dig import DocumentIndexGraph
import numpy as np

def ndists(npoints):
    return (npoints * (npoints - 1)) / 2

EPSILON = 0.02

HR_MIN = 0.5

class SimilarityHistogramClusterer(DocumentIndexGraph):
    """docstring for ClassName"""
    def __init__(self, threshold=0.15):
        super(SimilarityHistogramClusterer, self).__init__()
        self.threshold = threshold
        self.formed_clusters = []
        self.high_sim_counts = {} # counts similiraties above threshold for each cluster

    def get_histogram_ratio(self, cluster):
        if cluster.size() > 1:
            return self.high_sim_counts[cluster.id] * 1.0 / ndists(cluster.size())
        else:
            return 0

    def fit(self, plain_text_doc):
        document = self.index_document(plain_text_doc)
        self.assign_clusters(document)

    def assign_clusters(self, document):
        good_clusters = []
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
            if (hr_new >= hr_old) or (hr_new > HR_MIN and (hr_old - hr_new) < EPSILON):
                self.add_doc_to_cluster(document, cluster)
                self.high_sim_counts[cluster.id] = hr_new

        # if no similar cluster found, create new
        if not good_clusters:
            self.create_cluster(document)

    def create_cluster(self, first_doc):
        new_cluster_id = len(self.formed_clusters)
        self.formed_clusters.append(Cluster(id=new_cluster_id, first_doc=first_doc))
        self.high_sim_counts[new_cluster_id] = 0

    def add_doc_to_cluster(self, document, cluster):
        cluster.add_doc(document)


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
       