from eval.data import load_articles
from core.digbc import DocumentIndexGraphClusterer
import numpy as np
from scipy import argmax
from eval import score


docs, true_labels = load_articles('../../eval/data/event/handpicked.json')

ntrueclusters = len(set(true_labels))

# docs = docs[:10]

for threshold in np.arange(0.00295, 0.00330, 0.00005):
    dig = DocumentIndexGraphClusterer(threshold=threshold, hard=False)

    for idx, doc in enumerate(docs):
        dig.index_document(doc.text)

    doc_clus_map = {}
    for idx, clus in enumerate(dig.formed_clusters):
        for doc_id in clus.doc_ids:
            doc_clus_map.setdefault(doc_id,[])
            doc_clus_map[doc_id].append(idx)
    
    labels = []
    for id in sorted(doc_clus_map):
        clusters = [dig.get_cluster(cl_id) for cl_id in doc_clus_map[id]]
        sims = [dig.get_cluster_sim(cl, dig.get_doc(id)) for cl in clusters]
        max_i = argmax(sims)
        labels.append(clusters[max_i].id)

    nclusters = len(set(labels))
    print("%.5f : %d" % (threshold, nclusters))
    if nclusters == ntrueclusters:
        print("Cluster number matches for threshold = %.5f" % threshold)
        break

print(score(labels, true_labels))
