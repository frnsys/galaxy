from eval.data import load_articles
from core.digbc.dig import DocumentIndexGraphClusterer
import numpy as np



docs, true_labels = load_articles('../../eval/data/event/handpicked.json')

# docs = docs[:10]

for threshold in np.arange(0.0003, 0.001, 0.00005):
	dig = DocumentIndexGraphClusterer(threshold=threshold)

	for idx, doc in enumerate(docs):
	    dig.index_document(doc.text)

	doc_clus_map = {}
	for idx, clus in enumerate(dig.formed_clusters):
	    for doc_id in clus.doc_ids:
	        doc_clus_map[doc_id] = idx
	labels = [doc_clus_map[id] for id in sorted(doc_clus_map)]
	nclusters = len(set(labels))
	print("%.5f : %d" % (threshold, nclusters))
	if nclusters == 24:
		print("24 clusters for threshold = %.5f" % threshold)
		break

print(labels)
print(true_labels)
