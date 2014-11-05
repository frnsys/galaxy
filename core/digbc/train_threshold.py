from eval.data import load_articles
from core.digbc.dig import DocumentIndexGraphClusterer
import numpy as np



docs, true_labels = load_articles('../../eval/data/event/handpicked.json')

for threshold in np.arange(0.0001, 0.001, 0.00002):
	dig = DocumentIndexGraphClusterer(threshold=0.001)

	for idx, doc in enumerate(docs):
	    dig.index_document(doc.text)

	doc_clus_map = {}
	for idx, clus in enumerate(dig.formed_clusters):
	    for doc_id in clus.doc_ids:
	        doc_clus_map[doc_id] = idx
	labels = [doc_clus_map[id] for id in sorted(doc_clus_map)]
	print("%.4f : %d" % (threshold, len(labels)))
	if len(labels) == 23:
		import ipdb; ipdb.set_trace()

print(labels)
print(true_labels)
