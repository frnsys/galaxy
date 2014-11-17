from eval.data import load_articles
from core.digshc.shc import SimilarityHistogramClusterer

shc = SimilarityHistogramClusterer(threshold=0.3)

docs, true_labels = load_articles('eval/data/event/handpicked.json')

docs = docs[:20]
true_labels = true_labels[:20]

for idx, doc in enumerate(docs):
    shc.fit(doc.text)

doc_clus_map = {}
for idx, clus in enumerate(shc.formed_clusters):
    for doc_id in clus.doc_ids:
        doc_clus_map[doc_id] = idx
labels = [doc_clus_map[id] for id in sorted(doc_clus_map)]

print([cl.doc_ids for cl in shc.formed_clusters])
print(labels)
print(true_labels)
