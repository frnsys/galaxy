from eval.data import load_articles
from core.digbc.digbc import DocumentIndexGraphClusterer

dig = DocumentIndexGraphClusterer(threshold=0.003)

docs, true_labels = load_articles('eval/data/event/handpicked.json')

for idx, doc in enumerate(docs):
    dig.index_document(doc.text)

doc_clus_map = {}
for idx, clus in enumerate(dig.formed_clusters):
    for doc_id in clus.doc_ids:
        doc_clus_map[doc_id] = idx
labels = [doc_clus_map[id] for id in sorted(doc_clus_map)]

print(labels)
print(true_labels)
