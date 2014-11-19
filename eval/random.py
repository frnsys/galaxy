import random
from scipy.sparse import vstack

def shuffle(vecs, labels_true):
    # Pair up the vectors with their labels.
    labeled_vecs = list(zip(list(vecs), labels_true))

    # Shuffle them.
    random.shuffle(labeled_vecs)

    # Separate the lists again.
    vecs, labels_true = zip(*labeled_vecs)

    return vstack(vecs), list(labels_true)


def chunk(vecs, n_chunks=3):
    # Break it up into randomly-sized chunks!
    chunks = []
    for i in range(n_chunks):
        size = len(vecs)
        if i == n_chunks - 1:
            v = vstack(vecs)

        else:
            end = random.randint(1, size - n_chunks - i + 2)
            v = vstack(vecs[:end])
            vecs = vecs[end:]

        chunks.append(v)
    return chunks
