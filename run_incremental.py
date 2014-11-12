import os
import random
import pickle

import numpy as np
from scipy.sparse import vstack, lil_matrix
from sklearn.grid_search import ParameterGrid

from eval import weight_vectors, score
from eval.data import load_articles, build_vectors

from core.ihac import IHAC
from core.ihac.node import Node


def evaluate(datapath):
    articles, labels_true = load_articles(datapath)

    # Build the vectors if they do not exist.
    vecs_path = '/tmp/{0}.pickle'.format(datapath.replace('/', '.'))
    if not os.path.exists(vecs_path):
        build_vectors(articles, vecs_path)

    # Use idealized params (for IHAC w/ handpicked.json).
    param_grid = ParameterGrid({
        'metric': ['cosine'],
        'linkage_method': ['average'],
        'threshold': [60.],
        'weights': [[21., 81., 41.]],
        'lower_limit_scale': [0.8],
        'upper_limit_scale': [1.15]
    })
    pgs = [pg for pg in param_grid]

    result = cluster(vecs_path, pgs[0], labels_true)
    print(result)


def cluster(filepath, pg, labels_true):
    Node.lower_limit_scale = pg['lower_limit_scale']
    Node.upper_limit_scale = pg['upper_limit_scale']
    model = IHAC()

    # Reload the original vectors, so when we weigh them we can just
    # modify these vectors without copying them (to save memory).
    with open(filepath, 'rb') as f:
        vecs = pickle.load(f)

    vecs = weight_vectors(vecs, weights=pg['weights'])

    vecs, labels_true = shuffle(vecs, labels_true)
    model.fit(vecs.toarray())

    clusters, labels_pred = model.clusters(distance_threshold=pg['threshold'], with_labels=True)

    return score(labels_true, labels_pred)


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
            end = random.randint(1, size - n_chunks - i + 1)
            v = vstack(vecs[:end])
            vecs = vecs[end:]

        chunks.append(v)
    return chunks

if __name__ == '__main__':
    evaluate('eval/data/event/handpicked.json')
