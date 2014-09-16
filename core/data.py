import os
import json
from random import random, randint
from datetime import datetime

import numpy as np
from scipy.sparse import coo_matrix, hstack
from dateutil.parser import parse

from core.util import progress
from core.models import Article

def load_articles(test_file, with_labels=True):
    print('Loading articles from {0}...'.format(test_file))
    with open(test_file, 'r') as file:
        data = json.load(file)

    if with_labels:
        articles, labels_true = process_labeled_articles(data)
    else:
        articles = [process_article(a) for a in data]

    print('Loaded {0} articles.'.format(len(articles)))

    # Check if a vectorized file already exists.
    vecs_path = '/tmp/{0}.npy'.format(test_file.replace('/', '.'))
    if os.path.exists(vecs_path):
        print('Loading existing article vectors...')
        vecs = np.load(vecs_path)
    else:
        vecs = build_vectors(articles)
        np.save(vecs_path, vecs)

    if with_labels:
        print('Expecting {0} events.'.format(len(data)))
        return vecs, articles, labels_true

    return vecs, articles


def build_vectors(articles):
    bow_vecs, concept_vecs, pub_vecs, = [], [], []
    for a in progress(articles, 'Building article vectors...'):
        bow_vecs.append(a.vectors)
        concept_vecs.append(a.concept_vectors)
        pub_vecs.append(np.array([a.published]))
    bow_vecs = np.array(bow_vecs)
    concept_vecs = np.array(concept_vecs)
    pub_vecs = np.array(pub_vecs)

    # Merge the BoW features and the concept features as an ndarray.
    print('Merging vectors...')
    vectors = hstack([coo_matrix(pub_vecs), coo_matrix(bow_vecs), coo_matrix(concept_vecs)]).A
    print('Using {0} features.'.format(vectors.shape[1]))

    return vectors

def process_labeled_articles(data):
    # Build articles and true labels.
    articles, labels_true = [], []
    for idx, cluster in enumerate(data):
        members = []
        for a in cluster['articles']:
            article = process_article(a)
            members.append(article)
        articles += members
        labels_true += [idx for i in range(len(members))]
    return articles, labels_true

def process_article(a):
    a['id'] = hash(a['title'])

    # Handle MongoDB JSON dates.
    for key in ['created_at', 'updated_at']:
        date = a[key]['$date']
        if isinstance(date, int):
            a[key] = datetime.fromtimestamp(date/1000)
        else:
            a[key] = parse(a[key]['$date'])

    return Article(**a)


def split_list(objs, n_groups=3):
    """
    Takes a list of objs and splits them into randomly-sized groups.
    This is used to simulate how articles come in different groups.
    """
    shuffled = sorted(objs, key=lambda k: random())

    sets = []
    for i in range(n_groups):
        size = len(shuffled)
        end = randint(1, (size - (n_groups - i) + 1))

        yield shuffled[:end]

        shuffled = shuffled[end:]
