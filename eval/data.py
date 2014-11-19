import json
import pickle
from random import random, randint
from datetime import datetime

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import normalize
from dateutil.parser import parse

from core.models import Article
from eval.util import progress


def load_articles(datapath, with_labels=True, as_incremental=False):
    print('Loading articles from {0}...'.format(datapath))
    with open(datapath, 'r') as file:
        data = json.load(file)

    if with_labels:
        articles, labels_true = process_labeled_articles(data)
    else:
        articles = [process_article(a) for a in data]

    print('Loaded {0} articles.'.format(len(articles)))

    if as_incremental:
        articles = split_list(articles)

    if with_labels:
        print('Expecting {0} events.'.format(len(data)))
        return articles, labels_true

    return articles


def build_vectors(articles, savepath=None):
    bow_vecs, concept_vecs, pub_vecs, = [], [], []

    for a in progress(articles, 'Building article vectors...'):
        pub_vecs.append(np.array([a.published]))
        bow_vecs.append(a.vectors)
        concept_vecs.append(a.concept_vectors)

    pub_vecs = normalize(csr_matrix(pub_vecs), copy=False)
    bow_vecs = normalize(csr_matrix(bow_vecs), copy=False)
    concept_vecs = normalize(csr_matrix(concept_vecs), copy=False)

    print('Merging vectors...')
    vecs = hstack([pub_vecs, bow_vecs, concept_vecs])
    print('Using {0} features.'.format(vecs.shape[1]))

    if savepath:
        with open(savepath, 'wb') as f:
            pickle.dump(vecs, f)

    return vecs


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