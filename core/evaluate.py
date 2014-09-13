import importlib
import inspect
import webbrowser
from datetime import datetime

import numpy as np
from scipy.sparse import coo_matrix, hstack
from sklearn import metrics
from sklearn.grid_search import ParameterGrid

from core.cluster import hac
from core.util import progress
from core.report import build_report
from core.data import load_articles

METRICS = ['adjusted_rand', 'adjusted_mutual_info', 'completeness', 'homogeneity']

def evaluate(datapath):
    articles, labels_true = load_articles(datapath)

    bow_vecs, concept_vecs = [], []
    for a in progress(articles, 'Building article vectors...'):
        bow_vecs.append(a.vectors)
        concept_vecs.append(a.concept_vectors)
    bow_vecs = np.array(bow_vecs)
    concept_vecs = np.array(concept_vecs)

    # Merge the BoW features and the concept features as an ndarray.
    vectors = hstack([coo_matrix(bow_vecs), coo_matrix(concept_vecs)]).A
    #vectors = concept_vecs
    print('Using {0} features.'.format(vectors.shape[1]))

    param_grid = ParameterGrid({
        'metric': ['euclidean', 'cosine', 'jaccard'],
        'linkage_method': ['average', 'weighted'],
        'threshold': np.arange(0.1, 1.0, 0.05)
    })

    results = []

    for pg in progress(param_grid, 'Running {0} parameter combos...'.format(len(param_grid))):
        labels_pred = hac(vectors, **pg)
        scr = score(labels_true, labels_pred)
        clusters = labels_to_lists(articles, labels_pred)

        if hasattr(pg['metric'], '__call__'): pg['metric'] = pg['metric'].__name__
        results.append({
            'params': pg,
            'score': scr,
            'clusters': clusters,
            'labels': labels_pred,
            'id': hash(str(pg))
        })


    bests = {}
    for metric in METRICS:
        bests[metric] = max(results, key=lambda x:x['score'][metric])
        print('Best parameter combination: {0}, scored {1} [{2}]'.format(bests[metric]['params'], bests[metric]['score'][metric], metric))

    now = datetime.now()
    dataname = datapath.split('/')[-1].split('.')[0]
    filename = '{0}_{1}'.format(dataname, now.isoformat())
    report_path = build_report(filename, {
        'metrics': METRICS,
        'clusterables': articles,
        'results': results,
        'bests': bests,
        'expected': labels_to_lists(articles, labels_true),
        'dataset': datapath,
        'date': now
    })
    webbrowser.open('file://{0}'.format(report_path), new=2)



def labels_to_lists(objs, labels):
    tmp = {}

    for i, label in enumerate(labels):
        if label not in tmp:
            tmp[label] = []
        tmp[label].append(objs[i])

    return [v for v in tmp.values()]


def score(labels_true, labels_pred):
    """
    Score clustering results.

    These labels to NOT need to be congruent,
    these scoring functions only consider the cluster composition.

    That is::

        labels_true = [0,0,0,1,1,1]
        labels_pred = [5,5,5,2,2,2]
        score(labels_pred)
        >>> 1.0

    Even though the labels aren't exactly the same,
    all that matters is that the items which belong together
    have been clustered together.
    """
    return {metric: metrics.__dict__['{0}_score'.format(metric)](labels_true, labels_pred) for metric in METRICS}
