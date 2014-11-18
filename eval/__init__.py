import os
import pickle
from datetime import datetime
from itertools import permutations
from collections import namedtuple

import numpy as np
from sklearn import metrics
from sklearn.grid_search import ParameterGrid

from core.cluster import hac, ihac, digbc, digshc
from core.util import labels_to_lists
from eval.util import progress, file_logger, TableGenerator
from eval.report import build_report
from eval.data import load_articles, build_vectors
from eval.parallel import parallelize

METRICS = ['adjusted_rand', 'adjusted_mutual_info', 'completeness', 'homogeneity']

Member = namedtuple('Member', ['id', 'title'])

approaches = {
    'hac': hac,
    'ihac': ihac,
    'digbc': digbc,
    'digshc': digshc
}

def evaluate(datapath, approach='digshc', param_grid=None):
    logger = file_logger('eval')
    logger.info("\n\nEvaluating [{0}] on [{1}] with approach [{2}].".format(datapath, datetime.utcnow(), approach))

    articles, labels_true = load_articles(datapath)
    articles_ = [Member(a.id, a.title) for a in articles]

    if 'dig' in approach:
        # articles = [a.text for a in articles]
        vecs_path = None
    else:
        # Build the vectors if they do not exist.
        vecs_path = '/tmp/{0}.pickle'.format(datapath.replace('/', '.'))
        if not os.path.exists(vecs_path):
            build_vectors(articles, vecs_path)

    if param_grid is None:
        # Param grid for development, just one param combo so things run quickly.
        param_grid = ParameterGrid({
            'metric': ['cosine'],
            'linkage_method': ['average'],
            'threshold': [0.8],
            'weights': [[1,1,1]]
        })

        # pass
        if approach == 'digshc':
            param_grid = ParameterGrid({
                'alpha': np.arange(0.65, 0.81, 0.05),
                'threshold': [0.2],
                'epsilon': np.arange(0.005, 0.011, 0.001),
                'hr_min': np.arange(0.3, 0.51, 0.05)
            })


    # Not working right now, need more memory. scipy's pdist stores an array in memory
    # which craps out parallelization cause there's not enough memory to go around.
    #print('Running {0} parameter combos...'.format(len(param_grid)))
    #args_set = [(vectors, pg_set) for pg_set in np.array_split(list(param_grid), mp.cpu_count())]
    #results = parallelize(cluster_p, args_set)

    import time
    start_time = time.time()

    # For nicely formatted results in the log.
    tg = TableGenerator(list(list(param_grid)[0].keys()) + METRICS)
    logger.info(tg.build_headers())

    results = []
    for pg in progress(param_grid, 'Running {0} parameter combos...'.format(len(param_grid))):
        result = cluster(vecs_path, pg, approach, articles)

        # Score the result.
        result['score'] = score(labels_true, result['labels'])
        result['clusters'] = labels_to_lists(articles_, result['labels'])

        logger.info(tg.build_row(dict(list(result['params'].items()) + list(result['score'].items()))))

        results.append(result)

    elapsed_time = time.time() - start_time
    print('Clustered in {0}'.format(elapsed_time))

    avgs = average_results(results)
    bests, lines = calculate_bests(results)
    print('Average scores: {0}'.format(avgs))
    lines += '\n\n{0}'.format(avgs)

    now = datetime.now()
    dataname = datapath.split('/')[-1].split('.')[0]
    filename = '{0}_{1}_{2}'.format(approach, dataname, now.isoformat())

    # Simple text report.
    build_report(filename, '\n'.join(lines))

    # HTML report, more detailed.
    report_path = build_report(filename, {
        'metrics': METRICS,
        'clusterables': articles,
        'results': results,
        'bests': bests,
        'expected': labels_to_lists(articles, labels_true),
        'dataset': datapath,
        'date': now
    }, template='eval_report.html')

    return bests


def calculate_bests(results):
    bests = {}
    lines = []
    for metric in METRICS:
        srtd = sorted(results, key=lambda x:x['score'][metric], reverse=True)

        lines.append('======\n{0}\n======'.format(metric))
        lines += ['{0}, scored {1} [{2}]'.format(res['params'], res['score'][metric], metric) for res in srtd[:20]]
        lines.append('\n\n============================\n\n')

        bests[metric] = srtd[0]
        print('Best parameter combination: {0}, scored {1} [{2}]'.format(bests[metric]['params'], bests[metric]['score'][metric], metric))

    return bests, lines


def cluster_p(vectors, pg_set):
    return [cluster(vectors, pg) for pg in pg_set]


def cluster(filepath, pg, approach, articles):

    # Handled specially
    if 'dig' in approach:
        labels_pred = approaches[approach]([a.text for a in articles], **pg)

    else:
        pg_ = pg.copy()

        # Reload the original vectors, so when we weigh them we can just
        # modify these vectors without copying them (to save memory).
        with open(filepath, 'rb') as f:
            vecs = pickle.load(f)

        vecs = weight_vectors(vecs, weights=pg_['weights'])

        pg_.pop('weights', None)

        try:
            labels_pred = approaches[approach](vecs, **pg_)
        except KeyError:
            print('Unrecognized approach "{0}"'.format(approach))

        if hasattr(pg['metric'], '__call__'): pg['metric'] = pg['metric'].__name__

    return {
        'params': pg,
        'labels': labels_pred,
        'id': hash(str(pg))
    }


def average_results(results):
    avgs = {metric: [] for metric in METRICS}
    for result in results:
        for metric, scr in result['score'].items():
            avgs[metric].append(scr)

    for metric, scrs in avgs.items():
        avgs[metric] = sum(scrs)/len(scrs)

    return avgs

def weight_vectors(v, weights):
    # Convert to a scipy.sparse.lil_matrix because it is subscriptable.
    v = v.tolil()

    # Apply weights to the proper columns:
    # col 0 = pub, cols 1-101 = bow, 102+ = concepts
    # weights = [pub, bow, concept]
    v[:,0] *= weights[0]
    v[:,1:101] *= weights[1]
    v[:,101:] *= weights[2]
    return v


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


def test(datapath):
    """
    Test the clustering on a dataset that doesn't have labels.

    TO DO: this needs to be updated.
    """
    articles = load_articles(datapath, with_labels=False)
    vectors = build_vectors(articles[:20], datapath)

    import time
    start_time = time.time()
    print('Clustering...')
    labels = hac(vectors, 'cosine', 'average', 0.8)
    elapsed_time = time.time() - start_time
    print('Clustered in {0}'.format(elapsed_time))

    clusters = labels_to_lists(articles, labels)
    import ipdb; ipdb.set_trace()
    now = datetime.now()
    dataname = datapath.split('/')[-1].split('.')[0]
    filename = 'test_{0}_{1}'.format(dataname, now.isoformat())
    report_path = build_report(filename, {
        'clusterables': articles,
        'clusters': clusters,
        'dataset': datapath,
        'date': now
    }, template='test_report.html')
