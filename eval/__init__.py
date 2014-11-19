import os
import time
import pickle
from datetime import datetime
from collections import namedtuple

from core.cluster import hac, ihac, digbc, digshc
from core.util import labels_to_lists
from eval.report import build_report
from eval.data import load_articles, build_vectors
from eval.parallel import parallelize
from eval import random, scoring

Member = namedtuple('Member', ['id', 'title'])

# Map approach names to their functions.
approaches = {
    'hac': hac,
    'ihac': ihac,
    'digbc': digbc,
    'digshc': digshc
}

def evaluate(datapath, param_grid, approach='hac', incremental=False):
    print("\n\nEvaluating [{0}] on [{1}] with approach [{2}].".format(datapath, datetime.utcnow(), approach))

    articles, labels_true = load_articles(datapath)
    articles_ = [Member(a.id, a.title) for a in articles]

    # Build the vectors if they do not exist.
    vecs_path = '/tmp/{0}.pickle'.format(datapath.replace('/', '.'))
    if not os.path.exists(vecs_path):
        build_vectors(articles, vecs_path)

    start_time = time.time()

    if incremental:
        with open(vecs_path, 'rb') as f:
            vecs = pickle.load(f)

        vecs, labels_true = random.shuffle(vecs.tolil(), labels_true)
        results = [cluster(vecs, param_grid, approach, articles, labels_true)]

    else:
        print('Running {0} parameter combos...'.format(len(param_grid)))
        args_set = [(vecs_path, pg, approach, articles, labels_true) for pg in param_grid]
        results = parallelize(cluster_p, args_set)

    elapsed_time = time.time() - start_time
    print('Clustered in {0}'.format(elapsed_time))

    avgs = scoring.average_results(results)
    bests, lines = scoring.calculate_bests(results)
    print('Average scores: {0}'.format(avgs))
    lines += ['\n\nAverage scores:\n{0}\n\n'.format(avgs)]

    for result in results:
        result['clusters'] = labels_to_lists(articles_, result['labels'])

    now = datetime.now()
    dataname = datapath.split('/')[-1].split('.')[0]
    filename = '{0}_{1}_{2}'.format(approach, dataname, now.isoformat())

    # Simple text report.
    build_report(filename, '\n'.join(lines))

    # HTML report, more detailed.
    report_path = build_report(filename, {
        'metrics': scoring.METRICS,
        'clusterables': articles,
        'results': results,
        'bests': bests,
        'expected': labels_to_lists(articles, labels_true),
        'dataset': datapath,
        'date': now
    }, template='eval_report.html')

    return report_path


def cluster_p(vecs_path, params, approach, articles, labels_true):
    """
    A wrapper for running different param grids in a parallel context.
    """
    with open(vecs_path, 'rb') as f:
        vecs = pickle.load(f)
    return cluster(vecs, params, approach, articles, labels_true)


def cluster(vecs, params, approach, articles, labels_true):
    result = {
        'params': params,
        'id': hash(str(params))
    }

    # Handled specially
    if 'dig' in approach:
        labels_pred = approaches[approach]([a.text for a in articles], **params)

    else:
        try:
            labels_pred = approaches[approach](vecs, **params)
        except KeyError:
            print('Unrecognized approach "{0}"'.format(approach))

    result['labels'] = labels_pred

    # Score the result.
    result['score'] = scoring.score(labels_true, result['labels'])
    return result


def test(datapath, approach, params):
    """
    Test the clustering on a dataset that doesn't have labels.
    """
    articles = load_articles(datapath, with_labels=False)
    vecs = build_vectors(articles)

    start_time = time.time()
    print('Clustering...')

    labels = approaches[approach](vecs, **params)

    elapsed_time = time.time() - start_time
    print('Clustered in {0}'.format(elapsed_time))

    clusters = labels_to_lists(articles, labels)

    # Build a report.
    now = datetime.now()
    dataname = datapath.split('/')[-1].split('.')[0]
    filename = 'test_{0}_{1}'.format(dataname, now.isoformat())
    report_path = build_report(filename, {
        'clusterables': articles,
        'clusters': clusters,
        'dataset': datapath,
        'date': now
    }, template='test_report.html')
    return report_path


