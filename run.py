import sys
import json
from itertools import permutations

#import logging
#logging.basicConfig(level=logging.DEBUG, format='%(name)s ~ %(message)s')


from core.vectorize import train
from core import concepts
from eval import evaluate, test

import numpy as np
from sklearn.grid_search import ParameterGrid

cmd = sys.argv[1]
datapath = sys.argv[2]


approaches = {
    #'hac': ParameterGrid({
        #'metric': ['cosine'],
        #'linkage_method': ['average'],
        #'threshold': np.arange(0.1, 0.25, 0.05),
        #'weights': list( permutations(np.arange(1., 82., 20.), 3) )
    #}),
    #'ihac': ParameterGrid({
        #'metric': ['cosine'],
        #'threshold': np.arange(40., 100., 10.),
        #'weights': list( permutations(np.arange(21., 102., 20.), 3) ),
        #'lower_limit_scale': np.arange(0.1, 1.1, 0.1),
        #'upper_limit_scale': np.arange(1.1, 2.0, 0.05)
    #}),
    'ihac': ParameterGrid({
        'metric': ['euclidean'],
        'threshold': np.arange(40., 100., 10.),
        'weights': [(21., 81., 41.)],
        'lower_limit_scale': [0.8],
        'upper_limit_scale': [1.15]
    }),
    #'digbc': ParameterGrid({
        #'threshold': np.arange(0.00295, 0.0100, 0.00005)
    #})
}


# Train the feature pipeline.
if cmd == 'train':
    training_file = open(datapath, 'r')
    training_data = json.load(training_file)

    docs = ['{0} {1}'.format(d['title'], d['text']) for d in training_data]
    train(docs)
    concepts.train(docs)


# Test the clustering on a dataset that has labels.
elif cmd == 'evaluate':
    try:
        approach = sys.argv[3]
    except IndexError:
        approach = 'hac'

    evaluate(datapath, approach=approach, param_grid=approaches[approach])


# Test the clustering on a dataset that doesn't have labels.
elif cmd == 'test':
    test(datapath)


# Compare different clustering algorithms on different param grids.
elif cmd == 'compare':
    results = {}
    for approach, param_grid in approaches.items():
        print('Running the `{0}` algo...'.format(approach))
        results[approach] = evaluate(datapath, approach=approach, param_grid=param_grid)

    #print(results)
