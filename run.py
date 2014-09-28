import sys
import json

from core.vectorize import train
from eval import evaluate, test

cmd = sys.argv[1]
datapath = sys.argv[2]

# Train the feature pipeline.
if cmd == 'train':
    training_file = open(datapath, 'r')
    training_data = json.load(training_file)

    docs = ['{0} {1}'.format(d['title'], d['text']) for d in training_data]
    train(docs)


# Test the clustering on a dataset that has labels.
elif cmd == 'evaluate':
    evaluate(datapath)


# Test the clustering on a dataset that doesn't have labels.
elif cmd == 'test':
    test(datapath)
