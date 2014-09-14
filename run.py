import sys
import json

from core.vectorize import train
from core.evaluate import evaluate, test

cmd = sys.argv[1]
datapath = sys.argv[2]

if cmd == 'train':
    training_file = open(datapath, 'r')
    training_data = json.load(training_file)

    docs = ['{0} {1}'.format(d['title'], d['text']) for d in training_data]
    train(docs)

elif cmd == 'evaluate':
    evaluate(datapath)

elif cmd == 'test':
    test(datapath)
