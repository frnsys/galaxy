import sys
import json

from core.vectorize import train
from core import concepts
from eval import evaluate, test

cmd = sys.argv[1]
datapath = sys.argv[2]

# Train the feature pipeline.
if cmd == 'train':
    training_file = open(datapath, 'r')
    training_data = json.load(training_file)

    docs = ['{0} {1}'.format(d['title'], d['text']) for d in training_data]
    train(docs)
    concepts.train(docs)


# Test the clustering on a dataset that has labels.
elif cmd == 'evaluate':
    evaluate(datapath)


# Test the clustering on a dataset that doesn't have labels.
elif cmd == 'test':
    test(datapath)

# this was used to compare concept feature reduction
#elif cmd == 'compare':

    #from core.models import Article
    #concepts.vectorize = concepts.vectorize_new

    #evaluate(datapath)

elif cmd == 'train_concepts':
    pass
    # extract and save concepts once
    #import json
    #from core.concepts import concepts
    #training_file = open(datapath, 'r')
    #training_data = json.load(training_file)
    #docs = ['{0} {1}'.format(d['title'], d['text']) for d in training_data]
    #with open('concepts.json', 'w') as out:
        #json.dump([concepts(doc) for doc in docs], out)
    #n_components = int(sys.argv[2])
    #concepts.train([], n_components)
