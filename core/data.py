import json
from random import random, randint
from dateutil.parser import parse

from core.models import Article

def load_articles(test_file):
    print('Loading articles from {0}...'.format(test_file))
    with open(test_file, 'r') as file:
        data = json.load(file)

    # Build articles and true labels.
    articles, labels_true = [], []
    for idx, cluster in enumerate(data):
        members = []
        for a in cluster['articles']:
            a['id'] = hash(a['title'])

            # Handle MongoDB JSON dates.
            for key in ['created_at', 'updated_at']:
                a[key] = parse(a[key]['$date'])

            article = Article(**a)
            members.append(article)
        articles += members
        labels_true += [idx for i in range(len(members))]

    print('Loaded {0} articles.'.format(len(articles)))
    print('Expecting {0} events.'.format(len(data)))
    return articles, labels_true

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
