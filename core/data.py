import json
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
    return articles, labels_true
