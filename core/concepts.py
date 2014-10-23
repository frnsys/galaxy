"""
Conceptor
==============

Concept extraction from text.
"""

import os
import json
import string
import pickle
from urllib import request, error
from urllib.parse import urlencode

import ner
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

from conf import APP

class ConceptTokenizer():
    """
    Custom tokenizer for concept vectorization.
    """
    def __call__(self, doc):
        return tokenize(doc)

PIPELINE_PATH = os.path.expanduser(os.path.join(APP['PIPELINE_PATH'], 'concept_pipeline.pickle'))
if os.path.isfile(PIPELINE_PATH):
    PIPELINE = pickle.load(open(PIPELINE_PATH, 'rb'))
else:
    PIPELINE = False

def tokenize(doc):
    """
    Tokenize a document of ONLY concepts.
    It is expected that the concepts are delimited by '||'.

    e.g::

        'United States of America||China||Russia'
    """
    return doc.split('||')

def train(docs, n_components=100):
    """
    Trains and serializes (pickles) a vectorizing pipeline
    based on training data.

    `min_df` is set to filter out extremely rare words,
    since we don't want those to dominate the distance metric.

    `max_df` is set to filter out extremely common words,
    since they don't convey much information.
    """
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(input='content', stop_words='english', lowercase=True, tokenizer=ConceptTokenizer(), min_df=0.0, max_df=0.9)),
        ('tfidf', TfidfTransformer(norm=None, use_idf=True, smooth_idf=True)),
        ('feature_reducer', TruncatedSVD(n_components=n_components)),
        ('normalizer', Normalizer(copy=False))
    ])

    print('Training on {0} docs...'.format(len(docs)))
    pipeline.fit(['||'.join(concepts(doc)) for doc in docs])

    PIPELINE = pipeline

    print('Serializing pipeline to {0}'.format(PIPELINE_PATH))
    pipeline_file = open(PIPELINE_PATH, 'wb')
    pickle.dump(pipeline, pipeline_file)
    print('Training complete.')

def strip(text):
    """
    Removes punctuation from the beginning
    and end of text.
    """
    punctuation = string.punctuation + '“”‘’–"'
    return text.strip(punctuation)

def concepts(docs, strategy='stanford'):
    """
    Named entity recognition on
    a text document or documents.

    Requires that a Stanford NER server or a DBpedia Spotlight
    server is running at argos.conf.APP['KNOWLEDGE_HOST'],
    depending on which strategy you choose.

    Args:
        | docs (list)       -- the documents to process.
        | doc (str)         -- the document to process.
        | strategy (str)    -- the strategy to use, default is `stanford`. can be `stanford` or `spotlight`.

    Returns:
        | list              -- list of all entity mentions
    """
    if type(docs) is str:
        docs = [docs]

    entities = []

    if strategy == 'stanford':
        tagger = ner.SocketNER(host=APP['KNOWLEDGE_HOST'], port=8080)

        for doc in docs:
            try:
                ents = tagger.get_entities(doc)
            except UnicodeDecodeError as e:
                print.exception('Unexpected unicode decoding error: {0}'.format(e))
                ents = {}

            # We're only interested in the entity names,
            # not their tags.
            names = [ents[key] for key in ents]

            # Flatten the list of lists.
            names = [strip(name) for sublist in names for name in sublist]

            entities += names

    elif strategy == 'spotlight':
        '''
        Example response from DBpedia Spotlight:
        {
          "@text": "Brazilian state-run giant oil company Petrobras signed a three-year technology and research cooperation agreement with oil service provider Halliburton.",
          "@confidence": "0.0",
          "@support": "0",
          "@types": "",
          "@sparql": "",
          "@policy": "whitelist",
          "Resources":   [
                {
              "@URI": "http://dbpedia.org/resource/Brazil",
              "@support": "74040",
              "@types": "Schema:Place,DBpedia:Place,DBpedia:PopulatedPlace,Schema:Country,DBpedia:Country",
              "@surfaceForm": "Brazilian",
              "@offset": "0",
              "@similarityScore": "0.9999203720889515",
              "@percentageOfSecondRank": "7.564391175472872E-5"
            },
                {
              "@URI": "http://dbpedia.org/resource/Petrobras",
              "@support": "387",
              "@types": "DBpedia:Agent,Schema:Organization,DBpedia:Organisation,DBpedia:Company",
              "@surfaceForm": "Petrobras",
              "@offset": "38",
              "@similarityScore": "1.0",
              "@percentageOfSecondRank": "0.0"
            },
                {
              "@URI": "http://dbpedia.org/resource/Halliburton",
              "@support": "458",
              "@types": "DBpedia:Agent,Schema:Organization,DBpedia:Organisation,DBpedia:Company",
              "@surfaceForm": "Halliburton",
              "@offset": "140",
              "@similarityScore": "1.0",
              "@percentageOfSecondRank": "0.0"
            }
          ]
        }

        As you can see, it provides more (useful) data than we are taking advantage of.
        '''
        endpoint = 'http://{host}:2222/rest/annotate'.format(host=APP['KNOWLEDGE_HOST'])
        for doc in docs:
            data = {
                    'text': doc,
                    'confidence': 0,
                    'support': 0
                   }
            url = '{endpoint}?{data}'.format(endpoint=endpoint, data=urlencode(data))
            req = request.Request(url,
                    headers={
                        'Accept': 'application/json'
                    })
            try:
                res = request.urlopen(req)
            except error.HTTPError as e:
                print.exception('Error extracting entities (strategy=spotlight) with doc: {0}\n\nError: {1}'.format(doc, e.read()))
                raise e
            if res.status != 200:
                raise Exception('Response error, status was not 200')
            else:
                content = res.read()
                entities = json.loads(content.decode('utf-8'))['Resources']
                return [entity['@surfaceForm'] for entity in entities]

    else:
        raise Exception('Unknown strategy specified. Please use either `stanford` or `spotlight`.')

    return entities


def vectorize_old(concepts):
    """
    This vectorizes a list or a string of concepts;
    the regular `vectorize` method is meant to vectorize text documents;
    it is trained for that kind of data and thus is inappropriate for concepts.
    So instead we just use a simple hashing vectorizer.
    """
    h = HashingVectorizer(input='content', stop_words='english', norm=None, tokenizer=ConceptTokenizer())
    if type(concepts) is str:
        # Extract and return the vector for the single document.
        return h.transform([concepts]).toarray()[0]
    else:
        return h.transform(concepts)


def vectorize(concepts):
    """
    Vectorizes a list of concepts using
    a trained vectorizing pipeline.
    """
    if not PIPELINE:
        raise Exception('No pipeline is loaded. Have you trained one yet?')

    if type(concepts) is str:
        # Extract and return the vector for the single document.
        return PIPELINE.transform([concepts])[0]
    else:
        return PIPELINE.transform(concepts)
