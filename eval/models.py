import pytz
from datetime import datetime

from galaxy import vectorize, concept

epoch = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.UTC)

class Article():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def concepts(self):
        if not hasattr(self, '_concepts'):
            text = ' '.join([self.title, self.text])
            #self._concepts = concept.concepts(text, strategy='stanford') + concept.concepts(text, strategy='spotlight') + concept.keywords(text)
            self._concepts = concept.concepts(text, strategy='stanford')
        return self._concepts

    @property
    def concept_vectors(self):
        # Calculating these on the fly and not
        # caching the results because caching eats up a lot of memory.
        return concept.vectorize(self.concepts)

    @property
    def vectors(self):
        return vectorize(self.text)

    @property
    def published(self):
        """Convert datetime to seconds"""

        # If not timezone is set, assume UTC.
        # super annoying and it's probably not a good guess but it's
        # all we got for now.
        # In production, we will be setting article publish times as utc when
        # we fetch them, so it should be less of a problem there.
        if self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=pytz.UTC)
        delta = self.created_at - epoch
        return delta.total_seconds()
