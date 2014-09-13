from core.vectorize import vectorize
from core import concepts

class Article():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def concepts(self):
        if not hasattr(self, '_concepts'):
            text = ' '.join([self.title, self.text])
            self._concepts = concepts.concepts(text, strategy='stanford')
        return self._concepts

    @property
    def concept_vectors(self):
        if not hasattr(self, '_concept_vectors'):
            concept_doc = ' '.join(self.concepts)
            self._concept_vectors = concepts.vectorize(concept_doc)
        return self._concept_vectors

    @property
    def vectors(self):
        if not hasattr(self, '_vectors'):
            self._vectors = vectorize(self.text)
        return self._vectors
