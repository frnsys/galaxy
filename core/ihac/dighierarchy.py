from .hierarchy import Hierarchy
from core.digshc.dig import DocumentIndexGraph
import scipy
import numpy as np
import itertools

def vectorize(plain_text):
    # TODO: implement a vectorization that is compatible
    # with phrase-based similarity. Even though vectors
    # are not neede for inter-node distances, we still need
    # them to calculate cluster centers.

    # Actually, it is possible that centers are not
    # needed anymore in this version of the hierarchy,
    # but to implement a version without centers we
    # need to do a major refactor of the IHAC code
    raise NotImplementedError


class DIGHierarchy(Hierarchy):
    """
        Custom Hierarchy with phrase-based distance function
        calculated with a Document Index Graph structure
    """
    def __init__(self, vector):
        super(DIGHierarchy, self).__init__(vector)
        self.dig = DocumentIndexGraph()
        self.leaf_to_dig_id = {}

    def distance_function(self, a, b):
        a_docs = [leaf_to_dig_id(l.id) for l in a.leaves()]
        b_docs = [leaf_to_dig_id(l.id) for l in b.leaves()]

        pairs = list(itertools.product(a_docs, b_docs))
        dists = np.array([self.dig.get_distance(x, y) for (x, y) in pairs])

        return scipy.average(dists)

    def incorporate_doc(self, plain_text):
        doc = dig.index_document(plain_text)
        vec = vectorize(plain_text)
        leaf = self.incorporate(vec)
        self.leaf_to_dig_id[leaf.id] = doc.id
