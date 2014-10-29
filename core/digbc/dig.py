import networkx as nx

def sentence_split(plain_text):
    pass


class DocumentIndexGraph():
"""
    Document Index Graph structure
    as defined on the paper

    "Efficient Phrase-Based Document Indexing for Web Document Clustering"
"""
    def __init__(self):
        self.document_ids = []
        self.term_dict = {} # a dictionary mapping terms to graph node ids
        self.graph = nx.Graph()


    def index_document(self, plain_text):
        # split in sentences
        sentences = sentence_split(plain_text)
        # tokenize sentences
        sentences = [tokenize(s) for s in sentences]

        for sent in sentences:
            prev = sent[0]
            for term in sent[1:]:
                # add term to graph if not present

    def add_term_node(self, term):
        pass




class TermNode():
    """
        Represents a word node in the DocumentIndexGraph
        It is created the first time the word is seen
        on a document
    """
    def __init__(self, term, doc_id):
        self.term = term
        self.document_table = {doc_id: }


class DocumentTableEntry(object):
    """docstring for  """
    def __init__(self, doc_id):
        self.doc_id = doc_id
        self.term_frequencies = {'low': 0, 'med': 0, 'high': 0}
        self.edge_table = {}

    def add_edge(self, edge_id, sentence_n, pos):
        """
            Records the fact that given edge
            occurs on sentence_n of the represented
            document on position pos
        """
        self.edge_table.setdefault(edge_id, {})
        self.edge_table[edge_id][sentence_n] = pos



class Document():
    """
    Phrase-based document structure
    as defined on the paper

    "Efficient Phrase-Based Document Indexing for Web Document Clustering"
    """
    def __init__(self, plain_text):

