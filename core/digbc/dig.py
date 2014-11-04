import networkx as nx
import numpy as np
from copy import copy

# Level Definitions
HIGH = 0
MED = 1
LOW = 2

WEIGHTS = {
    HIGH: 2.0,
    MED: 1.5,
    LOW: 1.0
}

def sentencize(plain_text):
    sentences = plain_text.split(".")
    sentences = [tokenize(s) for s in sentences]
    return [RankedSentence(sentence=s, level=LOW) for s in sentences]

def tokenize(sentence):
    terms = sentence.strip().split(" ")
    return terms


class Document(object):
    """docstring for Document"""
    def __init__(self, id, plain_text):
        # parse ranked tokenized sentences
        self.id = id
        self.sentences = sentencize(plain_text)
        self.wlength = None # calculate only if needed

    def get_weight(self, sent_n):
        return WEIGHTS[self.sentences[sent_n].level]

    def get_length(self, sent_n):
        return len(self.sentences[sent_n].sentence)

    def weighted_length(self):
        if not self.wlength:
            wlengths = [self.get_weight(i) * self.get_length(i) for i in range(len(self.sentences))]
            self.wlength = sum(wlengths)

        return self.wlength



class DocumentIndexGraph(nx.DiGraph):
    """
    Document Index Graph structure
    as defined on the paper

    "Efficient Phrase-Based Document Indexing for Web Document Clustering"
    """
    def __init__(self):
        super(DocumentIndexGraph, self).__init__()
        self.document_tables = {}
        self.indexed_docs = []
        self.matching_phrases = {}

    def index_document(self, plain_text):
        doc_id = len(self.indexed_docs)
        document = Document(doc_id, plain_text)
        self.indexed_docs.append(document)

        for n, rsent in enumerate(document.sentences):
            sent, level = rsent.sentence, rsent.level
            previous_term = sent[0]
            for p, current_term in enumerate(sent[1:]):
                position = (n, p)
                self.add_edge(doc_id, position, previous_term, current_term, level)
                previous_term = current_term
            # update term freq for last term (add edge only updates for startig term of
            #    given pair)
            doc_table = self.get_doc_table(sent[-1])
            doc_table.setdefault(doc_id, DocumentTableEntry())
            doc_table[doc_id].term_freqs[level] += 1

    def get_doc(self, doc_id):
        return self.indexed_docs[doc_id]

    def add_edge(self, doc_id, position, term1, term2, level):
        # position is a tuple (sent_n, term_n) indicating
        # the sentence number and term number where the occurence was found
        edge = (term1, term2)
        super(DocumentIndexGraph, self).add_edge(*edge)
        doc_table = self.get_doc_table(term1)

        # retrieve list of docs sharing this edge
        # and update phrase matches
        matching_doc_ids = [d_id for d_id in doc_table.keys() if edge in doc_table[d_id].edge_table]
        for d_id in matching_doc_ids:
            d_edge_table = doc_table[d_id].edge_table
            # occurences of current edge in doc d
            d_edge_positions = copy(d_edge_table[edge])

            # search for previous matches ending on term1 and extend them
            for pmatch in self.get_matching_phrases(d_id, doc_id):
                if pmatch.phrase[-1] == term1 and pmatch.end_position(doc_id) == position:
                    # remove this edge position from list
                    d_pos = pmatch.end_position(d_id)
                    d_edge_positions.remove(d_pos)
                    # extend mathing phrase
                    pmatch.phrase.append(term2)

            # all the remaining edge positions belong to new matching phrases
            for d_pos in d_edge_positions:
                doc = self.indexed_docs[doc_id]
                d = self.indexed_docs[d_id]
                new_pmatch = PhraseMatch(term1, term2, doc, d, position, d_pos)
                self.get_matching_phrases(d_id, doc_id).append(new_pmatch)

        # update document table for term1 with current doc
        doc_table.setdefault(doc_id, DocumentTableEntry())
        edge_table = doc_table[doc_id].edge_table

        edge_table.setdefault(edge, [])
        edge_table[edge].append(position)

        doc_table[doc_id].term_freqs[level] += 1

    def get_doc_table(self, term):
        self.document_tables.setdefault(term, {})
        return self.document_tables[term]

    def get_matching_phrases(self, doc_a_id, doc_b_id):
        ordered_ids = tuple(sorted((doc_a_id, doc_b_id)))
        self.matching_phrases.setdefault(ordered_ids, [])

        return self.matching_phrases[ordered_ids]

    def get_phrase_freq(self, doc_id, phrase):
        # TODO: implement this
        return 1.0

    def get_p_similarity(self, doc_a_id, doc_b_id):
        doc_a = self.get_doc(doc_a_id)
        doc_b = self.get_doc(doc_b_id)

        numerator = 0.0
        pmatches = self.get_matching_phrases(doc_a.id, doc_b.id)
        for pmatch in pmatches:
            phrase = pmatch.phrase
            f_a = self.get_phrase_freq(doc_a.id, phrase) 
            f_b = self.get_phrase_freq(doc_b.id, phrase)
            sent_n_a = pmatch.positions[doc_a_id][0]
            sent_n_b = pmatch.positions[doc_b_id][0]
            w_a = doc_a.get_weight(sent_n_a)
            w_b = doc_b.get_weight(sent_n_b)
            numerator += (pmatch.g() * (f_a * w_a + f_b * w_b)) ** 2

        return (numerator ** 0.5) / (doc_a.weighted_length() + doc_b.weighted_length())





class DocumentTableEntry(object):
    """docstring for DocumentTableEntry"""
    def __init__(self):
        self.term_freqs = [0, 0, 0]
        self.edge_table = {}



class RankedSentence(object):
    """docstring for RankedSentence"""
    def __init__(self, sentence, level):
        self.sentence = sentence
        self.level = level


class PhraseMatch(object):
    """docstring for PhraseMatch"""
    def __init__(self, term1, term2, doc_a, doc_b, pos_a, pos_b):
        """
            doc_a_id, doc_b_id: ids of documents where the matching occurs
            pos_a, pos_b: tuples of the form (sent_n, term_n) that contain
                the respective starting positions of matching phrase within
                doc_a and doc_b
            term1, term2: initial matching terms
        """
        self.phrase = [term1, term2]
        self.doc_a = doc_a
        self.doc_b = doc_b
        self.positions = {
            doc_a.id: pos_a,
            doc_b.id: pos_b
        }

    def extend(self, term):
        self.phrase.append(term)

    def end_position(doc_id):
        return self.positions[doc_id] + (0, len(self.phrase) - 1)

    def g(self):
        gamma = 1.2
        l_i = len(self.phrase)
        doc_a, doc_b = self.doc_a, self.doc_b
        s_a = len(doc_a.sentences[self.positions[doc_a.id][0]].sentence)
        s_b = len(doc_b.sentences[self.positions[doc_b.id][0]].sentence)
        return (2.0 * l_i / (s_a + s_b)) ** gamma




if __name__ == '__main__':
    docs = ["river rafting. mild river rafting. river rafting trips",
            "wild river adventures. river rafting vacation plan",
            "fishin trips. fishing vacation plan. booking fishing trips. river fishing"]

    dig = DocumentIndexGraph()
    for doc in docs:
        dig.index_document(doc)

    print([dig.get_p_similarity(a, b) for (a, b) in [(0, 1), (1, 2), (0, 2)]])

    # TODO: turn the following sessions into unit tests
    # (this is the example from the figure 3 in the paper,
    #  we use 1-based indices for docs, sentences, etc. to make
    #  make it easier to match results with the paper)

    # test_document_index_graph_structure():

    # ipdb> dig.nodes()
    # ['river', 'vacation', 'booking', 'rafting', 'fishing', 'trips', 'plan', 'fishin', 'adventures', 'wild', 'mild']
    # ipdb> dig.edges()
    # [('river', 'adventures'), ('river', 'fishing'), ('river', 'rafting'), ('river', 'wild'), ('river', 'mild'), ('vacation', 'rafting'), ('vacation', 'fishing'), ('vacation', 'plan'), ('booking', 'fishing'), ('rafting', 'trips'), ('fishing', 'trips'), ('trips', 'fishin')]
    # ipdb> dig.document_tables["river"]
    # {1: {('river', 'rafting'): {1: 1, 2: 2, 3: 1}}, 2: {('river', 'rafting'): {2: 1}, ('river', 'adventures'): {1: 2}}, 3: {('river', 'fishing'): {4: 1}}}
    # ipdb> doc_table = dig.document_tables["river"]
    # ipdb> doc_table[1]
    # {('river', 'rafting'): {1: 1, 2: 2, 3: 1}}
    # ipdb> doc_table[2]
    # {('river', 'rafting'): {2: 1}, ('river', 'adventures'): {1: 2}}
    # ipdb> doc_table[3]
    # {('river', 'fishing'): {4: 1}}

    # test_matching_phrases_number()

    # ipdb> len(dig.get_matching_phrases(2,3))
    # 1
    # ipdb> len(dig.get_matching_phrases(1,2))
    # 3
    # ipdb> len(dig.get_matching_phrases(1,3))
    # 0

    # test_multiple_phrase_matches()
    # 
