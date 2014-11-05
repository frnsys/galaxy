import networkx as nx
import numpy as np
from copy import copy
import string

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from core.vectorize import vectorize
from scipy.spatial.distance import cosine

LEMMATIZER = WordNetLemmatizer()

STOPWORDS = set(list(string.punctuation) + stopwords.words('english'))

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
    sentences = sent_tokenize(plain_text)
    sentences = [tokenize(s) for s in sentences]
    return [RankedSentence(sentence=s, level=LOW) for s in sentences]


def tokenize(sentence):
    tokens = []
    for token in word_tokenize(sentence):
        if token in STOPWORDS:
            continue
        lemma = LEMMATIZER.lemmatize(token.lower())
        tokens.append(lemma)
    return tokens


class Document(object):
    """docstring for Document"""
    def __init__(self, id, plain_text):
        # parse ranked tokenized sentences
        self.id = id
        self.sentences = sentencize(plain_text)
        self.wlength = None # calculate only if needed
        self.tfidf = vectorize(plain_text)

    def get_weight(self, sent_n):
        return WEIGHTS[self.sentences[sent_n].level]

    def get_length(self, sent_n):
        return len(self.sentences[sent_n].sentence)

    def get_normalization_weight(self):
        if not self.wlength:
            wlengths = [self.get_weight(i) * self.get_length(i) for i in range(len(self.sentences))]
            self.wlength = sum(wlengths)

        return self.wlength



class DocumentIndexGraphClusterer(nx.DiGraph):
    """
    Document Index Graph structure

    and clustering algorithm
    as defined on the paper

    "Web Document Clustering Using Document Index Graph"
    """
    def __init__(self):
        super(DocumentIndexGraphClusterer, self).__init__()
        self.document_tables = {}
        self.indexed_docs = []
        self.matching_phrases = {}
        self.phrase_frequencies = {}
        # clustering attributes
        self.formed_clusters = []
        self.phrase_freqs = {} # counts total occurrences of phrases in cluster
        self.phrase_doc_freqs = {} # counts number of docs containing the phrase in cluster


    def index_document(self, plain_text):
        doc_id = len(self.indexed_docs)
        document = Document(doc_id, plain_text)
        self.indexed_docs.append(document)

        # for efficiency of implementation we perform the
        # following tasks simultaneously:
        #  1. store new doc in DIG structure
        #  2. calculate distances to existing clusters
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

        self.assign_cluster(document)

    def get_doc(self, doc_id):
        return self.indexed_docs[doc_id]

    def add_edge(self, doc_id, position, term1, term2, level):
        # position is a tuple (sent_n, term_n) indicating
        # the sentence number and term number where the occurence was found
        edge = (term1, term2)
        super(DocumentIndexGraphClusterer, self).add_edge(*edge)
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
        phrase = tuple(phrase)
        if not (doc_id, phrase) in self.phrase_frequencies:
            count = 0
            for rsentence in self.get_doc(doc_id).sentences:
                sentence = rsentence.sentence
                for i in range(len(sentence)):
                    if sentence[i] == phrase[0] and tuple(sentence[i: i + len(phrase)]) == phrase:
                        count += 1
            self.phrase_frequencies[(doc_id, phrase)] = count
        return self.phrase_frequencies[(doc_id, phrase)]

    def get_sim_p(self, doc_a, doc_b):
        numerator = 0.0
        pmatches = self.get_matching_phrases(doc_a.id, doc_b.id)
        for pmatch in pmatches:
            phrase = pmatch.phrase
            f_a = self.get_phrase_freq(doc_a.id, phrase) 
            f_b = self.get_phrase_freq(doc_b.id, phrase)
            sent_n_a = pmatch.positions[doc_a.id][0]
            sent_n_b = pmatch.positions[doc_b.id][0]
            w_a = doc_a.get_weight(sent_n_a)
            w_b = doc_b.get_weight(sent_n_b)
            numerator += (pmatch.g() * (f_a * w_a + f_b * w_b)) ** 2

        return (numerator ** 0.5) / (doc_a.get_normalization_weight() + doc_b.get_normalization_weight())

    def get_sim_t(self, doc_a, doc_b):
        return cosine(doc_a.tfidf, doc_b.tfidf)

    def get_blended_similarity(self, doc_a_id, doc_b_id, alpha=0.7):
        doc_a = self.get_doc(doc_a_id)
        doc_b = self.get_doc(doc_b_id)        
        sim_p = self.get_sim_p(doc_a, doc_b)
        sim_t = self.get_sim_t(doc_a, doc_b)

        return alpha * sim_p + (1 - alpha) * sim_t

    # Clustering methods
    def get_cluster_sim(self, cluster, doc):
        numerator = 0.0
        pmatches = []
        
        for doc_b_id in cluster.doc_ids:
            pmatches += self.get_matching_phrases(doc.id, doc_b_id)
        
        for pmatch in pmatches:
            phrase = pmatch.phrase

            f_d = self.get_phrase_freq(doc.id, phrase)
            sent_n_d = pmatch.positions[doc.id][0]
            w_d = doc.get_weight(sent_n_d)

            f_c = self.get_cluster_phrase_freq(cluster, phrase)
            d_id = pmatch.doc_a.id if pmatch.doc_a.id != doc.id else pmatch.doc_b.id
            sent_n_c = pmatch.positions[d_id][0]
            w_c = self.get_doc(d_id).get_weight(sent_n_c)

            doc_freq = self.get_cluster_phrase_doc_freq(cluster, phrase)

            ratio = doc_freq * 1.0 / len(cluster.doc_ids)
            
            numerator += (pmatch.g() * (ratio * f_c * w_c + f_d * w_d)) ** 2

        return (numerator ** 0.5) / (cluster.get_normalization_weight() + doc.get_normalization_weight())


    def get_cluster_phrase_freq(self, cluster, phrase):
        if not tuple(phrase) in self.phrase_freqs[cluster.id]:
            self._init_cluster_phrase_freqs(cluster, phrase)

        return self.phrase_freqs[cluster.id][tuple(phrase)]

    def get_cluster_phrase_doc_freq(self, cluster, phrase):
        if not tuple(phrase) in self.phrase_doc_freqs[cluster.id]:
            self._init_cluster_phrase_freqs(cluster, phrase)

        return self.phrase_doc_freqs[cluster.id][tuple(phrase)]

    def _init_cluster_phrase_freqs(self, cluster, phrase):
        """
            Initializes both frequency and document frequency
            for given phrase in cluster
        """
        phrase_freq = 0
        phrase_doc_freq = 0
        for doc_id in cluster.doc_ids:
            dfreq = self.get_phrase_freq(doc_id, phrase)
            if dfreq:
                phrase_freq += dfreq
                phrase_doc_freq += 1

        self.phrase_freqs[cluster.id][tuple(phrase)] = phrase_freq
        self.phrase_doc_freqs[cluster.id][tuple(phrase)] = phrase_doc_freq


    def assign_cluster(self, document, threshold=0.25):
        found_similar_clusters = False

        # calculate similarities and add to similar clusters
        for cluster in self.formed_clusters:
            if self.get_cluster_sim(cluster, document) > threshold:
                found_similar_clusters = True
                cluster.add_doc(document)
                # update phrase_freqs and phrase_doc_freqs where needed
                for phrase in self.phrase_freqs[cluster.id]:
                    dfreq = self.get_phrase_freq(document.id, phrase)
                    if dfreq:
                        self.phrase_freqs[cluster.id][phrase] += dfreq
                        self.phrase_doc_freqs[cluster.id][phrase] += 1

        # if no similar cluster found, create new
        if not found_similar_clusters:
            self.create_cluster(document)

    def create_cluster(self, first_doc):
        new_cluster_id = len(self.formed_clusters)
        self.formed_clusters.append(Cluster(id=new_cluster_id, first_doc=first_doc))
        self.phrase_freqs[new_cluster_id] = {}
        self.phrase_doc_freqs[new_cluster_id] = {}




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

    def end_position(self, doc_id):
        return self.positions[doc_id] + (0, len(self.phrase) - 1)

    def g(self):
        gamma = 1.2
        l_i = len(self.phrase)
        doc_a, doc_b = self.doc_a, self.doc_b
        s_a = len(doc_a.sentences[self.positions[doc_a.id][0]].sentence)
        s_b = len(doc_b.sentences[self.positions[doc_b.id][0]].sentence)
        return (2.0 * l_i / (s_a + s_b)) ** gamma


class Cluster(object):
    """docstring for Cluster"""
    def __init__(self, id, first_doc):
        self.id = id
        self.doc_ids = [first_doc.id]
        self.norm_weight = first_doc.get_normalization_weight()

    def add_doc(self, doc):
        self.doc_ids.append(doc.id)
        self.norm_weight += doc.get_normalization_weight()

    def get_normalization_weight(self):
        return self.norm_weight
        

if __name__ == '__main__':
    docs = ["river rafting. mild river rafting. river rafting trips",
            "wild river adventures. river rafting vacation plan",
            "fishin trips. fishing vacation plan. booking fishing trips. river fishing"]

    dig = DocumentIndexGraphClusterer()
    for doc in docs:
        dig.index_document(doc)

    import ipdb; ipdb.set_trace()
    # print([dig.get_blended_similarity(a, b) for (a, b) in [(0, 1), (1, 2), (0, 2)]])


