import networkx as nx

def sentencize(plain_text):
    return plain_text.split(".")

def tokenize(sentence):
    terms = sentence.strip().split(" ")
    return terms

class DocumentIndexGraph(nx.DiGraph):
    """
    Document Index Graph structure
    as defined on the paper

    "Efficient Phrase-Based Document Indexing for Web Document Clustering"
    """
    def __init__(self):
        super(DocumentIndexGraph, self).__init__()
        self.document_tables = {}
        self.next_doc_id = 1
        self.matching_phrases = {}


    def index_document(self, plain_text):
        doc_id = self.next_doc_id
        for prev_id in range(1, doc_id):
            self.matching_phrases[(prev_id, doc_id)] = []

        # split in sentences
        sentences = sentencize(plain_text)
        # tokenize sentences
        sentences = [tokenize(s) for s in sentences]

        for n, sent in enumerate(sentences):
            previous_term = sent[0]
            for p, current_term in enumerate(sent[1:]):
                position = (n + 1, p + 1)
                self.add_edge(doc_id, position, previous_term, current_term)
                previous_term = current_term

        self.next_doc_id += 1

    def add_edge(self, doc_id, position, term1, term2):
        # position is a tuple (sent_n, term_n) indicating
        # the sentence number and term number where the occurence was found
        edge = (term1, term2)
        super(DocumentIndexGraph, self).add_edge(*edge)
        self.document_tables.setdefault(term1, {})
        doc_table = self.document_tables[term1]

        # retrieve list of docs sharing this edge
        # and update phrase matches
        matching_doc_ids = [d_id for d_id in doc_table.keys() if edge in doc_table[d_id]]
        for d_id in matching_doc_ids:
            d_edge_table = doc_table[d_id]
            d_edge_table[edge]

            # search for previous matches ending on term1 and extend them
            for pmatch in self.get_matching_phrases(d_id, doc_id):
                if pmatch.phrase[-1] == term1 and pmatch.end_position(doc_id) == position:
                    pmatch.phrase.append(term2)

            # search for possible new matching phrases and create them


        # update document table with current doc
        doc_table.setdefault(doc_id, {})
        edge_table = doc_table[doc_id]

        edge_table.setdefault(edge, {})
        edge_table[edge][sentence_number] = position_in_sentence





    def get_matching_phrases(self, doc_a_id, doc_b_id):
        doc_a_id, doc_b_id = sorted((doc_a_id, doc_b_id))
        return self.matching_phrases[(doc_a_id, doc_b_id)]


class PhraseMatch(object):
    """docstring for PhraseMatch"""
    def __init__(self, term1, term2, doc_a_id, doc_b_id, pos_a, pos_b):
        """
            doc_a_id, doc_b_id: ids of documents where the matching occurs
            pos_a, pos_b: tuples of the form (sent_n, term_n) that contain
                the respective starting positions of matching phrase within
                doc_a and doc_b
            term1, term2: initial matching terms
        """
        self.phrase = [term1, term2]
        self.positions = {
            doc_a_id: pos_a,
            doc_b_id: pos_b
        }

    def extend(self, term):
        self.phrase.append(term)

    def end_position(doc_id):
        return self.positions[doc_id] + (0, len(self.phrase) - 1)


if __name__ == '__main__':
    docs = ["river rafting. mild river rafting. river rafting trips",
            "wild river adventures. river rafting vacation plan",
            "fishin trips. fishing vacation plan. booking fishing trips. river fishing"]

    dig = DocumentIndexGraph()
    for doc in docs:
        dig.index_document(doc)

    import ipdb; ipdb.set_trace()

    # TODO: turn the following session into a test
    # (this is the example from the figure 3 in the paper,
    #  we use 1-based indices for docs, sentences, etc. to make
    #  make it easier to match results with the paper)

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
