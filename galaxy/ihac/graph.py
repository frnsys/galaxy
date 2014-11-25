import logging
from itertools import chain

import numpy as np
from scipy.sparse import hstack, vstack, csr_matrix

class Graph():
    """
    The Graph manages the adjacency matrix representation of the hierarchy.
    self.mx[i,j] = 1 means i is parent of j
    """
    def __init__(self, mx=None):
        """
        Initialize the graph with an adjacency matrix.
        This should be a scipy sparse csr_matrix.
        """
        # NOTE: for now a csr_matrix is used for persistence,
        # but we are using a normal numpy array internally.
        # i'm still trying to figure out the most appropriate sparse matrix type for the job.
        if mx is not None:
            assert mx.__class__ == csr_matrix, 'The graph adjacency matrix must be a csr_matrix.'
            self.mx = mx.toarray()
        else:
            self.mx = csr_matrix((1,1)).toarray()

    def expand(self):
        """
        Expands the adjacency matrix to (n+1, n+1).
        """
        m = self.mx

        # Add a column and row.
        #m = hstack([m, csr_matrix((m.shape[0], 1))], format='csr')
        #m = vstack([m, csr_matrix((1, m.shape[1]))], format='csr')
        m = np.hstack([m, np.zeros((m.shape[0], 1))])
        m = np.vstack([m, np.zeros(m.shape[1])])

        self.mx = m

    def reset_node(self, n):
        """
        Reset relationships for this node.
        """
        self.mx[n]   = 0  # row, clears children
        self.mx[:,n] = 0  # col, clears parent

    def add_child(self, p, ch):
        """
        Add a new child ch to parent p.
        This will automatically remove ch from its current parent.
        """
        logging.debug('[ADD_CHILD]\t Inserting {0} to {1}...'.format(ch, p))

        # First, clear out any existing parents.
        self.mx[:,ch] = 0
        self.mx[p,ch] = 1

    def remove_child(self, p, ch):
        """
        Remove a child ch from parent p.
        """
        logging.debug('[REMOVE_CHILD]\t Removing {0} from {1}...'.format(ch, p))
        self.mx[p,ch] = 0

    def get_children(self, n):
        """
        Get the children of a node.
        This searches the node's row in the adjacency matrix
        for node ids where the relationship is 1 (parent of).
        """
        return np.where(self.mx[n] == 1)[0]

    def get_siblings(self, n):
        p = self.get_parent(n)
        if p is not None:
            children = self.get_children(p)
            return [ch for ch in children if ch != n]
        return []

    def get_leaves(self, n):
        """
        Get all leaves of a node.
        """
        if not self.is_cluster(n):
            return [n]
        else:
            groups = [self.get_leaves(ch) for ch in self.get_children(n)]
            return list(chain.from_iterable(groups))

    def is_cluster(self, n):
        """
        A cluster node has at least one child.
        """
        return np.any(self.mx[n] == 1)

    def get_parent(self, ch):
        if not np.any(self.mx[:,ch] > 0):
            return None
        return np.argmax(self.mx[:,ch])

    def set_parents(self, p, children):
        # Clear out children's existing parents.
        self.mx[:,children] = 0
        self.mx[p,children] = 1

    def remove_parent(self, ch):
        self.mx[:,ch] = 0

    @property
    def leaves(self):
        """
        Get all leaves in the hierarchy.
        A leaf node is one with no children
        (that is, it's row is entirely 0).
        """
        return np.where(np.all(self.mx == 0, axis=1))[0]

    @property
    def root(self):
        """
        Gets the root node.
        The root node is the node that has no parents,
        that is, its column is all 0,
        and has at least one child.
        """
        return np.where(np.all(self.mx == 0, axis=0) & np.any(self.mx == 1, axis=1))[0][0]

    def is_root(self, n):
        return np.all(self.mx[:,n] == 0) and np.any(self.mx[n] == 1)
