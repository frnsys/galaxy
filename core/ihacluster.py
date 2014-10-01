from scipy import clip
from scipy.spatial.distance import pdist, squareform, euclidean
from scipy.cluster.hierarchy import linkage, fcluster

import scipy
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

DISTANCE = 'euclidean'

def distance(vec1, vec2):
    if DISTANCE == 'euclidean':
        return euclidean(vec1, vec2)

class Node(object):
    @classmethod
    def init(cls, size):
        cls.size = size
        m_nodes = 2 * size - 1 # maximum number of nodes in the hierarchy        
        cls.nodes = []  # a list that will hold all the nodes created
        cls.distances = -1 * np.ones((m_nodes, m_nodes)) # A matrix to hold cluster center distances for pairs of node indices

    def __init__(self, vec=None, children=[]):
        """
            A new node is created by passing either:
            - a vector point, in case of a leaf node
            - a list of children, for a cluster node
        """
        Node.nodes.append(self)
        self.id = len(Node.nodes) - 1
        self.children = children
        if children:
            self.center = vec
        else:
            self.center = scipy.mean([c.center for c in children])
            # ndp representation
            if len(children) == 1:
                self.ndp = [0]
                self.mu = 0
                self.sigma = 0
            else:
                ndists = []
                self.ndp = [] # nearest sibling for each child
                for i, ch in enumerate(children):
                    dists = np.array([self.get_distance(ch, x) for x in children])
                    dists[i] = np.inf # to avoid getting itself as nearest sibling
                    j = np.argmin(dists)
                    self.ndp.append(children[j])
                    ndists.append(dists[j])
                self.mu = scipy.mean(ndists)
                self.sigma = scipy.std(ndists)

    def add_child(self, node):
        n = len(self.children)
        self.center = ((self.center * n) + node.center) / (n + 1)
        self.children.append(node)
        # update distances to new center
        for n in Node.nodes:
            Node.get_distance(self, n, update=True)
        # update ndp representation and find nearest sibling
        # for new child node
        ndists = []
        ns = None
        nsd = np.inf
        for i, ch in enumerate(self.children[:-1]):
            newd = Node.get_distance(ch, node)
            if newd < Node.get_distance(ch, self.ndp[i]):
                self.ndp[i] = node
            ndists.append(Node.get_distance(ch, self.ndp[i]))
            if newd < nsd:
                nsd = newd
                ns = ch
        self.ndp.append(ns)
        ndists.append(nsd)
        self.mu = scipy.mean(ndists)
        self.sigma = scipy.std(ndists)


    def remove_child(self, child):
        n = len(self.children)
        self.center = ((self.center * n) - child.center) / (n - 1)
        index = [ch.id for ch in self.children].index(child.id)
        del self.children[index]
        del self.ndp[index]
        # update distances to new center
        for n in Node.nodes:
            Node.get_distance(self, n, update=True)
        # update ndp representation
        ndists = []
        for i, ch in enumerate(self.children):
            if self.ndp[i].id == child.id:
                dists = np.array([self.get_distance(ch, x) for x in children])
                dists[i] = np.inf # to avoid getting itself as nearest sibling
                j = np.argmin(dists)
                ns = self.children[j]
                self.ndp[i] = ns
            ndists.append(Node.get_distance(ch, ns))
        self.mu = scipy.mean(ndists)
        self.sigma = scipy.std(ndists)

    def lower_limit(self):
        n = len(self.children)
        if n > 2:
            return mu - sigma
        else:
            return (2.0 / 3) * self.distances[0][0]

    def upper_limit(self):
        n = len(self.children)
        if n > 2:
            return mu + sigma
        else:
            return 1.5 * self.distances[0][0]

    #
    # Distance functions
    #
    @staticmethod
    def distance(ni, nj, update=False):
        i, j = sorted((ni.id, nj.id))
        current_dist = Node.distances[i, j]
        if current_dist < 0 or (update and current_dist >= 0):
            Node.distances[ni.id, nj.id] = distance(ni.center, nj.center)
        return Node.distances[i, j]

    def get_nearest_child(self, parent, node):
        dists = np.array([self.get_distance(ch, node) for ch in parent.children])
        i = np.argmin(dists)

        return parent.children[i], dists[i]


class Hierarchy(object):
    def __init__(self, size):
        """
            Size is the number of points we plan to cluster
        """
        self.size = size
        self.leaves = set() # Indices of leaf nodes
        self.root = None

    def resize(self):
        """
            resize data structures to hold process
            a new batch of points
        """
        pass
    
    def get_closest_leaf(self, node):
        """
            returns closest leaf node and the distance to it
        """
        mdist = np.inf
        cleaf = None
        for i in self.leaves:
            leaf = Node.nodes[i]
            dist = self.get_distance(leaf, node)
            if dist < mdist:
                mdist = dist
                cleaf = leaf

        return leaf, mdist


    def incorporate(self, vec):
        new_node = Node(vec=vec)
        self.leaves.append(new_node.id)
        if self.root is None:
            self.root = new_node
        if len(self.hierarchy.leaves) > 1:
            leaf, d = self.get_closest_leaf(new_node) 
            host = None
            node = leaf.parent
            while node and host is None:
                nchild, d = node.get_nearest_child(new_node)
                if d >= node.lower_limit() and d <= node.upper_limit():
                    self.ins_node(node, new_node)
                    host = node
                elif d < node.lower_limit:
                    # • if N J forms a higher dense region on N , 
                    # and N J forms a lower dense region on at least one of N ‘s
                    # child nodes 
                    dist = None
                    for ch in node.children:
                        if self.forms_lower_dense_region(new_node, ch):
                            host = nchild
                            break
                    # then perform INS HIERARCHY (N I , N J ) 
                    # where N I is the child node of N closest to the new point N J .
                    # QUESTION: or is N I supposed to be chosen among those child nodes
                    # for which N J forms a lower dense region?
                    if host:
                        self.ins_hierarchy(host, new_node)

            if host is None:
                self.ins_hierarchy(self.root, new_node)

            self.restructure_hierarchy()

    def restructure_hierarchy(self, host_node):
        """Algorithm Hierarchy Restructuring:
            
            Starting on host_node, we traverse ancestors doing
            the following:

            1. Recover the siblings of current that are misplaced.
                (A node N J is misplaced as N I ’s sibling iff
                N J does not form a lower dense region in N I )
                In such case we apply DEMOTE(N I , N J )

            2. Maintain the homogeneity of crntNode.
        """
        current = host_node
        while current:
            parent = current.parent
            siblings = parent.children
            misplaced = [s for s in siblings if not self.forms_lower_dense_region(s, current)]
            for node in misplaced:
                self.demote(current, node)

            self.repair_homogeneity(current)
            current = parent
    
    def repair_homogeneity(self, node):
        # Algorithm Homogeneity Maintenance(N )
        # 1. Let an input N be the node that is being examined.
        # 2. Repeat
        # 3. Let N I and N J be the pair of neighbors among N ‘s
        # child nodes with the smallest nearest distance.
        # 4. If N I and N J form a higher dense region,
        # 5. Then MERGE (N I , N J ) (see Figure 1d)
        # 6. Until there is no higher dense region found in N during
        # the last iteration.
        # 7. Let M I and M J be the pair of neighbors among N ‘s
        # child nodes with the largest nearest distance.
        # 8. If M I and M J form a lower dense region in N ,
        # 9. Then Let (N I , N J ) = SPLIT (Θ, N ). (see Figure 1e)
        # 10. Call Homogeneity Maintenance(N I ).
        # 11. Call Homogeneity Maintenance(N J ).
        pass

    def forms_lower_dense_region(self, a, c):
        """
        Let C be a homogenous cluster. 
        Given a new point A, let B be a C‘s cluster member that is the nearest
        neighbor to A. Let d be the distance from A to B. A (and B)
        is said to form a lower dense region in C if d > U_L
        """
        b = None
        pass

    def forms_higher_dense_region(self, a, c):
        """
        Let C be a homogenous cluster. Given a new
        point A, let B be a C‘s cluster member that is the nearest
        neighbor to A. Let d be the distance from A to B. A (and B)
        is said to form a higher dense region in C if d < L_L .
        """
        b = None
        pass


    #
    # Restructuring operators
    #
    def ins_node(self, ni, nj):
        ni.add_child(nj)

    def ins_hierarchy(self, n, ni, nj):
        n.remove_child(ni)
        nk = Node(children=[ni, nj])
        n.add_child(nk)

    def demote(self, n, ni, nj):
        n.remove_child(nj)
        ni.add_child(nj)

    def merge(self, ni, nj):
        n.remove_child(ni)
        n.remove_child(nj)
        nk = Node(children=[ni, nj])
        n.add_child(nk)

    def split(self, theta, nk):
        pass


class IHAClusterer(object):
    def __init__(self, vecs):
        Node.init(m_nodes)
        self.hierarchy = Hierarchy(len(vecs))

    def cluster(self):
        for vec in vecs:
            self.hierarchy.incorporate(vec)

        labels = fcluster(self.hierarchy)

        return labels


    def fcluster(hierarchy, density_threshold=None):
        """
            Creates flat clusters by pruning all clusters
            with density higher than the given threshold
            and taking the leaves of the resulting hierarchy

            In case no density_threshold is given,
            we use the average density accross the entire hierarchy
        """
        pass
