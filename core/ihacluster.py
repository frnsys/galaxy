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
                self.nsiblings = [0]
                self.mu = 0
                self.sigma = 0
            else:
                self.ndists = []
                self.nsiblings = [] # nearest sibling for each child
                for i, ch in enumerate(children):
                    dists = np.array([self.get_distance(ch, x) for x in children])
                    dists[i] = np.inf # to avoid getting itself as nearest sibling
                    j = np.argmin(dists)
                    self.nsiblings.append(children[j])
                    self.ndists.append(dists[j])
                self.mu = scipy.mean(self.ndists)
                self.sigma = scipy.std(self.ndists)

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
            if newd < Node.get_distance(ch, self.nsiblings[i]):
                self.nsiblings[i] = node
            ndists.append(Node.get_distance(ch, self.nsiblings[i]))
            if newd < nsd:
                nsd = newd
                ns = ch
        self.nsiblings.append(ns)
        ndists.append(nsd)
        self.mu = scipy.mean(ndists)
        self.sigma = scipy.std(ndists)

    def remove_child(self, child):
        n = len(self.children)
        self.center = ((self.center * n) - child.center) / (n - 1)
        index = [ch.id for ch in self.children].index(child.id)
        del self.children[index]
        del self.nsiblings[index]
        # update distances to new center
        for n in Node.nodes:
            Node.get_distance(self, n, update=True)
        # update ndp representation
        ndists = []
        for i, ch in enumerate(self.children):
            if self.nsiblings[i].id == child.id:
                dists = np.array([self.get_distance(ch, x) for x in children])
                dists[i] = np.inf # to avoid getting itself as nearest sibling
                j = np.argmin(dists)
                ns = self.children[j]
                self.nsiblings[i] = ns
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

    def get_nearest_child(self, node):
        dists = np.array([Node.get_distance(ch, node) for ch in self.children])
        i = np.argmin(dists)

        return self.children[i], dists[i]

    def get_closest_children(self):
        """
            returns the pair of children having
            the shortest nearest distance
        """
        i = argmin(self.ndists)
        ni = self.children[i]
        nj = self.nsiblings[i]
        d = self.ndists[i]
        return ni, nj, d

    def get_farthest_children(self):
        """
            returns the pair of children having
            the largest nearest distance
        """
        i = argmax(self.ndists)
        mi = self.children[i]
        mj = self.nsiblings[i]
        d = self.ndists[i]
        return mi, mj, d





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
            resize Node data structures to hold process
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
                elif d < node.lower_limit():
                    for ch in node.children:
                        if self.forms_lower_dense_region(new_node, ch):
                            host = nchild
                            break
                    if host:
                        self.ins_hierarchy(host, new_node)
                node = node.parent
            
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
        """
            Algorithm Homogeneity Maintenance

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
        """
        finished = False
        while not finished:
            ni, nj, d = node.get_closest_children()
            if d < n.lower_limit():
                self.merge(ni, nj)
            else:
                finished = True
        mi, mj, d = node.get_farthest_children()
        if d > node.upper_limit():
            # TODO: implement. figure out theta
            ni, nj = self.split(node, mi, mj)
            repair_homogeneity(ni)
            repair_homogeneity(nj)
        

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

    def split(self, nk, mi, mj):
        """
            Performs the split operation over a node nk
            by separating the nearest distance MST structure
            removing the edge joining mi and mj

            mi and mj must be children of nk and
            form a nearest distance edge
        """
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
