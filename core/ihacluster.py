from scipy import clip
from scipy.spatial.distance import pdist, squareform, euclidean
from scipy.cluster.hierarchy import linkage, fcluster

import scipy
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import networkx as nx

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

    @classmethod
    def get(cls, id):
        return cls.nodes[id]

    def __init__(self, vec=None, children=[], ndists=[], nsiblings=[]):
        """
            A new node is created by passing either:
            - a vector point, in case of a leaf node
            - a list of children, for a cluster node
        """
        Node.nodes.append(self)
        self.id = len(Node.nodes) - 1
        self.children = children
        self.parent = None
        if not children:
            self.center = vec
        else:
            self.initialize_ndp(children, ndists, nsiblings)

    def initialize_ndp(self, children, ndists=[], nsiblings=[]):
        self.children = children
        for ch in children:
            ch.parent = self
        self.center = scipy.mean([c.center for c in children], axis=0)
        # ndp representation
        if len(children) == 1:
            self.nsiblings = []
            self.mu = 0
            self.sigma = 0
        else:
            if ndists: # in case of split, we reuse distances
                self.ndists = ndists
                self.nsiblings = nsiblings
            else:
                self.ndists = []
                self.nsiblings = [] # nearest sibling for each child
                for i, ch in enumerate(children):
                    dists = np.array([Node.get_distance(ch, x) for x in children])
                    dists[i] = np.inf # to avoid getting itself as nearest sibling
                    j = np.argmin(dists)
                    self.nsiblings.append(children[j])
                    self.ndists.append(dists[j])
            self.mu = scipy.mean(self.ndists)
            self.sigma = scipy.std(self.ndists)


    def add_child(self, new_child):
        n = len(self.children)
        if n < 2:
            self.initialize_ndp(self.children + [new_child])
        else:
            self.center = ((self.center * n) + new_child.center) / (n + 1)
            self.children.append(new_child)
            new_child.parent = self
            # update distances to new center
            for n in Node.nodes:
                Node.get_distance(self, n, update=True)
            # update ndp representation and find nearest sibling
            # for new child new_child
            ns = None
            ns_dist = np.inf
            for i, ch in enumerate(self.children[:-1]):
                newd = Node.get_distance(ch, new_child)
                if newd < self.ndists[i]:
                    self.nsiblings[i] = new_child
                    self.ndists[i] = newd
                if newd < ns_dist:
                    ns_dist = newd
                    ns = ch
            self.nsiblings.append(ns)
            self.ndists.append(ns_dist)
            self.mu = scipy.mean(self.ndists)
            self.sigma = scipy.std(self.ndists)

    def remove_child(self, child):
        n = len(self.children)
        if n > 1:
            self.center = ((self.center * n) - child.center) / (n - 1)
            index = [ch.id for ch in self.children].index(child.id)
            del self.children[index]
            del self.nsiblings[index]
            # update distances to new center
            for n in Node.nodes:
                Node.get_distance(self, n, update=True)
            # update ndp representation
            for i, ch in enumerate(self.children):
                if self.nsiblings[i].id == child.id:
                    dists = np.array([Node.get_distance(ch, x) for x in self.children])
                    dists[i] = np.inf # to avoid getting itself as nearest sibling
                    j = np.argmin(dists)
                    ns = self.children[j]
                    self.nsiblings[i] = ns
                    self.ndists[i] = dists[j]
            self.mu = scipy.mean(self.ndists)
            self.sigma = scipy.std(self.ndists)
        else:
            self.children = []
            self.ndists = []
            self.nsiblings = []

    def split_children(self, mi, mj):
        """
            mi and mj must be children of self 
            connected by a nearest distance edge.
            Children MST structure is broken in two
            disjoint sets by removing the edge for mi and mj
            and new clusters are formed on those sets
        """
        children_ids = [ch.id for ch in self.children]
        nsibling_ids = [ns.id for ns in self.nsiblings]
        ndists_by_id = {ch.id: self.ndists[i] for i, ch in enumerate(self.children)}
        nsibling_ids_by_id = {ch.id: nsibling_ids[i] for i, ch in enumerate(self.children)}

        # create partitions s1 and s2
        edges = list(zip(children_ids, nsibling_ids))        
        graph = nx.Graph()
        graph.add_edges_from(edges)
        graph.remove_edge(mi.id, mj.id)
        s1_ids, s2_ids = list(nx.connected_components(graph))
        if mi.id in s1_ids:
            si_ids, sj_ids = s1_ids, s2_ids
        else:
            si_ids, sj_ids = s2_ids, s1_ids

        # mi and mj could have their nearest sibling changed in
        # new clusters, so we add them separately
        si_ids.remove(mi.id)
        sj_ids.remove(mj.id)
        # create ndist and nsiblings for new partitions
        si_nodes = [Node.get(id) for id in si_ids]
        sj_nodes = [Node.get(id) for id in sj_ids]
        
        si_ndists = [ndists_by_id[id] for id in si_ids]
        sj_ndists = [ndists_by_id[id] for id in sj_ids]

        si_nsiblings = [Node.get(nsibling_ids_by_id[id]) for id in si_ids]
        sj_nsiblings = [Node.get(nsibling_ids_by_id[id]) for id in sj_ids]

        # create new nodes n1 and n2 with the split data
        ni = Node(children=si_nodes, ndists=si_ndists, nsiblings=si_nsiblings)
        ni.add_child(mi)
        nj = Node(children=sj_nodes, ndists=sj_ndists, nsiblings=sj_nsiblings)
        nj.add_child(mj)

        import ipdb; ipdb.set_trace()
        return ni, nj

    def lower_limit(self):
        n = len(self.children)
        if n > 2:
            return self.mu - self.sigma
        else:
            return (2.0 / 3) * self.ndists[0]

    def upper_limit(self):
        n = len(self.children)
        if n > 2:
            return self.mu + self.sigma
        else:
            return 1.5 * self.ndists[0]

    def is_root(self):
        return self.parent is None                
    #
    # Distance functions
    #
    @staticmethod
    def get_distance(ni, nj, update=False):
        i, j = sorted((ni.id, nj.id))
        current_dist = Node.distances[i, j]
        if current_dist < 0 or update:
            Node.distances[i, j] = distance(ni.center, nj.center)
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
        i = np.argmin(self.ndists)
        ni = self.children[i]
        nj = self.nsiblings[i]
        d = self.ndists[i]
        return ni, nj, d

    def get_farthest_children(self):
        """
            returns the pair of children having
            the largest nearest distance
        """
        i = np.argmax(self.ndists)
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
    
    def fcluster(self, density_threshold=None):
        """
        Creates flat clusters by pruning all clusters
        with density higher than the given threshold
        and taking the leaves of the resulting hierarchy

        In case no density_threshold is given,
        we use the average density accross the entire hierarchy
        """
        pass

    def get_cluster_points(self, node):
        """
        Returns all the points inside the cluster represented
        by the given node. That is, the centers of the descendant leaf nodes
        """
        if node.id in self.leaves:
            return [node.center]
        else:
            points = []
            for ch in node.children:
                points += self.get_cluster_points(ch)
            return points

    def get_closest_leaf(self, node):
        """
            returns closest leaf node and the distance to it
        """
        mdist = np.inf
        cleaf = None
        for i in self.leaves:
            leaf = Node.nodes[i]
            dist = Node.get_distance(leaf, node)
            if dist < mdist:
                mdist = dist
                cleaf = leaf

        return leaf, mdist

    def incorporate(self, vec):
        new_node = Node(vec=vec)
        if self.root is None: # first cluster contains the new point
            first_cluster = Node(children=[new_node])
            self.root = first_cluster
        else:
            leaf, d = self.get_closest_leaf(new_node) 
            host = None
            node = leaf
            while not node.parent.is_root() and host is None:
                print("A")
                node = node.parent
                nchild, d = node.get_nearest_child(new_node)
                if d >= node.lower_limit() and d <= node.upper_limit():
                    host = node
                    self.ins_node(node, new_node)
                elif d < node.lower_limit():
                    for ch in node.children:
                        if self.forms_lower_dense_region(new_node, ch):
                            host = nchild
                            break
                    if host:
                        self.ins_hierarchy(host, new_node)
            
            print("host search finished")
            if host is not None: # node is top level cluster
                print("host found")
                self.restructure_hierarchy(host)
            else:
                # TODO: make sure the no restructuring is required in case of top level insertion                
                print("host not found")
                self.ins_hierarchy(node, new_node)
        self.leaves.add(new_node.id)


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
            if not current.is_root():
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
            if len(node.children) < 2:
                finished = True
            else:
                ni, nj, d = node.get_closest_children()
                if d < node.lower_limit():
                    self.merge(ni, nj)
                else:
                    finished = True
        if len(node.children) >= 2:
            mi, mj, d = node.get_farthest_children()
            if d > node.upper_limit():
                ni, nj = self.split(node, mi, mj)
                self.repair_homogeneity(ni)
                self.repair_homogeneity(nj)

    def forms_lower_dense_region(self, a, c):
        """
        Let C be a homogenous cluster. 
        Given a new point A, let B be a C‘s cluster member that is the nearest
        neighbor to A. Let d be the distance from A to B. A (and B)
        is said to form a lower dense region in C if d > U_L
        """
        if c.children:
            nearest_child, d = c.get_nearest_child(a)
            return d > c.upper_limit()
        else:
            return False

    def forms_higher_dense_region(self, a, c):
        """
        Let C be a homogenous cluster. Given a new
        point A, let B be a C‘s cluster member that is the nearest
        neighbor to A. Let d be the distance from A to B. A (and B)
        is said to form a higher dense region in C if d < L_L .
        """
        if c.children:
            nearest_child, d = c.get_nearest_child(a)
            return d < c.lower_limit()
        else:
            return False



    #
    # Restructuring operators
    #
    def ins_node(self, ni, nj):
        ni.add_child(nj)

    def ins_hierarchy(self, ni, nj):
        nn = ni.parent
        nn.remove_child(ni)
        nk = Node(children=[ni, nj])
        nn.add_child(nk)

    def demote(self, ni, nj):
        nn = ni.parent
        nn.remove_child(nj)
        ni.add_child(nj)

    def merge(self, ni, nj):
        nn = ni.parent
        nn.remove_child(ni)
        nn.remove_child(nj)
        nk = Node(children=[ni, nj])
        nn.add_child(nk)

    def split(self, nk, mi, mj):
        """
            Performs the split operation over a node nk
            by separating the nearest distance MST structure
            removing the edge joining mi and mj

            mi and mj must be children of nk and
            form a nearest distance edge
        """
        nn = nk.parent
        n1, n2 = nk.split_children(mi, mj)
        nn.remove_child(nk)
        nn.add_child(n1)
        nn.add_child(n2)
        # TODO: implement delete
        # nk.delete()


class IHAClusterer(object):
    def __init__(self, vecs):
        size = len(vecs)
        Node.init(size)
        self.vecs = vecs
        self.hierarchy = Hierarchy(len(vecs))

    def cluster(self):
        for vec in self.vecs:
            print("processing " + repr(vec))
            self.hierarchy.incorporate(vec)
            print("OK")

    def get_labels(self):
        labels = self.hierarchy.fcluster()
        return labels




# Test code
def create_2_1dim_clusters():
    cluster_a = np.arange(0,1,0.1)
    cluster_b = np.arange(2,3,0.1)
    print("Created clusters:\n")
    print("A: ")
    print(cluster_a)
    print("\nB: ")
    print(cluster_b)

    points = np.append(cluster_a, cluster_b)
    np.random.shuffle(points)
    points = [np.array([p]) for p in points]
    return points


def test_2_clusters_1_dimension():
    # create sample data
    points = create_2_1dim_clusters()
    # apply clustering
    clusterer = IHAClusterer(points)
    clusterer.cluster()
    return clusterer


if __name__ == '__main__':
    clusterer = test_2_clusters_1_dimension()
    root = clusterer.hierarchy.root
    top = root.children[0]
    hi = clusterer.hierarchy
    second = top.children
    cluster_a = hi.get_cluster_points(second[0])
    cluster_b = hi.get_cluster_points(second[1])

    print("Found clusters:\n")
    print("A: ")
    print(cluster_a)
    print("\nB: ")
    print(cluster_b)
