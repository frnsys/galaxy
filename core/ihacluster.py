#!/usr/bin/env python # -*- coding: utf-8
from scipy import clip
from scipy.spatial.distance import pdist, squareform, euclidean
from scipy.cluster.hierarchy import linkage, fcluster

import scipy
import numpy as np
import os
from sklearn.metrics.pairwise import pairwise_distances
import networkx as nx
import pickle

from eval.data import load_articles, build_vectors

DISTANCE = 'euclidean'

def n_pairs(m):
    return (m * (m - 1)) / 2


def distance(vec1, vec2):
    if DISTANCE == 'euclidean':
        return euclidean(vec1, vec2)


class Node(object):
    @classmethod
    def init(cls, size):
        cls.size = size # the number of leaf nodes the structure can handle
        cls.max_n_nodes = 4 * size - 1 # maximum number of nodes in the hierarchy
                                   # if every cluster node has at least two children
        n_distances = n_pairs(cls.max_n_nodes) # maximum number of cached distances
        cls.available_ids = set(range(cls.max_n_nodes))   
        cls.nodes = {}  # a dictionary to hold all the nodes created indexed by id
        cls.distances = -1 * np.ones(n_distances) # A *condensed* matrix for distances between cluster centers


    def __init__(self, parent=None):
        """
            A new node is created by passing either:
            - a vector point, in case of a leaf node
            - a list of children, for a cluster node
        """
        self.parent = parent
        if len(Node.available_ids) == 0:
            Node.enlarge_node_number()
        self.id = Node.available_ids.pop()
        Node.nodes[self.id] = self
        self.label = None
        self.children = []

    
    @classmethod
    def enlarge_point_number(cls, extra_size):
        """
            Enlarge data structures to handle more
            points
        """
        cls.size += extra_size
        new_max_n_nodes = 4 * cls.size - 1
        extra_n_nodes = new_max_n_nodes - cls.max_n_nodes
        cls.enlarge_node_number(extra_n_nodes)

    @classmethod
    def enlarge_node_number(cls, extra_n_nodes=100):
        """
            It is automatically called if for some reason
            the algorithm needs more nodes than we have
            reserved space for
        """
        cls.available_ids.update(set(range(cls.max_n_nodes, cls.max_n_nodes + extra_n_nodes)))
        old_n_distances = n_pairs(cls.max_n_nodes)
        cls.max_n_nodes += extra_n_nodes
        new_n_distances = n_pairs(cls.max_n_nodes)
        extra_n_distances = new_n_distances - old_n_distances
        cls.distances = np.append(cls.distances, -1 * np.ones(extra_n_distances))

    @classmethod
    def get(cls, id):
        return cls.nodes[id]

    @classmethod
    def get_bad_nodes(cls):
        cluster_nodes = [n for n in cls.nodes.values() if type(n) is ClusterNode]
        return [n for n in cluster_nodes if len(n.children) == 1]


    def __repr__(self):
        node_type = str(type(self)).split(".")[1].split("'")[0]
        center_str = "[" + ", ".join(["%.2f" % x for x in self.center]) + "]"
        node_str = "%s [%d]" % (node_type, self.id)
        # node_str = "%s [%d] (%s)" % (node_type, self.id, center_str)

        if self.is_root():
            node_str += "<ROOT>"
        return node_str

    def pretty_print(self, depth=0):
        """
        Prints the categorization tree.
        """
        ret = ('\t' * depth) + "|- %s: \n" % self.get_label()
        
        for c in self.children:
            ret += c.pretty_print(depth+1)

        return ret


    def get_label(self):
        if self.label:
            return self.label
        elif self.is_root():
            return "ROOT"
        elif self.center.shape == (1,):
            return "%.2f" % self.center
        else:
            return str(self.id)

    def delete(self):
        for node in Node.nodes.values():
            Node.del_distance(self, node)

        Node.available_ids.add(self.id)
        del Node.nodes[self.id]

    def get_cluster_leaves(self):
        """
        Returns all the points inside the cluster represented by
        the node, i.e.: all the descendant leaf nodes
        """
        raise NotImplementedError

    def get_siblings(self):
        if self.parent:
            return [ch for ch in self.parent.children if ch.id != self.id]
        else:
            return []

    #
    # Distance functions
    #
    @staticmethod
    def get_distance(ni, nj, update=False):
        i, j = sorted((ni.id, nj.id))
        if i == j:
            return 0.0
        else:
            pos = int(j * (j - 1) / 2) + i
            current_dist = Node.distances[pos]
            if current_dist < 0 or update:
                if ni.center is None or nj.center is None:
                    Node.distances[pos] = -1
                else:
                    Node.distances[pos] = distance(ni.center, nj.center)
            return Node.distances[pos]

    @staticmethod
    def del_distance(ni, nj):
        i, j = sorted((ni.id, nj.id))
        if i != j:
            pos = int(j * (j - 1) / 2) + i
            Node.distances[pos] = -1


    def forms_lower_dense_region(self, c):
        """
        Let C be a homogenous cluster. 
        Given a new point A, let B be a C's cluster member that is the nearest
        neighbor to A. Let d be the distance from A to B. A (and B)
        is said to form a lower dense region in C if d > U_L
        """
        if type(c) is ClusterNode:
            nearest_child, d = c.get_nearest_child(self)
            return d > c.upper_limit()
        else:
            return False

    def forms_higher_dense_region(self, c):
        """
        Let C be a homogenous cluster. Given a new
        point A, let B be a C‘s cluster member that is the nearest
        neighbor to A. Let d be the distance from A to B. A (and B)
        is said to form a higher dense region in C if d < L_L .
        """
        if type(c) is ClusterNode:
            nearest_child, d = c.get_nearest_child(a)
            return d < c.lower_limit()
        else:
            return False


class LeafNode(Node):
    def __init__(self, vec, parent=None):
        super(LeafNode, self).__init__(parent)
        self.center = vec

    def get_cluster_leaves(self):
        return [self]

    def is_root(self):
        return False


class ClusterNode(Node):
    def __init__(self, children, ndists=[], nsiblings=[], parent=None):
        """
            In case the cluster is being created as subcluster
            of an existing one, we can reuse ndists and nsiblings
        """
        super(ClusterNode, self).__init__(parent)
        self.initialize_ndp(children, ndists, nsiblings)

    def initialize_ndp(self, children, ndists=[], nsiblings=[]):
        self.children = children
        for ch in children:
            ch.parent = self

        if children:
            self.center = scipy.mean([c.center for c in children], axis=0)
        else:
            self.center = None
        # ndp representation
        if len(children) <= 1: # can only be zero when invoked in split
            self.nsiblings = []
            self.ndist_mean = 0
            self.ndist_std = 0
            self.ndists = [0.0]
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
            self.ndist_mean = scipy.mean(self.ndists)
            self.ndist_std = scipy.std(self.ndists)
        
    def add_child(self, new_child):
        n = len(self.children)
        if n < 2:
            self.initialize_ndp(self.children + [new_child])
        else:
            self.center = ((self.center * n) + new_child.center) / (n + 1)
            self.children.append(new_child)
            new_child.parent = self
            # update distances to new center
            for node in Node.nodes.values():
                Node.get_distance(self, node, update=True)
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
            self.ndist_mean = scipy.mean(self.ndists)
            self.ndist_std = scipy.std(self.ndists)

    def remove_child(self, child):
        child.parent = None
        nchildren = len(self.children)
        if nchildren == 1:
            self.center = None
            self.children = []
            self.ndists = []
            self.nsiblings = []
        else:
            # remove child, update center and distances
            self.center = ((self.center * nchildren) - child.center) / (nchildren - 1)
            index = [ch.id for ch in self.children].index(child.id)
            del self.children[index]
            del self.nsiblings[index]
            del self.ndists[index]
            for node in Node.nodes.values():
                Node.get_distance(self, node, update=True)

            if len(self.children) > 1:
                # update ndp representation
                for i, ch in enumerate(self.children):
                    if self.nsiblings[i].id == child.id:
                        dists = np.array([Node.get_distance(ch, x) for x in self.children])
                        dists[i] = np.inf # to avoid getting itself as nearest sibling
                        j = np.argmin(dists)
                        ns = self.children[j]
                        self.nsiblings[i] = ns
                        self.ndists[i] = dists[j]
                self.ndist_mean = scipy.mean(self.ndists)
                self.ndist_std = scipy.std(self.ndists)
            else:
                self.nsiblings = []
                self.ndist_mean = 0
                self.ndist_std = 0

    def split_children(self, mi, mj):
        """
            Splits the set of children of self
            in two groups containing mi and mj

            Children ndist MST structure is broken in two
            disjoint sets by removing the edge for mi and mj
            and new clusters are formed on those sets
            (unless some of them contains just one element)

            if mi and mj are connected by a nearest distance edge,
            groups are formed by removing that edge.

            Otherwise, we remove an edge near the middle
            of the ndist-MST path joining them.
        """
        children_ids = [ch.id for ch in self.children]
        nsibling_ids = [ns.id for ns in self.nsiblings]
        ndists_by_id = {ch.id: self.ndists[i] for i, ch in enumerate(self.children)}
        nsibling_ids_by_id = {ch.id: nsibling_ids[i] for i, ch in enumerate(self.children)}

        # create partitions s1 and s2
        edges = list(zip(children_ids, nsibling_ids))        
        graph = nx.Graph()
        graph.add_edges_from(edges)
        
        if graph.has_edge(mj.id, mi.id):
            graph.remove_edge(mi.id, mj.id)
        else:
            path = nx.shortest_path(graph, source=mi.id, target=mj.id)
            half = int(len(path)/2)
            graph.remove_edge(path[half], path[half + 1])

        try:
            s1_ids, s2_ids = list(nx.connected_components(graph))
            if mi.id in s1_ids:
                si_ids, sj_ids = s1_ids, s2_ids
            else:
                si_ids, sj_ids = s2_ids, s1_ids
        except Exception:
            # TODO: figure out why this happens
            # more than one component:
            # we isolate j and merge the other components into si
            components = list(nx.connected_components(graph))
            si_ids = []
            for comp_ids in components:
                if mj.id in comp_ids:
                    sj_ids = comp_ids
                else:
                    si_ids += comp_ids

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
        if len(si_nodes) > 0:
            ni = ClusterNode(children=si_nodes, ndists=si_ndists, nsiblings=si_nsiblings)
            ni.add_child(mi)
        else:
            ni = mi
        
        if len(sj_nodes) > 0:        
            nj = ClusterNode(children=sj_nodes, ndists=sj_ndists, nsiblings=sj_nsiblings)
            nj.add_child(mj)
        else:
            nj = mj

        return ni, nj

    def get_nearest_child(self, node):
        dists = np.array([Node.get_distance(ch, node) for ch in self.children])
        i = np.argmin(dists)
        return self.children[i], dists[i]

    def get_nearest_cluster_child(self, node):
        dists = np.array([Node.get_distance(ch, node) for ch in self.children if type(ch) is ClusterNode])
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

    def get_cluster_leaves(self):
        points = []
        for ch in self.children:
            points += ch.get_cluster_leaves()
        return points

    def lower_limit(self):
        # TODO: what happens in case of just one child? 
        # (for example, after a split)
        n = len(self.children)
        if n > 2:
            return self.ndist_mean - self.ndist_std
        else:
            return (2.0 / 3) * self.ndists[0]

    def upper_limit(self):
        n = len(self.children)
        if n > 2:
            return self.ndist_mean + self.ndist_std
        else:
            return 1.5 * self.ndists[0]

    def is_root(self):
        return self.parent is None

    def visualize(self, onedim=False):
        import matplotlib.pyplot as plt

        current_level = [self]
        root_label = self.get_label()

        G = nx.DiGraph()
        G.add_node(root_label)
        pos = {}
        pos[root_label] = (self.center, 1.0) if onedim else (0.5, 1)
        n_level = 0
        while current_level:
            next_level = []
            n_level += 1
            for i, n in enumerate(current_level):
                step = 1.0 / (len(current_level) + 1)
                n_label = n.get_label()
                for ch in n.children:
                    ch_label = ch.get_label()
                    G.add_node(ch_label)
                    x_coord = ch.center if onedim else (i + 1) * step
                    pos[ch_label] = (x_coord, 1.0 - 0.02 * n_level) 
                    G.add_edge(n_label, ch_label)
                    if type(ch) == ClusterNode:
                        next_level.append(ch)
            current_level = next_level
        
        plt.title("IHAC hierarchy")
        # if not onedim:
        #     pos = nx.spring_layout(G)
        # pos = nx.graphviz_layout(G, prog='dot')
        nx.draw(G, pos, with_labels=True, arrows=False)

        plt.show()
               

class IHACHierarchy(object):
    def __init__(self, size, vec1, vec2, vec1_tag=None, vec2_tag=None):
        """
            Size is the number of points we plan to cluster
        """
        leaf1 = LeafNode(vec1)
        leaf2 = LeafNode(vec2)
        leaf1.label = vec1_tag
        leaf2.label = vec2_tag
        self.root = ClusterNode(children=[leaf1, leaf2])
        self.size = size
        self.leaves = [leaf1, leaf2] # Indices of leaf nodes

    def incorporate(self, vec, vec_tag=None):
        new_leaf = LeafNode(vec=vec)
        new_leaf.label = vec_tag
        closest_leaf, dist = self.get_closest_leaf(new_leaf)

        current = closest_leaf.parent
        nchild = closest_leaf
        found_host = None
        while current and not found_host:
            if dist >= current.lower_limit() and dist <= current.upper_limit():
                self.ins_node(current, new_leaf)
                found_host = current; break
            elif dist < current.lower_limit():
                for ch in current.children:
                    if new_leaf.forms_lower_dense_region(ch) or type(ch) is LeafNode:
                        self.ins_hierarchy(nchild, new_leaf)
                        found_host = nchild.parent; break # new cluster
                        # QUESTION: or should we make found_host = current here?
            next = current.parent
            if next:
                current = next
                nchild, dist = current.get_nearest_cluster_child(new_leaf)
            else: # reached top level
                break

        # print("host search finished")
        if found_host: # node is top level cluster
            # print("host found")
            self.restructure_hierarchy(found_host)
        else:
            # print("host not found")
            self.ins_hierarchy(current, new_leaf)
        self.leaves.append(new_leaf)

    def restructure_hierarchy(self, host_node):
        """ Starting on host_node, we traverse ancestors doing
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
                misplaced = [s for s in current.get_siblings() if not s.forms_lower_dense_region(current)]
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
            # 6. Until there is no higher dense region found_host in N during
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
            if len(node.children) < 3:
                finished = True
            else:
                ni, nj, d = node.get_closest_children()
                if d < node.lower_limit():
                    self.merge(ni, nj)
                else:
                    finished = True
        if len(node.children) >= 3:
            mi, mj, d = node.get_farthest_children()
            if d > node.upper_limit():
                # WARNING! split over nodes that are not nearest won't
                # create two partitions as expected
                # TODO: find ndist edge to perform this split
                ni, nj = self.split(node, mi, mj)
                if type(ni) is ClusterNode:
                    self.repair_homogeneity(ni)
                if type(nj) is ClusterNode:
                    self.repair_homogeneity(nj)
    
    def fcluster(self, distance_threshold=None):
        """
        Creates flat clusters by pruning all clusters
        with density higher than the given threshold
        and taking the leaves of the resulting hierarchy

        In case no distance_threshold is given,
        we use the average density accross the entire hierarchy
        (average of averages per level)
        """
        if distance_threshold is None:
            distance_threshold = self.get_average_density()

        clusters = []
        labels = {} # dictionary mapping leaf ids to cluster indices
        current_level = [self.root]
        while current_level:
            next_level = []
            for n in current_level:
                new_cluster = None
                if type(n) is LeafNode:
                    new_cluster = [n]
                if type(n) is ClusterNode:
                    if n.ndist_mean <= distance_threshold:
                        new_cluster = n.get_cluster_leaves()
                    else:
                        next_level += n.children
                if new_cluster:
                    ind_cluster = len(clusters)
                    clusters.append([l.center for l in new_cluster])
                    labels.update({l.id: ind_cluster for l in new_cluster})
            current_level = next_level

        return clusters, labels

    def get_average_density(self):
        level_averages = []
        current_level = [self.root]
        while current_level:
            next_level = []
            level_averages.append(scipy.mean([n.ndist_mean for n in current_level]))
            for n in current_level:
                next_level += [ch for ch in n.children if type(ch) is ClusterNode]
            current_level = next_level
        return scipy.mean(level_averages)

    def visualize(self, onedim=False):
        onedim = self.root.center.shape == (1,)
        root = self.root.visualize(onedim=onedim)

    def get_closest_leaf(self, node):
        """
            returns closest leaf node and the distance to it
        """
        dists = np.array([Node.get_distance(leaf, node) for leaf in self.leaves])
        i = np.argmin(dists)
        return self.leaves[i], dists[i]

    #
    # Restructuring operators
    #
    def ins_node(self, ni, nj):
        ni.add_child(nj)

    def ins_hierarchy(self, ni, nj):
        if not ni.is_root():
            nn = ni.parent
            nn.remove_child(ni)
            nk = ClusterNode(children=[ni, nj])
            nn.add_child(nk)
        else:
            nk = ClusterNode(children=[ni, nj])
            self.root = nk

    def demote(self, ni, nj):
        nn = ni.parent
        nn.remove_child(nj)
        ni.add_child(nj)

    def merge(self, ni, nj):
        nn = ni.parent
        nn.remove_child(ni)
        nn.remove_child(nj)
        nk = ClusterNode(children=[ni, nj])
        nn.add_child(nk)

    def split(self, nk, mi, mj):
        """
            Performs the split operation over a node nk
            by separating the nearest distance MST structure
            removing the edge joining mi and mj

            mi and mj must be children of nk and
            form a nearest distance edge
        """
        ni, nj = nk.split_children(mi, mj)
        if nk.is_root():
            self.root = ClusterNode(children=[ni, nj])
        else:
            nn = nk.parent        
            nn.remove_child(nk)
            nn.add_child(ni)
            nn.add_child(nj)
        nk.delete()

        return ni, nj


class IHAClusterer(object):
    def __init__(self):
        pass

    def save(self, prefix):
        with open(prefix + 'hierarchy.dump', 'wb') as f:
            data = {
                'Node': {
                    'size': Node.size,
                    'max_n_nodes': Node.max_n_nodes,
                    'available_ids': Node.available_ids,
                    'nodes': Node.nodes,
                    'distances': Node.distances,
                },
                'self': {
                    'hierarchy': self.hierarchy,
                    'vecs': self.vecs,
                    'size': self.size                
                }
            }    
            pickle.dump(data, f)

    def load(self, prefix):
        with open(prefix + 'hierarchy.dump', 'rb') as f:    
            data = pickle.load(f)
            for name, val in data["Node"].items():
                setattr(Node, name, val)
            for name, val in data["self"].items():
                setattr(self, name, val)

    def fit(self, vecs, vec_tags=None):
        self.size = len(vecs)
        Node.init(self.size)
        self.vecs = vecs
        # print("initializing with %s and %s" % (repr(vecs[0]), repr(vecs[1])))
        vec0_tag = vec_tags[0] if vec_tags else None
        vec1_tag = vec_tags[1] if vec_tags else None
        self.hierarchy = IHACHierarchy(len(vecs), vecs[0], vecs[1], vec0_tag, vec1_tag)

        for i, vec in enumerate(self.vecs[2:]):
            # print("processing " + repr(vec))
            vec_tag = vec_tags[2 + i] if vec_tags else None
            self.hierarchy.incorporate(vec, vec_tag)
            # print("OK")

    def fit_more(self, vecs):
        Node.enlarge_point_number(len(vecs))
        for vec in vecs:
            # print("processing " + repr(vec))
            self.hierarchy.incorporate(vec)
            # print("OK")
        self.vecs += vecs
        self.size += len(vecs)
        self.labels_ = None # reset labels



    def get_labels(self):
        _, dict_labels = self.hierarchy.fcluster()
        leaf_ids = [l.id for l in self.hierarchy.leaves]
        self.labels_ = [dict_labels[lid] for lid in leaf_ids]
        return self.labels_


# TODO: refactor as unittests and move to tests.py
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


def test_no_cluster_node_with_single_cluster_child():
    points = [0.30, 0.40, 0.80, 2.70, 0.20, 2.40]
    points = [np.array([p]) for p in points]
    points_1, points_2 = points[:4], points[4:]
    clusterer = IHAClusterer()
    clusterer.fit(points_1)
    hi = clusterer.hierarchy
    bad_nodes = Node.get_bad_nodes()

    clusterer.fit_more(points_2)
    bad_nodes = Node.get_bad_nodes()
    hi.visualize()
    # TODO: move to tests.py and add asserts
    # bad_nodes must be empty


def test_2_clusters_1_dimension():
    # create sample data
    points = create_2_1dim_clusters()
    # apply clustering
    clusterer = IHAClusterer()
    clusterer.fit(points)
    root = clusterer.hierarchy.root
    hi = clusterer.hierarchy
    second = root.children
    cluster_a = second[0].get_cluster_leaves()
    cluster_a = [n.center for n in cluster_a]
    cluster_b = second[1].get_cluster_leaves()
    cluster_b = [n.center for n in cluster_b]

    print("Found clusters:\n")
    print("A: ")
    print(cluster_a)
    print("\nB: ")
    print(cluster_b)

    hi.visualize()


def test_2level_clusters_1_dimension():
    # create sample data
    cluster_a1 = np.arange(0,0.4,0.01)
    cluster_a2 = np.arange(0.6,1,0.01)
    cluster_b1 = np.arange(2,2.4,0.01)
    cluster_b2 = np.arange(2.6,3,0.01)

    points = np.append(cluster_a1, cluster_a2)
    points = np.append(points, cluster_b1)
    points = np.append(points, cluster_b2)

    np.random.shuffle(points)
    points = [np.array([p]) for p in points]

    # apply clustering
    clusterer = IHAClusterer()
    clusterer.fit(points)
    hi = clusterer.hierarchy

    hi.visualize()


def test_3_points():
    # initializing with array([ 0.7]) and array([ 0.3])
    # processing array([ 0.6])
    points = [np.array(x) for x in [0.7, 0.3, 0.6]]
    clusterer = IHAClusterer()
    clusterer.fit(points)
    clusterer.hierarchy.visualize()


def test_3_clusters_2_dimensions():
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler

    dataset = datasets.make_blobs(n_samples=100, random_state=8)
    X, y = dataset
    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    ihac = IHAClusterer()
    ihac.fit(X)
    # ihac.hierarchy.visualize()

    labels = ihac.get_labels()
    print("Labels: ")
    print(labels)
    print("N clusters = %d" % len(set(labels)))
    y_pred = np.array(labels).astype(np.int)

    import matplotlib.pyplot as plt
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)
    plt.show()


def test_with_articles(datapath):
    N = 40
    articles, labels_true = load_articles(datapath)

    articles, labels_true = articles[:N], labels_true[:N]

    vecs_file = 'test_articles_%d.pickle' % N
    if not os.path.exists(vecs_file):
        vecs = build_vectors(articles, vecs_file)
    else:
        with open(vecs_file, 'rb') as f:
            vecs = pickle.load(f)


    ihac = IHAClusterer()
    vecs = vecs.toarray()
    vec_tags = [art.title[:50] for art in articles]
    ihac.fit(vecs, vec_tags)

    with open("ihac_article_hierarchy_%d.txt" % N, "w") as outfile:
        outfile.write(ihac.hierarchy.root.pretty_print())


if __name__ == '__main__':
    # test_3_clusters_2_dimensions()
    # test_3_points()
    datapath = "../eval/data/event/handpicked.json"
    test_with_articles(datapath)
