"""
    Implementation of Katz-CLASSIT

    a variant of COBWEB algorithm adapted for clustering
    of (vectorized) text articles, that uses
    the Katz distribution to model word frequency distributions

    Each attribute (feature) is assumed to be the number of
    occurrences of a particular word in the represented article
"""
import json
import scipy
from random import choice
from random import shuffle

class KatzClassitNode:
    counter = 0

    def __init__(self, otherTree=None):
        """
        The constructor creates a cobweb node with default values. It can also
        be used as a copy constructor to "deepcopy" a node.
        """
        self.concept_id = self.gensym()
        #self.concept_name = "Concept" + self.gensym()
        self.doc_count = 0
        self.av_counts = {}
        self._df = {}
        self._cf = {}
        self.children = []
        self.parent = None

        # check if the constructor is being used as a copy constructor
        if otherTree:
            self.update_counts_from_node(otherTree)
            self.parent = otherTree.parent

            for child in otherTree.children:
                self.children.append(self.__class__(child))

    def shallow_copy(self):
        """
        Creates a copy of the current node and its children 
        (but not their children)
        """
        temp = self.__class__()
        temp.update_counts_from_node(self)

        for child in self.children:
            temp_child = self.__class__()
            temp_child.update_counts_from_node(child)
            temp.children.append(temp_child)

        return temp

    #####
    # Category Utility, Probability and Counting
    #####
    # TODO: can we remove av_counts ?
    def increment_counts(self, instance):
        """
        Increment the collection and document frequency counts
        at the current node according to the specified instance.

        input:
            instance: {w1: v1, w2: v2, ...} - a hashtable of
            occurring words and frequency values. 
        """
        self.doc_count += 1 
        for word, fval in instance.items():
            self.av_counts.setdefault(word, {})
            self.av_counts[word].setdefault(fval, 0)
            self.av_counts[word][fval] += 1

            self._df.setdefault(word, 0)
            self._df[word] += 1

            self._cf.setdefault(word, 0)
            self._cf[word] += fval
    
    def update_counts_from_node(self, node):
        """
        Increments the counts of the current node
        by the amounts in the specified node.
        """
        self.doc_count += node.doc_count
        for word in node.av_counts:
            for fval in node.av_counts[word]:
                self.av_counts[word] = self.av_counts.setdefault(word,{})
                self.av_counts[word].setdefault(fval, 0)
                self.av_counts[word][fval] += node.av_counts[word][fval]

            self._df.setdefault(word, 0)
            self._df[word] += node._df[word]

            self._cf.setdefault(word, 0)
            self._cf[word] += node._cf[word]

    def category_utility(self):
        """
        Returns the category utility of a particular division of a concept
        into its children. 
        This is used as the heuristic to guide the concept formation.
        """
        k = len(self.children)
        if k == 0:
            return 0.0

        child_correct_guesses = 0.0
        for child in self.children:
            p_of_child = child.doc_count / (1.0 * self.doc_count)
            child_correct_guesses += p_of_child * child.cu_k()

        return (child_correct_guesses - self.cu_k()) * 1.0 / k
    
    def cu_k(self):
        """
        Returns the number of correct guesses that are expected from
        the given concept cluster

        This is the sum of contributions of each attribute for the
        current cluster node
        """
        correct_guesses = 0.0
        for word in self._df:
            correct_guesses += self.cu_ik(word)

        return correct_guesses

    def cu_ik(self, word):
        """
        Computes the contribution of attribute towards
        category utility of current node

        (CU_ik)
        """
        p0  = self.p0(word)
        pp  = self.p(word)
        return (1 - 2*p0 * (1 - p0) - pp * (1 - 2*p0)) / (1 + pp)

    def p0(self, word):
        """ Probability that the word does not occur in a document
            of the current cluster
        """
        return 1 - self.df(word)

    def p(self, word):
        """ Probabiblity that an occurrence of the word in a document
            of current cluster is a repeat 
        """
        return 1 - self.df(word) / self.cf(word)

    def df(self, word):
        """ Document Frequency = number of documents in the
            current cluster that contain the given word

            (i.e: number of documents having an attribute for 
            this word, as we assume absent frequency counts to
            represent 0 occurrences of a word)
        """
        return self._df[word]

    def cf(self, word):
        """ Collection Frequency = total number of occurences of
            the given word within the documents of the current cluster
            
            (i.e: the sum of frequency attributes among all instances)
        """
        return self._cf[word]


    #####
    # Control Structure methods
    #####
    def get_best_operation(self, instance, best1, best2,
        possible_ops=["best", "new", "merge", "split"]):
        """
        Given a set of possible operations, find the best and return its cu and
        the action name.
        """
        # If there is no best, then create a new child.
        if not best1:
            raise ValueError("Need at least one best child.")
            #return (self.cu_for_new_child(instance), 'new')

        if best1:
            best1_cu, best1 = best1
        if best2:
            best2_cu, best2 = best2
        operations = []

        if "best" in possible_ops:
            operations.append((best1_cu,"best"))
        if "new" in possible_ops: 
            operations.append((self.cu_for_new_child(instance),'new'))
        if "merge" in possible_ops and len(self.children) > 2 and best2:
            operations.append((self.cu_for_merge(best1, best2, instance),'merge'))
        if "split" in possible_ops and len(best1.children) > 0:
            operations.append((self.cu_for_split(best1),'split'))

        operations.sort(reverse=True)
        #print(operations[0])
        return operations[0]

    def two_best_children(self, instance):
        """
        Returns the two best children to incorporate the instance
        into in terms of category utility.

        input:
            instance: {a1: v1, a2: v2,...} - a hashtable of word. and values. 
        output:
            (0.2,2),(0.1,3) - the category utility and indices for the two best
            children (the second tuple will be None if there is only 1 child).
        """
        if len(self.children) == 0:
            raise Exception("No children!")

        children_cu = [(self.cu_for_insert(child, instance), i, child) for i,
                       child in enumerate(self.children)]
        children_cu.sort(reverse=True)

        if len(children_cu) == 0:
            return None, None
        if len(children_cu) == 1:
            return (children_cu[0][0], children_cu[0][2]), None 

        return ((children_cu[0][0], children_cu[0][2]), (children_cu[1][0],
                                                         children_cu[1][2]))

    def cu_for_insert(self, child, instance):
        """
        Computer the category utility of adding the instance to the specified
        child.
        """
        temp = self.__class__()
        temp.update_counts_from_node(self)
        temp.increment_counts(instance)

        for c in self.children:
            temp_child = self.__class__()
            temp_child.update_counts_from_node(c)
            temp.children.append(temp_child)
            if c == child:
                temp_child.increment_counts(instance)
        return temp.category_utility()

    def create_new_child(self, instance):
        """
        Creates a new child (to the current node) with the counts initialized by
        the given instance. 
        """
        new_child = self.__class__()
        new_child.parent = self
        new_child.increment_counts(instance)
        self.children.append(new_child)
        return new_child

    def create_child_with_current_counts(self):
        """
        Creates a new child (to the current node) with the counts initialized by
        the current node's counts.
        """
        if self.doc_count > 0:
            new = self.__class__(self)
            new.parent = self
            self.children.append(new)
            return new

    def cu_for_new_child(self, instance):
        """
        Returns the category utility for creating a new child using the
        particular instance.
        """
        temp = self.shallow_copy()
        temp.increment_counts(instance)
        temp.create_new_child(instance)
        return temp.category_utility()

    def merge(self, best1, best2):
        """
        Merge the two specified nodes.

        input:
            best1: the best child
            best2: the second best child
        output:
            The new child formed from the merge
        """
        new_child = self.__class__()
        new_child.parent = self
        new_child.update_counts_from_node(best1)
        new_child.update_counts_from_node(best2)
        best1.parent = new_child
        best2.parent = new_child
        new_child.children.append(best1)
        new_child.children.append(best2)
        self.children.remove(best1)
        self.children.remove(best2)
        self.children.append(new_child)

        return new_child

    def cu_for_merge(self, best1, best2, instance):
        """
        Returns the category utility for merging the two best children.

        input:
            best1: the best child in the children array.
            best2: the second best child in the children array.
        output:
            0.02 - the category utility for the merge of best1 and best2.
        """
        temp = self.__class__()
        temp.update_counts_from_node(self)
        temp.increment_counts(instance)

        new_child = self.__class__()
        new_child.update_counts_from_node(best1)
        new_child.update_counts_from_node(best2)
        new_child.increment_counts(instance)
        temp.children.append(new_child)

        for c in self.children:
            if c == best1 or c == best2:
                continue
            temp_child = self.__class__()
            temp_child.update_counts_from_node(c)
            temp.children.append(temp_child)

        return temp.category_utility()

    def split(self, best):
        """
        Split the best node and promote its children
        """
        self.children.remove(best)
        for child in best.children:
            child.parent = self
            self.children.append(child)

    def cu_for_fringe_split(self, instance):
        """
        Determine the category utility of performing a fringe split (i.e.,
        adding a leaf to a leaf). It turns out that this is useful for
        identifying unnecessary fringe splits, when the two leaves are
        essentially identical. It can be used to keep the tree from growing and
        to increase the tree's predictive accuracy.
        """
        temp = self.__class__()
        temp.update_counts_from_node(self)
        
        temp.create_child_with_current_counts()
        temp.increment_counts(instance)
        temp.create_new_child(instance)

        return temp.category_utility()

    def cu_for_split(self, best):
        """
        Return the category utility for splitting the best child.
        
        input:
            best1: a child in the children array.
        output:
            0.03 - the category utility for the split of best1.
        """
        temp = self.__class__()
        temp.update_counts_from_node(self)

        for c in self.children + best.children:
            if c == best:
                continue
            temp_child = self.__class__()
            temp_child.update_counts_from_node(c)
            temp.children.append(temp_child)

        return temp.category_utility()

    def __hash__(self):
        """
        The basic hash function. This hashes the concept name, which is
        generated to be unique across concepts.
        """
        return hash("KatzClassitNode" + str(self.concept_id))

    def gensym(self):
        """
        Generates a unique id and increments the class counter. This is used to
        create a unique name for every concept. 
        """
        self.__class__.counter += 1
        return str(self.__class__.counter)

    def __str__(self):
        """
        Call pretty print
        """
        return self.pretty_print()

    def pretty_print(self, depth=0):
        """
        Prints the categorization tree.
        """
        ret = str(('\t' * depth) + "|-" + str(self.av_counts) + ":" +
                  str(self.doc_count) + '\n')
        
        for c in self.children:
            ret += c.pretty_print(depth+1)

        return ret

    def depth(self):
        """
        Returns the depth of the current node in the tree.
        """
        if self.parent:
            return 1 + self.parent.depth()
        return 0

    def is_parent(self, other_concept):
        """
        Returns True if self is a parent of other concept.
        """
        temp = other_concept
        while temp != None:
            if temp == self:
                return True
            try:
                temp = temp.parent
            except:
                print(temp)
                assert False
        return False

    def num_concepts(self):
        """
        Return the number of concepts contained in the tree defined by the
        current node. 
        """
        children_count = 0
        for c in self.children:
           children_count += c.num_concepts() 
        return 1 + children_count 

    def get_probability(self, word, val):
        """
        Gets the probability of a particular attribute value at the given
        concept.

        """
        while word not in self.av_counts:
            if not self.parent:
                return 0.0
            self = self.parent

        if val not in self.av_counts[word]:
            return 0.0

        return (1.0 * self.av_counts[word][val]) / self.doc_count


class KatzClassitHierarchy:
    def __init__(self):
        """
        Initialize the tree with a KatzClassitNode
        """
        self.root = KatzClassitNode()

    def __str__(self):
        return str(self.root)

    def ifit(self, instance):
        """
        Given an instance incrementally update the categorization tree.
        """
        return self.cobweb(instance)

    def fit(self, vecs):
        """
        Call incremental fit on each element in a list of instances.
        """
        for i, instance in enumerate(vecs):
            self.ifit(instance)

    def cobweb(self, instance):
        """
        Incrementally integrates an instance into the categorization tree.
        This function operates iteratively to integrate this instance and uses
        category utility as the heuristic to make decisions.
        """
        current = self.root

        while current:
            if (not current.children and current.cu_for_fringe_split(instance)
                <= 0.0):
                current.increment_counts(instance)
                return current 

            elif not current.children:
                new = current.__class__(current)
                current.parent = new
                new.children.append(current)

                if new.parent:
                    new.parent.children.remove(current)
                    new.parent.children.append(new)
                else:
                    self.root = new

                new.increment_counts(instance)
                return new.create_new_child(instance)
            else:
                best1, best2 = current.two_best_children(instance)
                action_cu, best_action = current.get_best_operation(instance,
                                                                    best1,
                                                                    best2)
                if best1:
                    best1_cu, best1 = best1
                if best2:
                    best2_cu, best2 = best2

                if best_action == 'best':
                    current.increment_counts(instance)
                    current = best1
                elif best_action == 'new':
                    current.increment_counts(instance)
                    return current.create_new_child(instance)
                elif best_action == 'merge':
                    current.increment_counts(instance)
                    new_child = current.merge(best1, best2)
                    current = new_child
                elif best_action == 'split':
                    current.split(best1)
                else:
                    raise Exception("Should never get here.")

    def cobweb_categorize_leaf(self, instance):
        """
        Sorts an instance in the categorization tree defined at the current
        node without modifying the counts of the tree.

        This version always goes to a leaf.
        """
        current = self.root
        while current:
            if not current.children:
                return current
            
            best1, best2 = current.two_best_children(instance)

            if best1:
                best1_cu, best1 = best1
                current = best1
            else:
                return current

    def cobweb_categorize(self, instance):
        """
        Sorts an instance in the categorization tree defined at the current
        node without modifying the counts of the tree.

        Uses the new and best operations; when new is the best operation it
        returns the current node otherwise it iterates on the best node. 
        """
        current = self.root
        while current:
            if not current.children:
                return current

            best1, best2 = current.two_best_children(instance)
            action_cu, best_action = current.get_best_operation(instance,
                                                                 best1, best2,
                                                                 ["best",
                                                                  "new"]) 
            if best1:
                best1_cu, best1 = best1
            else:
                return current

            if best_action == "new":
                return current
            elif best_action == "best":
                current = best1

    def predict(self, instance):
        """
        Given an instance predict any missing attribute values without
        modifying the tree.
        """
        prediction = {}

        # make a copy of the instance
        for word in instance:
            prediction[word] = instance[word]

        concept = self.cobweb_categorize(instance)
        
        for word in concept.av_counts:
            if word in prediction:
                continue
            
            values = []
            for val in concept.av_counts[word]:
                values += [val] * concept.av_counts[word][val]

            prediction[word] = choice(values)

        return prediction

    def concept_attr_value(self, instance, word, val):
        """
        Gets the probability of a particular attribute value for the concept
        associated with a given instance.
        """
        concept = self.cobweb_categorize(instance)
        return concept.get_probability(word, val)

    def flexible_prediction(self, instance, guessing=False):
        """
        Fisher's flexible prediction task. It computes the accuracy of
        correctly predicting each attribute value (removing it from the
        instance first). It then returns the average accuracy. 
        """
        probs = []
        for word in instance:
            temp = {}
            for word2 in instance:
                if word == word2:
                    continue
                temp[word2] = instance[word2]
            if guessing:
                probs.append(self.get_probability(word, instance[word]))
            else:
                probs.append(self.concept_word_value(temp, word, instance[word]))
        return sum(probs) / len(probs)


    def cluster(self, instances, depth=1):
        """
        Used to cluster examples incrementally and return the cluster labels.
        The final cluster labels are at a depth of 'depth' from the root. This
        defaults to 1, which takes the first split, but it might need to be 2
        or greater in cases where more distinction is needed.
        """
        temp_clusters = [self.ifit(instance) for instance in instances]

        print(len(set([c.concept_id for c in temp_clusters])))
        clusters = []
        for i,c in enumerate(temp_clusters):
            while (c.parent and c not in c.parent.children):
                c = c.parent

            promote = True
            while c.parent and promote:
                n = c
                for i in range(depth+2):
                    if not n:
                        promote = False
                        break
                    n = n.parent

                if promote:
                    c = c.parent

            clusters.append("Concept" + c.concept_id)

        with open('visualize/output.json', 'w') as f:
            f.write(json.dumps(self.root.output_json()))

        return clusters

    def baseline_guesser(self, filename, length, iterations):
        """
        Equivalent of predictions, but just makes predictions from the root of
        the concept tree. This is the equivalent of guessing the distribution
        of all attribute values. 
        """
        n = iterations
        runs = []
        nodes = []

        for i in range(0,n):
            print("run %i" % i)
            t = self.__class__()
            accuracy, num = t.sequential_prediction(filename, length, True)
            runs.append(accuracy)
            nodes.append(num)
            #print(json.dumps(t.output_json()))

        #print(runs)
        print("MEAN Accuracy")
        for i in range(0,len(runs[0])):
            a = []
            for r in runs:
                a.append(r[i])
            print("%0.2f" % (scipy.mean(a)))

        print()
        print("STD Accuracy")
        for i in range(0,len(runs[0])):
            a = []
            for r in runs:
                a.append(r[i])
            print("%0.2f" % (scipy.std(a)))

        print()
        print("MEAN Concepts")
        for i in range(0,len(runs[0])):
            a = []
            for r in nodes:
                a.append(r[i])
            print("%0.2f" % (scipy.mean(a)))

        print()
        print("STD Concepts")
        for i in range(0,len(runs[0])):
            a = []
            for r in nodes:
                a.append(r[i])
            print("%0.2f" % (scipy.std(a)))

    def predictions(self, filename, length, iterations):
        """
        Perform the sequential prediction task many times and compute the mean
        and std of all flexible predictions.
        """
        n = iterations 
        runs = []
        nodes = []
        for i in range(0,n):
            print("run %i" % i)
            t = self.__class__()
            accuracy, num = t.sequential_prediction(filename, length)
            runs.append(accuracy)
            nodes.append(num)
            #print(json.dumps(t.output_json()))

        #print(runs)
        print("MEAN Accuracy")
        for i in range(0,len(runs[0])):
            a = []
            for r in runs:
                a.append(r[i])
            print("%0.2f" % (scipy.mean(a)))

        print()
        print("STD Accuracy")
        for i in range(0,len(runs[0])):
            a = []
            for r in runs:
                a.append(r[i])
            print("%0.2f" % (scipy.std(a)))

        print()
        print("MEAN Concepts")
        for i in range(0,len(runs[0])):
            a = []
            for r in nodes:
                a.append(r[i])
            print("%0.2f" % (scipy.mean(a)))

        print()
        print("STD Concepts")
        for i in range(0,len(runs[0])):
            a = []
            for r in nodes:
                a.append(r[i])
            print("%0.2f" % (scipy.std(a)))


class KatzClassitClusterer(object):
    def __init__(self):
        pass

    def save(self, prefix):
        with open(prefix + 'hierarchy.dump', 'wb') as f:
            data = {
                # 'Node': {
                #     'size': Node.size,
                #     'max_n_nodes': Node.max_n_nodes,
                #     'available_ids': Node.available_ids,
                #     'nodes': Node.nodes,
                #     'distances': Node.distances,
                # },
                # 'self': {
                #     'hierarchy': self.hierarchy,
                #     'vecs': self.vecs,
                #     'size': self.size                
                # }
            }    
            pickle.dump(data, f)

    def load(self, prefix):
        with open(prefix + 'hierarchy.dump', 'rb') as f:    
            data = pickle.load(f)
            # for name, val in data["Node"].items():
            #     setattr(Node, name, val)
            # for name, val in data["self"].items():
            #     setattr(self, name, val)

    def fit(self, vecs):
        self.size = len(vecs)
        Node.init(self.size)
        self.vecs = vecs
        # print("initializing with %s and %s" % (repr(vecs[0]), repr(vecs[1])))
        self.hierarchy = Hierarchy(len(vecs), vecs[0], vecs[1])

        for vec in self.vecs[2:]:
            # print("processing " + repr(vec))
            self.hierarchy.incorporate(vec)
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



def sparse_matrix_to_array_of_dicts(matrix):
    pass

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


    hierarchy = KatzClassitHierarchy()
    vecs = vecs.toarray()

    # vecs is a sparse matrix
    # where each row contains the non-zero attributes of an article
    # we must turn it into an array of dict representations

    artdicts = sparse_matrix_to_array_of_dicts(vecs)

    vec_tags = [art.title[:50] for art in articles]
    
    hierarchy.fit(vecs, vec_tags)

    with open("katzclassit_article_hierarchy_%d.txt" % N, "w") as outfile:
        outfile.write(ihac.hierarchy.root.pretty_print())




if __name__ == "__main__":
    #KatzClassit().predictions("data_files/cobweb_test.json", 10, 100)
    #KatzClassit().predictions("data_files/mushrooms.json", 30, 10)
    #KatzClassit().baseline_guesser("data_files/cobweb_test.json", 10, 100)
    #print(KatzClassit().cluster("cobweb_test.json", 10, 1))

    #t = KatzClassit()
    #print(t.sequential_prediction("cobweb_test.json", 10))
    #t.verify_counts()

    #test = {}
    #print(t.predict(test))

    tree = KatzClassitHierarchy()
    tree.ifit({'a': 2, 'b': 3})
    tree.ifit({'a': 5, 'c': 2})
    tree.ifit({'b': 1})
    tree.ifit({'c': 2})
    tree.ifit({'a': 3})
    tree.ifit({'a': 2})
    print(tree)
