"""
    Implementation of Katz-CLASSIT

    a variant of COBWEB algorithm adapted for clustering
    of (vectorized) text articles, that uses
    the Katz distribution to model word frequency distributions
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
        self.count = 0
        self.av_counts = {}
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

    def increment_counts(self, instance):
        """
        Increment the counts at the current node according to the specified
        instance.

        input:
            instance: {a1: v1, a2: v2, ...} - a hashtable of attr and values. 
        """
        self.count += 1 
        for attr in instance:
            self.av_counts[attr] = self.av_counts.setdefault(attr,{})
            self.av_counts[attr][instance[attr]] = (self.av_counts[attr].get(
                instance[attr], 0) + 1)
    
    def update_counts_from_node(self, node):
        """
        Increments the counts of the current node by the amount in the specified
        node.
        """
        self.count += node.count
        for attr in node.av_counts:
            for val in node.av_counts[attr]:
                self.av_counts[attr] = self.av_counts.setdefault(attr,{})
                self.av_counts[attr][val] = (self.av_counts[attr].get(val,0) +
                                     node.av_counts[attr][val])

    def expected_correct_guesses(self):
        """
        Returns the number of correct guesses that are expected from the given
        concept. This is the sum of the probability of each attribute value
        squared. 
        """

        # replace with 
        #     def contribution_utility(i, k):
        # """The contribution of the attribute i towards 
        # the Category Utility of the cluster k.
        

        correct_guesses = 0.0

        for attr in self.av_counts:
            if attr[0] == "_":
                continue
            for val in self.av_counts[attr]:
                prob = (self.av_counts[attr][val] / (1.0 * self.count))
                correct_guesses += (prob * prob)

        return correct_guesses

    def category_utility(self):
        """
        Returns the category utility of a particular division of a concept into
        its children. This is used as the heuristic to guide the concept
        formation.
        """
        # TODO: reimplement to use formula based on Katz distribution of 
        # word frequency attributes


        if len(self.children) == 0:
            return 0.0

        child_correct_guesses = 0.0

        for child in self.children:
            p_of_child = child.count / (1.0 * self.count)
            child_correct_guesses += p_of_child * child.expected_correct_guesses()

        return ((child_correct_guesses - self.expected_correct_guesses()) /
                (1.0 * len(self.children)))

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
            instance: {a1: v1, a2: v2,...} - a hashtable of attr. and values. 
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
        if self.count > 0:
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
                  str(self.count) + '\n')
        
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

    def output_json(self):
        """
        Outputs the categorization tree in JSON form so that it can be
        displayed, I usually visualize it with d3js in a web browser.
        """
        output = {}
        output['name'] = "Concept" + self.concept_id
        output['size'] = self.count
        output['children'] = []

        temp = {}
        for attr in self.av_counts:
            for value in self.av_counts[attr]:
                temp[attr + " = " + str(value)] = self.av_counts[attr][value]

        for child in self.children:
            output['children'].append(child.output_json())

        output['counts'] = temp

        return output

    def get_probability(self, attr, val):
        """
        Gets the probability of a particular attribute value at the given
        concept.

        """
        while attr not in self.av_counts:
            if not self.parent:
                return 0.0
            self = self.parent

        if val not in self.av_counts[attr]:
            return 0.0

        return (1.0 * self.av_counts[attr][val]) / self.count




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
        for attr in instance:
            prediction[attr] = instance[attr]

        concept = self.cobweb_categorize(instance)
        
        for attr in concept.av_counts:
            if attr in prediction:
                continue
            
            values = []
            for val in concept.av_counts[attr]:
                values += [val] * concept.av_counts[attr][val]

            prediction[attr] = choice(values)

        return prediction

    def concept_attr_value(self, instance, attr, val):
        """
        Gets the probability of a particular attribute value for the concept
        associated with a given instance.
        """
        concept = self.cobweb_categorize(instance)
        return concept.get_probability(attr, val)

    def flexible_prediction(self, instance, guessing=False):
        """
        Fisher's flexible prediction task. It computes the accuracy of
        correctly predicting each attribute value (removing it from the
        instance first). It then returns the average accuracy. 
        """
        probs = []
        for attr in instance:
            temp = {}
            for attr2 in instance:
                if attr == attr2:
                    continue
                temp[attr2] = instance[attr2]
            if guessing:
                probs.append(self.get_probability(attr, instance[attr]))
            else:
                probs.append(self.concept_attr_value(temp, attr, instance[attr]))
        return sum(probs) / len(probs)

    def train_from_json(self, filename, length=None):
        """
        Build the concept tree from a set of examples in a provided json file.
        """
        json_data = open(filename, "r")
        instances = json.load(json_data)
        if length:
            shuffle(instances)
            instances = instances[:length]
        self.fit(instances)
        json_data.close()

    def sequential_prediction(self, filename, length, guessing=False):
        """
        Given a json file, perform an incremental sequential prediction task. 
        Try to flexibly predict each instance before incorporating it into the 
        tree. This will give a type of cross validated result.
        """
        json_data = open(filename, "r")
        instances = json.load(json_data)
        #shuffle(instances)
        #instances = instances[0:length]

        accuracy = []
        nodes = []
        for j in range(1):
            shuffle(instances)
            for n, i in enumerate(instances):
                if n >= length:
                    break
                accuracy.append(self.flexible_prediction(i, guessing))
                nodes.append(self.num_concepts())
                self.ifit(i)
        json_data.close()
        return accuracy, nodes

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
    tree.ifit({'a': 'v', 'b': 'v'})
    tree.ifit({'a': 'v2'})
    tree.ifit({'a': 'v'})
    tree.ifit({'a': 'v'})
    tree.ifit({'a': 'v'})
    tree.ifit({'a': 'v'})
    print(tree)















##################

# First draft of Katz-based category utility function

# TODO: figure out how to insert it in the above structure

# TODO2: figure out if collection frequency is calculated relative to
# * documents under current
# * documents seen so far
# * an external corpus



    N = 6   #number of documents, we must calculate it from the previosly loaded
            #data-structure 

    def p0(word, doc_collection):
        """Probability that the word does not occur in a document.
        """
        return 1 - df(word,doc_collection)/N
        
    def p(word, doc_collection):
        """Probabiblity that the occurrence of the word is a repeat 
        in a document.
        """
        return 1 - df(word,doc_collection)/cf(word,doc_collection)

    def contribution_utility(i, k):
        """The contribution of the attribute i towards 
        the Category Utility of the cluster k.
        """
        p0  = p0(i,k)
        pp  = p(i,k)
        return (1 - 2*p0*(1 - p0) - pp*(1 - 2*p0))/(1 + pp)

    def cf(i, doc_collection):
        """Collection Frecuency = number of times word i occurred 
           in the document collection.
        """
        count = 0
        for k in doc_collection.elements:
            count += k.word_count(i)        #only need to take the attribute value
        return count

    def df(i, doc_collection):
        """Document Frequency = number of documents in the
        entire collection that contain the word i.
        """
        count = 0
        for k in doc_collection.elements:
            if k.find_word(i):
                count += 1
        return count

    def category_utility(cluster):
        cluster_partition = cluster.children
        K = len(cluster_partition)
        CUp = 0      #category_utility result 
        for k in K:  # looping on child clusters 
            CUik = 0 # contribution_utility of attribute i to cluster k
            CUip = 0 # contribution_utility of attribute i to parent_cluster
            for i in cluster_partition[k].attributes():
                CUik += contribution_utility(i,k)
                CUip += contribution_utility(i,cluster)
            CUp += P(cluster_partition[k])*(CUik - CUip)    
        return CUp  
