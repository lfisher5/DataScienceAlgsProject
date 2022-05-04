import operator
import utils.myutils as myutils
import math
import random
import numpy as np


class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.
    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.
    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = []
        self.header = None
        self.attribute_domains = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.
        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        self.X_train = X_train
        self.y_train = y_train
        # next, make a copy of your header... tdidt() is going
        # to modify the list
        header = []
        for i in range(len(X_train[0])):
            header.append('att' + str(i))
        self.header = header
        available_attributes = header.copy()

        self.generate_att_domain()

        self.tree = self.tdidt(train, available_attributes)

    def select_attribute(self, instances, attributes):
        # TODO: use entropy to compute and choose the attribute
        # with the smallest Enew
        # for now, we will just choose randomly
        entropies = []

        att_idxs = [self.header.index(att) for att in attributes]

        for i in att_idxs:
            att_val_entropies = []
            att_groups = myutils.group_by_determ(
                instances, i, self.attribute_domains[self.header[i]])
            for j in range(len(att_groups)):
                tot_inst_for_att_val = len(att_groups[j])
                _, label_att_groups = myutils.group_by_values(
                    att_groups[j], -1)
                log_nums = []
                for k in range(len(label_att_groups)):
                    log_nums.append(len(label_att_groups[k]))
                log_denom = sum(log_nums)
                e_att = 0
                for k in range(len(log_nums)):
                    if log_nums[k] != 0 and log_denom != 0:
                        e_att += -(log_nums[k] / log_denom) * \
                            math.log(log_nums[k] / log_denom, 2)
                att_val_entropies.append(
                    e_att * tot_inst_for_att_val / len(instances))
            entropies.append(sum(att_val_entropies))

        return attributes[entropies.index(min(entropies))]

    def partition_instances(self, instances, split_attribute):
        # lets use a dictionary
        partitions = {}  # key (string): value (subtable)
        att_index = self.header.index(split_attribute)  # e.g. 0 for level
        # e.g. ["Junior", "Mid", "Senior"]
        att_domain = self.attribute_domains['att' + str(att_index)]
        for att_value in att_domain:
            partitions[att_value] = []
            # task: finish
            for instance in instances:
                if instance[att_index] == att_value:
                    partitions[att_value].append(instance)

        return partitions

    def tdidt(self, current_instances, available_attributes):
        # basic approach (uses recursion!!):
        # print("available_attributes:", available_attributes)

        # select an attribute to split on
        case_3 = False
        attribute = self.select_attribute(
            current_instances, available_attributes)
        available_attributes.remove(attribute)
        tree = ["Attribute", attribute]
        # group data by attribute domains (creates pairwise disjoint partitions)
        partitions = self.partition_instances(current_instances, attribute)
        # print('partitions', partitions)
        # for each partition, repeat unless one of the following occurs (base case)
        for att_value, att_partition in partitions.items():
            # print("curent attribute value:", att_value, len(att_partition))
            value_subtree = ["Value", att_value]

            #    CASE 1: all class labels of the partition are the same => make a leaf node
            if len(att_partition) > 0 and self.check_all_same_class(att_partition):
                leaf_node = ['Leaf', att_partition[0][-1],
                             len(att_partition), len(current_instances)]

                value_subtree.append(leaf_node)

            #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
            elif len(att_partition) > 0 and len(available_attributes) == 0:
                labels, freqs = myutils.get_frequencies(att_partition, -1)
                leaf_label = labels[freqs.index(max(freqs))]
                leaf_node = ['Leaf', leaf_label,
                             len(att_partition), len(current_instances)]
                # stats = self.compute_partition_stats(att_partition)
                value_subtree.append(leaf_node)
                # TODO: we have a mix of labels, handle clash with majority
                # vote leaf node

            #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
            elif len(att_partition) == 0:
                case_3 = True
                # TODO: "backtrack" to replace the attribute node
                # with a majority vote leaf node

            else:  # the previous conditions are all false... recurse!!
                subtree = self.tdidt(
                    att_partition, available_attributes.copy())

                value_subtree.append(subtree)
            if not case_3:
                tree.append(value_subtree)
            else:
                labels, freqs = myutils.get_frequencies(current_instances, -1)
                leaf_label = labels[freqs.index(max(freqs))]
                leaf_node = ['Leaf', leaf_label,
                             len(current_instances), len(current_instances)]
                tree = leaf_node
                # TODO: append subtree to value_subtree and to tree
                # appropriately
        return tree

    def generate_att_domain(self):
        attribute_domain = {}
        for i in range(len(self.X_train[0])):
            col_avail_vals = sorted(
                list(set([val[i] for val in self.X_train])))
            attribute_domain[self.header[i]] = col_avail_vals

        self.attribute_domains = attribute_domain

    def check_all_same_class(self, instances):
        # True if all instances have same label
        # Helpful for base case #1 (all class labels are the same... make a leaf node)
        all_same = True
        for instance in instances:
            if instance[-1] != instances[0][-1]:
                all_same = False
        return all_same

    def compute_partition_stats(self, instances):

        labels, subtables = myutils.group_by_values(instances, -1)
        stats = []
        for i in range(len(labels)):
            stats.append(labels[i], len(subtables[i]), len(instances))
        print('stats:', stats)

        return stats

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_pred = []
        for instance in X_test:
            pred = self.tdidt_predict(self.header, self.tree, instance)
            y_pred.append(pred)

        return y_pred  # TODO: fix this

    def tdidt_predict(self, header, tree, instance):
        # recursively traverse the tree
        # we need to know where we are in the tree...
        # are we at a leaf node (base case) or
        # attribute node
        info_type = tree[0]
        if info_type == "Leaf":
            return tree[1]
        # we need to match the attribute's value in the
        # instance with the appropriate value list
        # in the tree
        # a for loop that traverses through
        # each value list
        # recurse on match with instance's value
        att_index = header.index(tree[1])

        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance[att_index]:
                # we have a match, recurse
                return self.tdidt_predict(header, value_list[2], instance)

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.
        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        if attribute_names is None:
            atts = self.header

        print(self.print_tdidt_rules(self.tree, atts, class_name))

    def print_tdidt_rules(self, tree, header, class_label, rule="IF ", depth=0):

        if tree[0] == 'Leaf':
            # print('Leaf node:', tree[1])
            rule = rule[:len(rule) - 5]
            rule += ' THEN ' + class_label + ' = ' + tree[1]
            print(rule)
            return
        elif tree[0] == 'Attribute':
            # print('Attribute:', tree[1])
            rule += tree[1] + ' == '
        elif tree[0] == 'Value':
            # print('Value:', tree[1])
            rule += tree[1] + ' AND '

        for i in range(2, len(tree)):
            self.print_tdidt_rules(tree[i], header, class_label, rule)

    def recurse_tree_viz(self, subtree, tree_data, parent=None):

        if subtree[0] == 'Leaf':
            # print('Leaf node:', subtree[1])
            tree_data += ' [label=' + subtree[1] + '];\n'

            return
        elif subtree[0] == 'Attribute':
            # print('Attribute:', subtree[1])
            parent = subtree[1]
            tree_data += subtree[1] + ' [shape=box];\n'

        elif subtree[0] == 'Value':
            # print('Value:', subtree[1])
            tree_data += parent + ' -- ' + subtree[1]

        for i in range(2, len(subtree)):
            self.recurse_tree_viz(subtree[i], tree_data, parent=subtree[1])

    # BONUS method

    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).
        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        atts = []
        if attribute_names is not None:
            atts = attribute_names

        dot_data = 'graph g {'
        tree_data = '\n'
        self.recurse_tree_viz(self.tree, tree_data)
        dot_data += tree_data + '}'
        #cmd = 'dot -Tsvg -o' + pdf_fname + dot_fname
        # os.system(cmd)

        pass  # TODO: (BONUS) fix this


'''##############################################
# Programmer: Lauren Fisher
# Class: CptS 322-01, Spring 2022
# Programming Assignment #6
# 3/30/22
#
# Description: This program uses lists to replicate existing sklearn functions for different classifiers
##############################################'''


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.
    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """

    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.
        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        neighbors = []
        for i, x in enumerate(self.X_train):
            neighbors.append(
                [i, myutils.compute_euclidean_distance(x, X_test)])

        neighbors.sort(key=operator.itemgetter(-1))

        k_neigh = neighbors[:self.n_neighbors]
        indxs = myutils.get_column(k_neigh, 0)
        dists = myutils.get_column(k_neigh, 1)

        return dists, indxs

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        row_distances = []
        y_predicted = []
        for test_instance in X_test:
            for i, train_instance in enumerate(self.X_train):
                d = myutils.compute_euclidean_distance(
                    train_instance, test_instance)
                row_distances.append([d, i])

            row_distances.sort(key=operator.itemgetter(0))
            top_k_instances = self.get_top_k_instances(row_distances)
            prediction = self.select_class_label(top_k_instances)
            y_predicted.append(prediction)

        return y_predicted

    def get_top_k_instances(self, dists):
        k_dists = dists[:self.n_neighbors]
        return k_dists

    def select_class_label(self, instances):
        labels = []
        for instance in instances:
            labels.append(self.y_train[instance[1]])

        label = max(labels, key=labels.count)
        return label


class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.
    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()
    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """

    def __init__(self):
        """Initializer for DummyClassifier.
        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        self.most_common_label = max(y_train, key=y_train.count)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for _ in X_test:
            y_predicted.append(self.most_common_label)
        return y_predicted


# class MyNaiveBayesClassifier: # NOTE: Lauren's code commented out
#     """Represents a Naive Bayes classifier.
#     Attributes:
#         priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
#             label in the training set.
#         posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
#             attribute value/label pair in the training set.
#     Notes:
#         Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
#         You may add additional instance attributes if you would like, just be sure to update this docstring
#         Terminology: instance = sample = row and attribute = feature = column
#     """

#     def __init__(self):
#         """Initializer for MyNaiveBayesClassifier.
#         """
#         self.priors = None
#         self.sorted_labels = None
#         self.posteriors = None
#         self.num_instances = None

#     def fit(self, X_train, y_train):
#         """Fits a Naive Bayes classifier to X_train and y_train.
#         Args:
#             X_train(list of list of obj): The list of training instances (samples)
#                 The shape of X_train is (n_train_samples, n_features)
#             y_train(list of obj): The target y values (parallel to X_train)
#                 The shape of y_train is n_train_samples
#         Notes:
#             Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
#                 and the posterior probabilities for the training data.
#             You are free to choose the most appropriate data structures for storing the priors
#                 and posteriors.
#         """
#         # compute priors
#         self.num_instances = len(y_train)
#         labels, nums = myutils.get_categorical_frequencies(y_train, 0, False)
#         self.sorted_labels = labels
#         self.priors = {key: value for key, value in zip(labels, nums)}

#         # print(self.priors)

#         idxs = list(range(len(X_train)))
#         subtables = myutils.group_by(idxs, y_train)
#         possible_vals = []
#         for col_idx in range(len(X_train[0])):
#             possible_vals.append(
#                 list(sorted(set([row[col_idx] for row in X_train]))))

#         label_base = 'att'
#         posteriors = dict()

#         # iterate over possible targets from subtables
#         # create copy of table for for row idxs in subtables subsubtables[0]
#         # call get_frequencies for each column for row idxs in subtables subsubtables[0]
#         # if a label does not appear from get_freqs result, insert 0 in the correct index

#         for i in range(len(labels)):
#             row_idxs = [subtables[i][j][0] for j in range(len(subtables[i]))]

#             table_copy = [X_train[k] for k in row_idxs]
#             for col_idx in range(len(X_train[0])):
#                 vals, freqs = myutils.get_categorical_frequencies(
#                     table_copy, col_idx)

#                 if len(vals) is not len(possible_vals[col_idx]):
#                     #print('frequency problem: ', possible_vals[col_idx])
#                     new_freqs = [0] * len(possible_vals[col_idx])
#                     for k, val in enumerate(vals):
#                         new_freqs.insert(
#                             possible_vals[col_idx].index(val), freqs[k])

#                     freqs = new_freqs
#                     vals = possible_vals[col_idx]

#                 atts = [label_base + str(col_idx) + '=' + str(vals[j])
#                         for j in range(len(vals))]
#                 for att_idx in range(len(atts)):
#                     if atts[att_idx] in posteriors:
#                         posteriors[atts[att_idx]].append(freqs[att_idx])
#                     else:
#                         posteriors[atts[att_idx]] = [freqs[att_idx]]

#         self.posteriors = posteriors
#         # print(posteriors)

#     def predict(self, X_test):
#         """Makes predictions for test instances in X_test.
#         Args:
#             X_test(list of list of obj): The list of testing samples
#                 The shape of X_test is (n_test_samples, n_features)
#         Returns:
#             y_predicted(list of obj): The predicted target y values (parallel to X_test)
#         """
#         #print('X_test', X_test)
#         y_predicted = []
#         for test_sample in X_test:
#             test_post = []
#             #print('test_post', test_post)
#             for col_idx in range(len(test_sample)):
#                 label = 'att' + str(col_idx) + '=' + str(test_sample[col_idx])

#                 posts = self.posteriors[label]
#                 att_post = []
#                 for label_idx in range(len(posts)):
#                     att_post.append((posts[label_idx] /
#                                      self.priors[self.sorted_labels[label_idx]]))
#                 test_post.append(att_post)

#             result = [1] * len(self.sorted_labels)
#             for col_idx in range(len(test_post)):
#                 for label_idx in range(len(self.sorted_labels)):
#                     result[label_idx] *= test_post[col_idx][label_idx]

#             for i in range(len(result)):
#                 result[i] *= (self.priors[self.sorted_labels[label_idx]
#                                           ] / self.num_instances)

#             #print('result', result)
#             pred = self.sorted_labels[result.index(max(result))]
#             y_predicted.append(pred)

#         return y_predicted

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None
        self.att_values = None

    def y_labels(self, y_train):
        """Obtains unique labels y_train and sorts them

        Args:
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        
        Returns:
            y_labels_sorted(list): sorted list of unique identifiers for class labels
        """
        y_labels = []
        for i in range(len(y_train)):
            if y_train[i] not in y_labels:
                y_labels.append(y_train[i])
        
        y_labels_sorted = sorted(y_labels)

        return y_labels_sorted

    def stitched_datasets(self, X_train, y_train):
        """Stitches X_train and y_train together

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        
        Returns:
            stitched_X_y(list of lists): 2D list of mixed data types
        """
        # Create a list of possible attribute values using stitched list 
        stitched_X_y = []
        X_vals = []
        for i in range(len(X_train)):
            X_vals = []
            for j in range(len(X_train[0])):
                X_vals.append(X_train[i][j])
            X_vals.append(y_train[i])
            stitched_X_y.append(X_vals)
        
        return stitched_X_y

    def attributes_and_values(self, X_train):
        """For each column, this method determines the unique attribute values and places them in a 2D list. 

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
        
        Returns:
            att_values(list of lists): 2D list of mixed data types
        """
        att_values = []
        values = []
        values_sorted = []
        for j in range(len(X_train[0])):
            values = []
            values_sorted = []
            for i in range(len(X_train)):
                if X_train[i][j] not in values:
                    values.append(X_train[i][j])
            values_sorted = sorted(values)
            att_values.append(values_sorted) # each element refers to an attribute/column
        
        return att_values

    def convert_idxs_to_class_labels(self, y_train, y_pred):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
            y_pred(list of obj): The predicted y values (parallel to X_train)
                The shape of y_pred is n_train_samples
        
        Returns:
            y_pred_class(list of lists): 2D list of mixed data types
        """
        y_pred_class = y_pred.copy()
        y_labels_sorted = self.y_labels(y_train)

        for i in range(len(y_pred_class)):
            for n in range((len(y_labels_sorted))):
                if y_pred_class[i] == y_labels_sorted.index(y_labels_sorted[n]):
                    y_pred_class[i] = y_labels_sorted[n]

        return y_pred_class

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        y_labels_sorted = self.y_labels(y_train)

        self.priors = [] 
        for i in range(len(y_labels_sorted)): 
            count = 0
            for j in range(len(y_train)):
                if y_labels_sorted[i] == y_train[j]:
                    count += 1 
            self.priors.append(count / len(y_train))
        
        self.att_values = []
        stitched_X_y = self.stitched_datasets(X_train, y_train)
        self.att_values = self.attributes_and_values(X_train)
        
        # Calculating posteriors (most disgusting piece of code I've ever written)
        self.posteriors = []
        att_posterior = []
        # Must create a count of all the attribute values along with their classification 
        for m in range(len(y_labels_sorted)):
            for j in range(len(stitched_X_y[0]) - 1): # -1 to elimante the possibility to index to the class label, shares same size as att_values 
                att_posterior = []
                for k in range(len(self.att_values[j])): # j as len will change based on the number of attribute values  
                    count = 0
                    for i in range(len(stitched_X_y)):
                        if stitched_X_y[i][j] == self.att_values[j][k] and stitched_X_y[i][-1] == y_labels_sorted[m]:
                            count += 1
                            # print("att_values:", att_values[j][k], "--- stitched", stitched_X_y[i][j], "--- class:", y_labels_sorted[m], "--- count:", count) # used for debugging 
                    att_posterior.append((count / len(y_train) / self.priors[m])) # provides floats 
                    # att_posterior.append(count) # interger numbers (easier to debug)
                self.posteriors.append(att_posterior) # in the form of [[att1_values count for "no"], [att2_values count for "no"]
        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        num_labels = len(self.priors)
        posteriors_copy = self.posteriors.copy() # copy as to not change posteriors 
        pred_product = []
        post = []

        # Different approach -- take slices of posteriors shell for yes and no classifications, then index through values to match att_values to posteriors
        for n in range(num_labels):
            posteriors_slice = []
            posteriors_slice = posteriors_copy[n * len(self.att_values) : (n + 1) * len(self.att_values)] # slice contains the shape of att_values essentially mapping posteriors to att_values
            for i in range(len(X_test)): # number of test cases 
                post = []
                for j in range(len(X_test[0])): # number of attributes 
                    for k in range(len(self.att_values[j])):
                        if X_test[i][j] == self.att_values[j][k]:
                            # print(X_test[i], X_test[i][j], att_values[j][k], " --- ", i, j) # Used for debugging 
                            post.append(posteriors_slice[j][k])
                pred_product.append(np.prod(post))

        # Creates slices of predictions based on class label
        label_slices = int(len(pred_product) / num_labels)
        pred_slices = [] 

        for n in range(num_labels):
            pred_slices.append(pred_product[label_slices * n : (n + 1) * label_slices])

        # Multiplies priors into posteriors
        for i in range(len(pred_slices)):
            for j in range(len(pred_slices[0])):
                pred_slices[i][j] = pred_slices[i][j] * self.priors[i]
        
        # Matches prediction values of class to X_test element
        pred_temp = [] 
        pred_matched = [] 

        for j in range(len(pred_slices[0])):
            pred_temp = []
            for i in range(len(pred_slices)): 
                pred_temp.append(pred_slices[i][j])
            pred_matched.append(pred_temp)
            
        # Predicts the index for the class label
        pred_idxs = []

        for i in range(len(pred_matched)):
            pred_idxs.append(pred_matched[i].index(max(pred_matched[i])))

        return pred_idxs


class MyModifiedDecisionTreeClassifier:
    """Represents a decision tree classifier.
    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.
    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self):
        """Initializer for MyModifiedDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = []
        self.F = 0
        self.header = None
        self.attribute_domains = None

    def fit(self, X_train, y_train, F):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.
        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        self.X_train = X_train
        self.y_train = y_train
        self.F = F
        # next, make a copy of your header... tdidt() is going
        # to modify the list
        header = []
        for i in range(len(X_train[0])):
            header.append('att' + str(i))
        self.header = header
        available_attributes = header.copy()

        self.generate_att_domain()

        self.tree = self.tdidt(train, available_attributes)

    def select_attribute(self, instances, attributes):
        # TODO: use entropy to compute and choose the attribute
        # with the smallest Enew
        # for now, we will just choose randomly
        entropies = []
        selected_attributes = self.random_attribute_subset(attributes)
        att_idxs = [self.header.index(att) for att in selected_attributes]

        for i in att_idxs:
            att_val_entropies = []
            att_groups = myutils.group_by_determ(
                instances, i, self.attribute_domains[self.header[i]])
            for j in range(len(att_groups)):
                tot_inst_for_att_val = len(att_groups[j])
                _, label_att_groups = myutils.group_by_values(
                    att_groups[j], -1)
                log_nums = []
                for k in range(len(label_att_groups)):
                    log_nums.append(len(label_att_groups[k]))
                log_denom = sum(log_nums)
                e_att = 0
                for k in range(len(log_nums)):
                    if log_nums[k] != 0 and log_denom != 0:
                        e_att += -(log_nums[k] / log_denom) * \
                            math.log(log_nums[k] / log_denom, 2)
                att_val_entropies.append(
                    e_att * tot_inst_for_att_val / len(instances))
            entropies.append(sum(att_val_entropies))

        return attributes[entropies.index(min(entropies))]

    def partition_instances(self, instances, split_attribute):
        # lets use a dictionary
        partitions = {}  # key (string): value (subtable)
        att_index = self.header.index(split_attribute)  # e.g. 0 for level
        # e.g. ["Junior", "Mid", "Senior"]
        att_domain = self.attribute_domains['att' + str(att_index)]
        for att_value in att_domain:
            partitions[att_value] = []
            # task: finish
            for instance in instances:
                if instance[att_index] == att_value:
                    partitions[att_value].append(instance)

        return partitions

    def random_attribute_subset(self, attributes):
        # shuffle and pick first F
        if len(attributes) < self.F:
            return attributes
        shuffled = attributes[:]  # make a copy
        random.shuffle(shuffled)
        return shuffled[:self.F]

    def tdidt(self, current_instances, available_attributes):
        # basic approach (uses recursion!!):
        # print("available_attributes:", available_attributes)

        # select an attribute to split on
        case_3 = False
        attribute = self.select_attribute(
            current_instances, available_attributes)
        available_attributes.remove(attribute)
        tree = ["Attribute", attribute]
        # group data by attribute domains (creates pairwise disjoint partitions)
        partitions = self.partition_instances(current_instances, attribute)
        # print('partitions', partitions)
        # for each partition, repeat unless one of the following occurs (base case)
        for att_value, att_partition in partitions.items():
            # print("curent attribute value:", att_value, len(att_partition))
            value_subtree = ["Value", att_value]

            #    CASE 1: all class labels of the partition are the same => make a leaf node
            if len(att_partition) > 0 and self.check_all_same_class(att_partition):
                leaf_node = ['Leaf', att_partition[0][-1],
                             len(att_partition), len(current_instances)]

                value_subtree.append(leaf_node)

            #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
            elif len(att_partition) > 0 and len(available_attributes) == 0:
                labels, freqs = myutils.get_frequencies(att_partition, -1)
                leaf_label = labels[freqs.index(max(freqs))]
                leaf_node = ['Leaf', leaf_label,
                             len(att_partition), len(current_instances)]
                # stats = self.compute_partition_stats(att_partition)
                value_subtree.append(leaf_node)
                # TODO: we have a mix of labels, handle clash with majority
                # vote leaf node

            #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
            elif len(att_partition) == 0:
                case_3 = True
                # TODO: "backtrack" to replace the attribute node
                # with a majority vote leaf node

            else:  # the previous conditions are all false... recurse!!
                subtree = self.tdidt(
                    att_partition, available_attributes.copy())

                value_subtree.append(subtree)
            if not case_3:
                tree.append(value_subtree)
            else:
                labels, freqs = myutils.get_frequencies(current_instances, -1)
                leaf_label = labels[freqs.index(max(freqs))]
                leaf_node = ['Leaf', leaf_label,
                             len(current_instances), len(current_instances)]
                tree = leaf_node
                # TODO: append subtree to value_subtree and to tree
                # appropriately
        return tree

    def generate_att_domain(self):
        attribute_domain = {}
        for i in range(len(self.X_train[0])):
            col_avail_vals = sorted(
                list(set([val[i] for val in self.X_train])))
            attribute_domain[self.header[i]] = col_avail_vals

        self.attribute_domains = attribute_domain

    def check_all_same_class(self, instances):
        # True if all instances have same label
        # Helpful for base case #1 (all class labels are the same... make a leaf node)
        all_same = True
        for instance in instances:
            if instance[-1] != instances[0][-1]:
                all_same = False
        return all_same

    def compute_partition_stats(self, instances):

        labels, subtables = myutils.group_by_values(instances, -1)
        stats = []
        for i in range(len(labels)):
            stats.append(labels[i], len(subtables[i]), len(instances))
        print('stats:', stats)

        return stats

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_pred = []
        for instance in X_test:
            pred = self.tdidt_predict(self.header, self.tree, instance)
            y_pred.append(pred)

        return y_pred  # TODO: fix this

    def tdidt_predict(self, header, tree, instance):
        # recursively traverse the tree
        # we need to know where we are in the tree...
        # are we at a leaf node (base case) or
        # attribute node
        info_type = tree[0]
        if info_type == "Leaf":
            return tree[1]
        # we need to match the attribute's value in the
        # instance with the appropriate value list
        # in the tree
        # a for loop that traverses through
        # each value list
        # recurse on match with instance's value
        att_index = header.index(tree[1])

        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance[att_index]:
                # we have a match, recurse
                return self.tdidt_predict(header, value_list[2], instance)

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.
        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        if attribute_names is None:
            atts = self.header

        print(self.print_tdidt_rules(self.tree, atts, class_name))

    def print_tdidt_rules(self, tree, header, class_label, rule="IF ", depth=0):

        if tree[0] == 'Leaf':
            # print('Leaf node:', tree[1])
            rule = rule[:len(rule) - 5]
            rule += ' THEN ' + class_label + ' = ' + tree[1]
            print(rule)
            return
        elif tree[0] == 'Attribute':
            # print('Attribute:', tree[1])
            rule += tree[1] + ' == '
        elif tree[0] == 'Value':
            # print('Value:', tree[1])
            rule += tree[1] + ' AND '

        for i in range(2, len(tree)):
            self.print_tdidt_rules(tree[i], header, class_label, rule)

    def recurse_tree_viz(self, subtree, tree_data, parent=None):

        if subtree[0] == 'Leaf':
            # print('Leaf node:', subtree[1])
            tree_data += ' [label=' + subtree[1] + '];\n'

            return
        elif subtree[0] == 'Attribute':
            # print('Attribute:', subtree[1])
            parent = subtree[1]
            tree_data += subtree[1] + ' [shape=box];\n'

        elif subtree[0] == 'Value':
            # print('Value:', subtree[1])
            tree_data += parent + ' -- ' + subtree[1]

        for i in range(2, len(subtree)):
            self.recurse_tree_viz(subtree[i], tree_data, parent=subtree[1])

    # BONUS method

    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).
        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        atts = []
        if attribute_names is not None:
            atts = attribute_names

        dot_data = 'graph g {'
        tree_data = '\n'
        self.recurse_tree_viz(self.tree, tree_data)
        dot_data += tree_data + '}'
        #cmd = 'dot -Tsvg -o' + pdf_fname + dot_fname
        # os.system(cmd)

        pass  # TODO: (BONUS) fix this
