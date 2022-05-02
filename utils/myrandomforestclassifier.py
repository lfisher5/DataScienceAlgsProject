import random
from utils import myevaluation
from utils import myclassifiers
import operator
from itertools import chain


class MyRandomForestClassifier:
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
        self.forest = []
        self.N = None
        self.M = None
        self.F = None

    def fit(self, X_train, y_train, N=5, M=3, F=3):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.
        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
        """
        self.N = N
        self.M = M
        self.F = F

        remainder_idxs, test_idxs = myevaluation.stratified_kfold_cross_validation(
            X_train, y_train, n_splits=6, random_state=101, shuffle=True)

        remainder_idxs = remainder_idxs[0]
        test_idxs = test_idxs[0]

        remainder_X = [X_train[row] for row in remainder_idxs]
        remainder_y = [y_train[row] for row in remainder_idxs]
        test_X = [X_train[row] for row in test_idxs]
        test_y = [y_train[row] for row in test_idxs]
        dec_trees = []
        trues_preds = []
        accs = []
        for i in range(self.N):
            x_train, x_val, Y_train, Y_val = myevaluation.bootstrap_sample(
                remainder_X, remainder_y, random_state=101)
            dec_tree = myclassifiers.MyModifiedDecisionTreeClassifier()
            dec_tree.fit(x_train, Y_train, F)
            dec_trees.append(dec_tree)
            y_preds = dec_tree.predict(x_val)
            trues_preds.append([y_preds, x_val])
            acc_score = myevaluation.multi_cl_accuracy(y_preds, Y_val)
            accs.append([i, acc_score])
        accs.sort(key=operator.itemgetter(-1), reverse=True)
        accs = accs[:M]
        dec_trees_clean = []
        for acc in accs:
            dec_trees_clean.append(dec_trees[acc[0]])

        preds = []
        trues = []
        for dec_tree in dec_trees_clean:
            preds.append(dec_tree.predict(test_X))
            trues.append(test_y)
        preds = list(chain.from_iterable(preds))
        trues = list(chain.from_iterable(trues))
        tot_acc = myevaluation.multi_cl_accuracy(preds, trues)
        print(tot_acc)


# The Random Forest Procedure

# Divide D into a test and remainder set
# Take 1/3 for test set, 2/3 for remainder set
# Ensure test set has same distribution of class labels as D ("stratified")
# Randomly select instances when generating test set

# Create N bootstrap samples from remainder set
# Each results in a training (63%) and validation (36%) set
# Build and test a classifier for each of the N bootstrap samples
# Each classifier is a decision tree using F-sized random attribute subsets
# Determine accuracy of classifier using validation set

# Pick the M best classifiers generated in step 2

# Use test set from step 1 to determine performance of the ensemble of M classifiers (using simple majority voting)
