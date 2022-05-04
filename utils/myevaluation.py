import numpy as np
import random
from sklearn.metrics import accuracy_score


def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.
    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!
    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if random_state is not None:
        np.random.seed(random_state)

    shuff_X = X[:]  # copy the table
    shuff_y = y[:]
    if shuffle:
        randomize_in_place(shuff_X, shuff_y)

    # randomize the table
    n = len(X)

    # return train and test sets
    if type(test_size) == float:
        # 2/3 of randomized table is train, 1/3 is test
        split_index = int(test_size * n)

        return shuff_X[0:n-split_index-1], shuff_X[n-split_index-1:], shuff_y[0:n-split_index-1], shuff_y[n-split_index-1:]
    else:
        split_index = test_size
        return shuff_X[0:n-split_index], shuff_X[n-split_index:], shuff_y[0:n-split_index], shuff_y[n-split_index:]


def kfold_cross_validation(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.
    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold
    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    if random_state is not None:
        np.random.seed(random_state)

    X_idxs = list(range(len(X)))
    if shuffle:
        randomize_in_place(X_idxs)
    fold_size = int(len(X) / n_splits)
    X_train_folds = []
    X_test_folds = []
    start_idx = 0
    for i in range(n_splits):
        if i == n_splits - 1:
            test_fold = X_idxs[start_idx: len(X)]
            train_fold = X_idxs[0:start_idx]

        else:
            test_fold = X_idxs[start_idx:start_idx + fold_size]
            train_fold = X_idxs[0:start_idx] + \
                X_idxs[start_idx + fold_size: len(X)]

        X_train_folds.append(train_fold)
        X_test_folds.append(test_fold)
        start_idx += fold_size

    return X_train_folds, X_test_folds


def stratified_kfold_cross_validation(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.
    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.
    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    if random_state is not None:
        np.random.seed(random_state)

    X_idxs = list(range(0, len(X)))
    if shuffle:
        randomize_in_place(X_idxs)

    groups = group_by(X_idxs, y)

    i = 0
    fold = 0
    train_fold = [[] for _ in range(n_splits)]
    test_fold = [[] for _ in range(n_splits)]
    for group in groups:
        for i in range(len(group)):
            test_fold[fold].append(group[i][0])
            fold = (fold + 1) % (n_splits)

    for i in range(len(X)):
        for j in range(len(test_fold)):
            if X_idxs[i] not in test_fold[j]:
                train_fold[j].append(X_idxs[i])

    return train_fold, test_fold


def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.
    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results
    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
    """
    if random_state is not None:
        np.random.seed(random_state)

    n = len(X)
    X_sample = []
    X_out_of_bag = []

    if y is not None:
        y_sample = []
        y_out_of_bag = []
    else:
        y_sample = None
        y_out_of_bag = None

    if n_samples is not None:
        n = n_samples

    for _ in range(n):
        rand_idx = np.random.randint(0, len(X))
        X_sample.append(X[rand_idx])
        if y is not None:
            y_sample.append(y[rand_idx])

    for i in range(len(X)):
        if X[i] not in X_sample:
            X_out_of_bag.append(X[i])
            if y is not None:
                y_out_of_bag.append(y[i])

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag


def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix
    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class
    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """

    conf_matrix = [[0 for _ in range(len(labels))] for _ in range(len(labels))]
    for k in range(len(y_pred)):

        i = labels.index(y_true[k])
        j = labels.index(y_pred[k])
        conf_matrix[i][j] += 1

    return conf_matrix


def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.
    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).
    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """

    true_pos = 0

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            true_pos += 1

    acc = true_pos
    if normalize:
        acc /= len(y_true)

    return acc


def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels
    Returns:
        precision(float): Precision of the positive class
    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]

    tp = 0
    fp = 0

    for i in range(len(y_true)):
        if y_true[i] == pos_label and y_pred[i] == y_true[i]:
            tp += 1
        if y_true[i] != y_pred[i] and y_pred[i] == pos_label:
            fp += 1

    if (tp + fp) != 0:
        precision = float(tp) / float((tp + fp))
    else:
        precision = 0

    return precision


def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels
    Returns:
        recall(float): Recall of the positive class
    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """

    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]

    tp = 0
    fn = 0

    for i in range(len(y_true)):

        if y_true[i] == pos_label and y_pred[i] == y_true[i]:
            tp += 1
        if y_true[i] == pos_label and y_true[i] != y_pred[i]:
            fn += 1

    recall = tp / (tp + fn)

    return recall


def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels
    Returns:
        f1(float): F1 score of the positive class
    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    f1 = 0
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)

    if precision + recall != 0:
        f1 = 2 * (precision * recall) / (precision + recall)

    return f1


def randomize_in_place(a_list, par_list=None):
    for i in range(len(a_list)):
        j = random.randint(0, len(a_list) - 1)
        a_list[i], a_list[j] = a_list[j], a_list[i]
        if par_list is not None:
            par_list[i], par_list[j] = par_list[j], par_list[i]


def group_by(idxs, y):
    """Groups instances by given column in table
        Args:
            table(list of list): 2d dataset
            header(list of str): list of attributes
            group_by_col_name(str): name of column to extract
        Returns:
            group_names: unique values in table
            group_subtables: table of fitting instances for groups
        """
    group_names = sorted(list(set(y)))  # e.g. [75, 76, 77]
    group_subtables = [[] for _ in group_names]  # e.g. [[], [], []]

    for i in idxs:
        # which subtable does this row belong?
        groupby_val_subtable_index = group_names.index(y[i])
        group_subtables[groupby_val_subtable_index].append(
            [i, y[i]])  # make a copy

    return group_subtables


def cross_val_stats(classifier_name, trues, predicts):
    stats = []
    stat_names = ['Accuracy', 'Error Rate', 'Confusion Matrix']

    stats.append(accuracy_score(
        trues, predicts, normalize=True))
    stats.append(1 - stats[0])
    stats.append(confusion_matrix(
        trues, predicts, list(set(trues))))
    print('-' * 10)
    print('Classifier:', classifier_name)
    print('-' * 10)
    for j in range(len(stats)):
        print('-' * 10)
        print(stat_names[j] + ':', stats[j])


def multi_cl_accuracy(predicts, trues):
    return accuracy_score(trues, predicts, normalize=True)
