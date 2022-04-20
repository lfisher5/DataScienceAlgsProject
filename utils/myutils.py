'''##############################################
# Programmer: Lauren Fisher
# Class: CptS 322-01, Spring 2022
# Programming Assignment #5
# 2/9/22
#
# Description: This program has utility reusable functions for classifiers 
##############################################'''
import numpy as np
import random
import myevaluation
import numpy as np
from itertools import chain


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


def group_by_determ(table, col_idx, vals):
    group_subtables = [[] for _ in vals]  # e.g. [[], [], []]

    for row in table:
        groupby_val = row[col_idx]  # e.g. this row's modelyear
        # which subtable does this row belong?
        groupby_val_subtable_index = vals.index(groupby_val)
        group_subtables[groupby_val_subtable_index].append(
            row.copy())  # make a copy

    return group_subtables


def randomize_in_place(a_list, par_list=None):
    for i in range(len(a_list)):
        j = random.randint(0, len(a_list) - 1)
        a_list[i], a_list[j] = a_list[j], a_list[i]
        if par_list is not None:
            par_list[i], par_list[j] = par_list[j], par_list[i]


def get_frequencies(table, col_idx):
    """Gets frequencies for column values in table
        Args:
            table(list of list): 2d dataset
            header(list of str): list of attributes
            col_name(str): name of column to extract
        Returns:
            values: unique values in table
            counts: frequency of values
        """
    col = get_column(table, col_idx)
    col.sort()
    values = []
    counts = []
    for value in col:
        if value in values:  # seen it before
            counts[-1] += 1  # okay because sorted
        else:  # haven't seen it before
            values.append(value)
            counts.append(1)

    return values, counts


def doe_mileage_discretizer(inst):
    disc = []
    for val in inst:
        if val <= 13:
            disc.append(1)
        elif val > 13 and val < 15:
            disc.append(2)
        elif val >= 15 and val < 16:
            disc.append(3)
        elif val >= 16 and val < 19:
            disc.append(4)
        elif val >= 19 and val < 23:
            disc.append(5)
        elif val >= 23 and val < 26:
            disc.append(6)
        elif val >= 26 and val < 30:
            disc.append(7)
        elif val >= 30 and val < 36:
            disc.append(8)
        elif val >= 36 and val < 45:
            disc.append(9)
        elif val >= 45:
            disc.append(10)

    return disc


def compute_euclidean_distance(v1, v2):
    dists = []
    for i in range(len(v1)):
        if isinstance(v1[i], str):
            if v1[i] == v2[i]:
                dist_el = 0
            else:
                dist_el = 1
        else:
            dist_el = (v2[i] - v1[i]) ** 2
        dists.append(dist_el)

    dist = np.sqrt(sum([dist for dist in dists]))

    return dist


def get_column(table, col_index):
    """Gets column from table
        Args:
            table(list of list): 2d dataset
            header(list of str): list of attributes
            col_name(str): name of column to extract
        Returns:
            col: list from of extracted column
        """

    col = []
    for row in table:
        value = row[col_index]
        col.append(value)
    return col


def get_categorical_frequencies(table, col_index, is_table=True):
    """Gets categorical frequencies for column values in table
        Args:
            table(list of list): 2d dataset
            header(list of str): list of attributes
            col_name(str): name of column to extract
        Returns:
            values: unique values in table
            counts: frequency of values
        """
    if is_table:
        col = get_column(table, col_index)
    else:
        col = table
    col = [str(x) for x in col]
    values = list(sorted(set(col)))
    counts = [0] * len(values)

    for val in col:
        for idx, u_val in enumerate(values):
            if val == u_val:
                counts[idx] += 1

    return values, counts


def get_and_drop_random_instances(table, n_instances):
    random.seed(1)
    rand_instances = []
    rand_idxs = []

    for _ in range(n_instances):
        rand_idx = random.randint(0, len(table.data))
        rand_instances.append(table.data[rand_idx])
        rand_idxs.append(rand_idx)

    table.drop_rows(rand_idxs)

    return rand_instances


def pretty_print_predictions(X_test_inst, predictions, y_test):
    for i in range(len(X_test_inst)):
        print('Instance: ', X_test_inst[i])
        print('Predicted: ', predictions[i])
        print('Actual: ', y_test[i])
        print('-'*10)
    print('Accuracy: ', calculate_acc(predictions, y_test), '%')


def calculate_acc(predictions, y_test):
    sum = 0
    for i in range(len(predictions)):
        if predictions[i] == y_test[i]:
            sum += 1
    return 100 * sum / len(predictions)


def pretty_print_performance(model_name, y_true, y_pred):
    print('----------------------------')
    print(model_name)
    acc = myevaluation.accuracy_score(y_true, y_pred)
    print('Accuracy:', acc)
    print('Error Rate:', 1-acc)
    print('----------------------------')


def group_by_values(table, col_idx):
    """Groups instances by given column in table
        Args:
            table(list of list): 2d dataset
            header(list of str): list of attributes
            group_by_col_name(str): name of column to extract
        Returns:
            group_names: unique values in table
            group_subtables: table of fitting instances for groups
        """
    groupby_col = get_column(table, col_idx)
    group_names = sorted(list(set(groupby_col)))  # e.g. [75, 76, 77]
    group_subtables = [[] for _ in group_names]  # e.g. [[], [], []]

    for row in table:
        groupby_val = row[col_idx]  # e.g. this row's modelyear
        # which subtable does this row belong?
        groupby_val_subtable_index = group_names.index(groupby_val)
        group_subtables[groupby_val_subtable_index].append(
            row.copy())  # make a copy

    return group_names, group_subtables


def cross_val_stats(classifier_name, trues, predicts):
    stats = []
    stat_names = ['Accuracy', 'Error Rate', 'Precision',
                  'Recall', 'F1 Score', 'Confusion Matrix']

    stats.append(myevaluation.accuracy_score(
        trues, predicts, normalize=True))
    stats.append(1 - stats[0])
    stats.append(myevaluation.binary_precision_score(trues, predicts))
    stats.append(myevaluation.binary_recall_score(trues, predicts))
    stats.append(myevaluation.binary_f1_score(trues, predicts))
    stats.append(myevaluation.confusion_matrix(
        trues, predicts, list(set(trues))))
    print('-' * 10)
    print('Classifier:', classifier_name)
    print('-' * 10)
    for j in range(len(stats)):
        print('-' * 10)
        print(stat_names[j] + ':', stats[j])


def multi_model_stats_gen(data, target_vals):
    X_train_idx_folds, X_test_idx_folds = myevaluation.stratified_kfold_cross_validation(
        data, target_vals, n_splits=10)

    X_train = [[data[idx] for idx in fold] for fold in X_train_idx_folds]
    y_train = [[target_vals[idx] for idx in fold]
               for fold in X_train_idx_folds]
    X_test = [[data[idx] for idx in fold] for fold in X_test_idx_folds]
    y_test = [[target_vals[idx] for idx in fold]
              for fold in X_test_idx_folds]

    knn = MyKNeighborsClassifier(3)
    dummy = MyDummyClassifier()
    bayes = MyNaiveBayesClassifier()
    dec_tree = MyDecisionTreeClassifier()

    classifier_names = ['knn', 'bayes', 'dummy', 'decision_tree']
    models = [knn, bayes, dummy, dec_tree]
    preds = [[] for _ in models]

    for i in range(len(X_train)):
        for j in range(len(models)):
            models[j].fit(X_train[i], y_train[i])
            preds[j] += models[j].predict(X_test[i])

    # converting 2d list into 1d
    # using chain.from_iterables
    y_test = list(chain.from_iterable(y_test))

    for i in range(len(preds)):
        cross_val_stats(classifier_names[i], y_test, preds[i])
