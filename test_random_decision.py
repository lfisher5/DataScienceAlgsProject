import numpy as np
from utils.myrandomforestclassifier import MyRandomForestClassifier
import utils.myutils

interview_header = ["level", "lang", "tweets", "phd", "interviewed_well"]
X_train_interview = [
    ["Senior", "Java", "no", "no"],
    ["Senior", "Java", "no", "yes"],
    ["Mid", "Python", "no", "no"],
    ["Junior", "Python", "no", "no"],
    ["Junior", "R", "yes", "no"],
    ["Junior", "R", "yes", "yes"],
    ["Mid", "R", "yes", "yes"],
    ["Senior", "Python", "no", "no"],
    ["Senior", "R", "yes", "no"],
    ["Junior", "Python", "yes", "no"],
    ["Senior", "Python", "yes", "yes"],
    ["Mid", "Python", "no", "yes"],
    ["Mid", "Java", "yes", "no"],
    ["Junior", "Python", "no", "yes"]
]

y_train_interview = ['False', 'False', 'True', 'True', 'True', 'False',
                     'True', 'False', 'True', 'True', 'True', 'True', 'True', 'False']


def test_random_decision_tree_fit():
    forest = MyRandomForestClassifier()
    forest.fit(X_train_interview, y_train_interview)
    assert True is False


def test_random_decision_tree_predict():
    assert True is False
