import numpy as np
import pandas as pd
from time import time
from sklearn import svm
import imp_Tomek_Link as itl
import Nonequilibrium_cut_point as ncp
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from utils import score


data = None

def shu(dat): return np.random.shuffle(dat)


def sep(dat, n_spl, random):
    y = dat.values[:, -1]
    kf = StratifiedKFold(n_splits=n_spl, shuffle=True, random_state=random)
    return kf.split(dat, y)


def sd_main(dat, n_spl=10, random=None):
    assert random != None
    np.random.seed(random)
    dat = np.array(dat)
    shu(dat)
    dat = pd.DataFrame(dat)
    res = sep(dat, n_spl, random=random)
    return res, dat


def main_TL(algorithm, res, k, random, choose, distance, is_over):
    k_score = []

    start = time()
    for train_index, test_index in res:
        test = data.iloc[test_index]
        train = data.iloc[train_index]

        tk = 0
        if(choose == "Ncp"):
            train, _, _ = ncp.main(train, k, name=None, distance=distance, random=random, is_over=is_over)
        else:
            train, _, _ = itl.main(train, k, name=None, distance=distance, random=random, is_over=is_over)

        train_x = train[train.columns[:-1]]
        train_y = train[train.columns[-1]]

        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random)
        best_val_score = [0, 0]
        for res_train_index, res_val_index in kf.split(train_x, train_y):
            # 10fold
            res_train_x = train_x.iloc[res_train_index]
            res_train_y = train_y.iloc[res_train_index]

            res_val_x = train_x.iloc[res_val_index]
            res_val_y = train_y.iloc[res_val_index]

            if algorithm == "CART":
                clf = DecisionTreeClassifier(criterion='gini')
            elif algorithm == "ID3":
                clf = DecisionTreeClassifier(criterion='entropy')
            elif algorithm == "RF":
                clf = RandomForestClassifier()
            elif algorithm == "NB":
                clf = GaussianNB()
            elif algorithm == "SVM":
                clf = svm.SVC(probability=True)
            elif algorithm == "MLP":
                from run import number_of_neurons
                clf = MLPClassifier(number_of_neurons)
            elif algorithm == "GBDT":
                clf = GradientBoostingClassifier()

            try:
                clf.fit(res_train_x, res_train_y)
                predict_val_y = clf.predict_proba(res_val_x.values)
                val_score = score(predict_val_y, res_val_y)

                test_x = test[test.columns[: -1]]
                test_y = test[test.columns[-1]]
                predict_y = clf.predict_proba(test_x.values)
                one_score = score(predict_y, test_y)
            except Exception as e:
                val_score = [0.5, 0.5]
                one_score = [0.5, 0.5]

            if sum(best_val_score) < sum(val_score):
                best_val_score = one_score

        # end of 10 fold
        k_score.append(best_val_score)
    k_score = np.array(k_score)
    end = time()
    return k_score, end-start



def main(algorithm, tdata, k, random, choose, distance, is_over):
    global data
    data = tdata
    res, data = sd_main(data, 10, random)
    return main_TL(algorithm, res, k, random, choose=choose, distance=distance, is_over=is_over)
