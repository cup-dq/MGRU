import os
import numpy as np
import pandas as pd
import warnings
from random import randint
import main_to
from utils import score

warnings.filterwarnings("ignore")

# ------- Paramater -------
# This experiment uses ten-fold cross-validation
f_n = "data/ecoli1.csv"

# models
N_Mothod =  ["CART", "SVM","GBDT"]

# The new algorithm adopts "TL" or "Ncp" (non-stationary cut point or called UCSS)
N_Choose = ["TL", "Ncp"]

# If you use "TL", you can choose whether the distance is Mahalanobis or standardized Euclidean
N_distance = ["Mahalanobis", "Euclidian"]

random = np.random.randint(0,100)

# Number of nearest neighbors
number_of_neurons = [10]

# Number of nearest neighbors
k_neighbors = 5
CMP_mothod = ["NULL","SMOTE", "TomekLinks"]
# "MESA","RUSBoost","SMOTEBoost","SPEnsemble"
# ------- Paramater -------


def main(mothod, choose="TL", distance="Mahalanobis", is_over=None):
    """
    Func: Calculate Gmean and F1 of the new algorithm
    """
    k_score = list()
    k_score_time = list()

    k_list = range(K-1, 0, -1)
    for k in k_list:
        score, cost_time = main_to.main(mothod, data, k, random, choose=choose, distance=distance, is_over=True)
        if len(score) != 10:
            k_score.append([[0]] * 10)
            k_score_time.append(0)
        else:
            k_score.append(score)
            k_score_time.append(cost_time)

    k_score = np.array(k_score)
    roc_auc = k_score[:,:,0]
    pr_auc = k_score[:,:,1]
    k_score_time = np.array(k_score_time)

    idx = roc_auc.mean(axis=1).argmax()
    roc_auc_max = roc_auc.mean(axis=1)[idx]

    pr_auc_max = pr_auc.mean(axis=1)[idx]
    pr_auc_max = 1 - pr_auc_max if pr_auc_max < 0.5 else pr_auc_max

    index = k_list[idx]
    index_time = k_score_time[idx]
    k_score = k_score.mean(axis=1).max()
    print(f"[MGRU] algorithm:    {choose}:{distance}   Classifier:{mothod}      overlap:{is_over}")
    print(f"k = {index}, max_auc = {roc_auc_max}, pr_auc = {pr_auc_max}, time = {index_time}")


def main_go():
    for clf in N_Mothod:
        for mod in N_Choose:
            if(mod == "Ncp"):
                main(mothod=clf, choose=mod, distance=None, is_over=True)
            elif(mod == "TL"):
                for dis in N_distance:
                    main(mothod=clf, choose=mod, distance=dis, is_over=True)

def ddel():
    t = os.listdir("cache")
    for file in t:
        os.remove("cache/"+file)


if __name__ == "__main__":
    # Empty the cache
    ddel()

    data = pd.read_csv(f_n)
    K = pd.read_csv(f_n).shape[1] + 1
    main_go()