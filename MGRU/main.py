import os
import numpy as np
import pandas as pd
import warnings
import main_to
warnings.filterwarnings("ignore")

def MGRU(data, method, choose="TL", distance="Mahalanobis", is_over=None, seed=None):
    """
    Func: Calculate Gmean and F1 of the new algorithm
        
    # Classifier settings
        N_Mothod =  ["CART", "SVM", "GBDT"]
    # The new algorithm adopts "TL" or "Ncp" (non-stationary cut point or called UCSS)
        N_Choose = ["TL", "Ncp"]
    # If you use "TL", you can choose whether the distance is Mahalanobis or standardized Euclidean
        N_distance = ["Mahalanobis", "Euclidian"]
    """
    if seed:
        seed = np.random.randint(0, 100)

    # Empty the cache
    for file in os.listdir("cache"):
        os.remove("cache/"+file)
    # Number of columns obtained by the dataset
    K = pd.read_csv(f_n).shape[1] + 1

    k_score = list()
    k_score_time = list()

    k_list = range(K-1, 0, -1)
    for k in k_list:
        score, cost_time = main_to.main(method, data, k, seed, choose=choose, distance=distance, is_over=True)
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
    print(f"[MGRU] algorithm:    {choose}:{distance}   Classifier:{method}      overlap:{is_over}")
    print(f"k = {index}, max_auc = {roc_auc_max}, pr_auc = {pr_auc_max}, time = {index_time}")


if __name__ == "__main__":
    # Read dataset
    f_n = "data/ecoli1.csv"
    data = pd.read_csv(f_n)
    seed = np.random.randint(0, 100)

    # This experiment uses ten-fold cross-validation
    MGRU(data, method="CART", choose="Ncp", distance=None, is_over=True, seed=seed)
    MGRU(data, method="CART", choose="TL", distance="Mahalanobis", is_over=True, seed=seed)
