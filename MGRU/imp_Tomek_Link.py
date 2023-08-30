from scipy.spatial.distance import pdist
import numpy as np
import pandas as pd
import os

def msjl(data, distance, col, random):
    mu = np.array([data.mean(axis=0)]).T
    ans = []
    index_train = data.index
    col = str(col)+"_"+random
    cache = "cache/"
    if(distance == "Mahalanobis"):
        if(os.path.exists(cache+"t_Mahalanobis"+col)):
            with open(cache+"t_Mahalanobis"+col, "r") as f:
                ans = f.read()
                return eval(ans)
        for i in range(data.shape[0]):
            x = np.array([data.loc[index_train[i]]]).T
            ans_t = np.sqrt(
                np.dot(np.dot((x - mu).T, np.cov(data.loc[index_train[0]])), (x - mu)))
            ans.append(ans_t[0][0])

        with open(cache+"t_Mahalanobis"+col, "w") as f:
            f.write(str(ans))
        return ans
    elif(distance=="Euclidian"):
        if(os.path.exists(cache+"t_Euclidian"+col)):
            try:
                with open(cache+"t_Euclidian"+col,"r") as f:
                    ans = f.read()
                    return eval(ans)
            except:
                for i in range(data.shape[0]):
                    x = np.array( [data.loc[index_train[i]]] )
                    X=np.vstack([x[0],mu.T[0]])
                    sk=np.var(X,axis=0,ddof=1)
                    d1=np.sqrt(((x - mu) ** 2 /sk).sum())
                    ans.append(d1)
                with open(cache+"t_Euclidian"+col,"w") as f:
                    f.write(str(ans))
                return ans
        for i in range(data.shape[0]):
            x = np.array( [data.loc[index_train[i]]] )
            X=np.vstack([x[0],mu.T[0]])
            sk=np.var(X,axis=0,ddof=1)
            d1=np.sqrt(((x - mu) ** 2 /sk).sum())
            ans.append(d1)
        with open(cache+"t_Euclidian"+col,"w") as f:
            f.write(str(ans))
        return ans


def sort_columns(data, i):
    data = np.array(data)
    data = data[data[:, i].argsort()]
    return data


def tomek_link(data):
    data = data.T
    ans = []
    for i in range(data.shape[1] - 1):
        data_t = sort_columns(data, i)
        data_t = pd.DataFrame(data_t)
        ans_t = []
        dat_columns = data_t[len(data_t.columns) - 2]
        append = int(bool(int(dat_columns[0]) ^ int(dat_columns[1])))
        ans_t.append(append)
        for j in range(1, data.shape[0] - 1):
            v_t = (int(bool(int(dat_columns[j - 1]) ^ int(dat_columns[j])))) or ( int(bool(int(dat_columns[j + 1]) ^ int(dat_columns[j]))))
            ans_t.append(v_t)
        ans_t.append(int( bool(int(dat_columns[data.shape[0] - 1]) ^ int(dat_columns[data.shape[0] - 2]))))

        t1 = data_t[data_t.columns[-1]]
        t2 = pd.DataFrame([ans_t, list(t1.values)]).T
        ans_t_index = pd.DataFrame(sort_columns(t2, 1))
        ans.append(ans_t_index[0])
    return ans


def for_class(data):
    t = data.values[:, -1]
    one = t == 1
    two = t == 2
    return sum(one), sum(two)


def main(dat, k, random, name=None, distance="Mahalanobis", is_over=False):
    a1, b1 = for_class(dat)
    ans = []
    dat_non_class = dat.drop([dat.columns[-1]], axis=1)
    for i in range(len(dat.columns) - 1):
        dat_temp = dat_non_class.drop([dat.columns[i]], axis=1)
        ans_temp = msjl(dat_temp, distance=distance, col=i, random=str(random))
        ans.append(ans_temp)

    ans.append(dat[dat.columns[-1]].values.tolist())
    ans.append([i for i in range(len(dat[dat.columns[-1]]))])
    ans = pd.DataFrame(ans)

    ans = tomek_link(ans)
    ans = pd.DataFrame(ans).T
    res_k = ans.sum(axis=1)

    if is_over:
        labels = dat.values[:,-1].flatten()
        pos_labels = np.argwhere(labels == 2).flatten()
        last_labels = np.argwhere(labels == 1).flatten()
        res_k = res_k[pos_labels]
        res_k_later_index = res_k[res_k >= k].index.values
        res_k_later_index = np.hstack([res_k_later_index,last_labels])
        res_dat = dat.iloc[res_k_later_index]
        a2, b2 = for_class(res_dat)
    else:
        res_k_later_index = res_k[res_k >= k].index
        res_k_later_index = dat.index[res_k_later_index]
        res_dat = dat.loc[res_k_later_index]
        a2, b2 = for_class(res_dat)
    return res_dat, a2, b2
