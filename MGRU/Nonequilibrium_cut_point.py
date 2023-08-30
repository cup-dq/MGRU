import os
import numpy as np
import pandas as pd

def sort_columns(data, i):
    data = np.array(data)
    data = data[data[:, i].argsort()]
    return data


def for_class(data):
    t = data.values[:, -1]
    one = t == 1
    two = t == 2
    return sum(one), sum(two)


def tomek_link(data):
    data = data.T
    ans = []
    for i in range(data.shape[1] - 1):
        data_t = sort_columns(data, i)
        data_t = pd.DataFrame(data_t)
        ans_t = []
        dat_columns = data_t[len(data_t.columns) - 2]

        ans_t.append(int(bool(int(dat_columns[0]) ^ int(dat_columns[1]))))
        for j in range(1, data.shape[0] - 1):
            v_t = (int(bool(int(dat_columns[j - 1]) ^ int(dat_columns[j])))) or (int(bool(int(dat_columns[j + 1]) ^ int(dat_columns[j]))))
            ans_t.append(v_t)
        ans_t.append(int( bool(int(dat_columns[data.shape[0] - 1]) ^ int(dat_columns[data.shape[0] - 2]))))

        t0 = data_t[data_t.columns[-1]]
        t1 = pd.DataFrame([ans_t, list(t0.values)]   ).T
        ans_t_index = pd.DataFrame(sort_columns(t1 , 1))
        ans.append(ans_t_index[0])
    return ans


def main(dat, k, random, name, distance, is_over=None):
    a1, b1 = for_class(dat)
    ans = dat.T.values.tolist()
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
