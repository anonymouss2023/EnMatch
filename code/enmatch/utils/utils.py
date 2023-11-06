import numpy as np


def combinate(L, num_per_team, max_num=100):
    tmp = []
    L = np.ravel(L)
    for _ in range(max_num):
        np.random.shuffle(L)
        tmp.append(L[:2*num_per_team].copy())
    return np.array(tmp).reshape((max_num, 2, num_per_team))


def var2index(L, var_list):
    # allow L and var_list containing multi same numbers
    L = list(L)
    if isinstance(var_list, list) or isinstance(var_list, type(np.array([]))):
        tmp = []
        for var in var_list:
            tmp.append(L.index(var))
        return tmp
    else:
        return L.index(var_list)


def remove(L, tmp):
    L = list(L)
    for x in np.ravel(tmp):
        L.remove(x)
    return L


