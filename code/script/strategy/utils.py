import numpy as np



# file_name = '../../dataset/simplematch_6_3_100_100000_linear_opt.csv.2'
# file_name = '../../dataset/simplematch_18_3_100_100000_linear_opt.csv.easy'
# file_name = '../../dataset/simplematch_20_10_1000_100000_linear_opt.csv.2.easy'
file_name = '../../dataset/simplematch_60_10_1000_100000_linear_opt.csv.easy'
num_players = int(file_name.split('_')[1])
batch_size = 1
num_per_team = int(file_name.split('_')[2])
num_matches = int(num_players/2/num_per_team)

default_config = {
    'env':'SeqSimpleMatchEnv-v0',
    'trial_name':'test',
    'sample_file': file_name,
    'batch_size':batch_size,
    'support_rllib_mask':True,
    'is_eval':True,
    'num_per_team':num_per_team,
    'num_matches':num_matches,
    'max_steps':num_players,
    'upper_bound':1000,
    'action_size':num_players,
    # 'reward_after_team':False,
    'vocab_size': 10000,
}


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
