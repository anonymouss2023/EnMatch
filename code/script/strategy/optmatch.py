import json
import numpy as np
import gym
import copy
from script.strategy.utils import var2index, combinate, remove, default_config


def run(players, num_per_team, obj_fn, num_matches = 1, enumerate_max = 100):
    # players = sorted(players)
    tmp = []
    obj = 0
    L = copy.deepcopy(players)
    # 一场场匹配，每次选最优的
    for i in range(num_matches):
        # L = players[i*2*num_per_team:(i+1)*2*num_per_team]
        candidates = combinate(L, num_per_team, max_num=enumerate_max)
        scores = obj_fn(candidates)
        scores_sorted = sorted(enumerate(scores), key=lambda x:x[1])
        best_score = pow(2, 15)
        best_res = None
        for i, _ in scores_sorted[:1]:
            LL = candidates[i]
            candidates_tmp = combinate(LL, num_per_team, max_num=enumerate_max)
            score_tmp = obj_fn(candidates_tmp)
            obj += np.min(score_tmp)
            best_ind = np.argmin(score_tmp)
            if best_res is None or score_tmp[best_ind] <= best_score:
                best_res = candidates_tmp[best_ind]
                best_score = score_tmp[best_ind]
        tmp.append(best_res)
        L = remove(L, best_res)
    return np.array(tmp).reshape((num_matches, 2, num_per_team)), obj

if __name__ == '__main__':
    from enmatch.env import *
    config = {**default_config}
    num_per_team = config['num_per_team']
    num_matches = config['num_matches']
    rewards, objs = [], []
    recsim = SeqSimpleMatchRecEnv(config = config, state_cls=SeqSimpleMatchState)
    env = RecEnvBase(recsim)
    obj_fn = env.sim.obj_fn
    env.reset()
    num_instances = 1000
    for _ in range(num_instances):
        samples:SimpleMatchState = env.samples
        # raw_state = samples.records
        players = samples.players[0]
        # matches, obj = run(players, num_per_team, obj_fn, num_matches=num_matches, enumerate_max=max(100,pow(num_per_team,3)))
        matches, obj = run(players, num_per_team, obj_fn, num_matches=num_matches, enumerate_max=10)
        # print(matches.shape)
        # matches = np.array([[[0,1,2],[3,4,5]]])
        # matches = np.array([[[0,1,2],[3,4,5]],[[6,7,8],[9,10,11]],[[12,13,14],[15,16,17]]])
        # print(matches.shape)
        # for match in matches:
        #     actions = match.reshape(-1)
        #     actions = var2index(players, match.reshape(-1))
        #     assert len(actions) == 2*num_per_team
        #     for action in actions:
        #         obs, reward, done, info = env.step(action)
        #     rewards.append(reward)
        objs.append(obj)
        env.reset()
    # print(rewards)
    # print(objs)
    print('instances:',num_instances,'avg rewards:',np.sum(rewards)/num_instances,'avg obj:',np.sum(objs)/num_instances, sep=' ')