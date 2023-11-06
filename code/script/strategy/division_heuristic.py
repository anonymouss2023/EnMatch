import json
import numpy as np
import gym
from script.strategy.utils import var2index, combinate, default_config


def run(players, num_per_team, obj_fn, num_matches = 1, enumerate_max = 100):
    players = sorted(players)
    tmp = []
    for i in range(num_matches):
        L = players[i*2*num_per_team:(i+1)*2*num_per_team]
        candidates = combinate(L, num_per_team, max_num=enumerate_max)
        best_ind = np.argmin(obj_fn(candidates))
        tmp.append(candidates[best_ind])
    result = np.array(tmp).reshape((num_matches, 2, num_per_team))
    return result, obj_fn(result)

if __name__ == '__main__':
    from enmatch.env import *
    rewards, objs = [], []
    config = {**default_config}
    num_per_team = config['num_per_team']
    num_matches = config['num_matches']
    recsim = SeqSimpleMatchRecEnv(config = config, state_cls=SeqSimpleMatchState)
    env = RecEnvBase(recsim)
    obj_fn = env.sim.obj_fn
    env.reset()
    num_instances = 1000
    for _ in range(num_instances):
        samples:SimpleMatchState = env.samples
        # raw_state = samples.records
        players = samples.players[0]
        matches, obj = run(players, num_per_team, obj_fn, num_matches=num_matches, enumerate_max = 10)
        # matches, obj = run(players, num_per_team, obj_fn, num_matches=num_matches, enumerate_max = max(100,pow(num_per_team,3)))
        # objs.extend(matches2obj(env, matches, players, num_per_team))
        # print(matches.shape)
        # matches = np.array([[[0,1,2],[3,4,5]]])
        # matches = np.array([[[0,1,2],[3,4,5]],[[6,7,8],[9,10,11]],[[12,13,14],[15,16,17]]])
        # print(matches.shape)
        # print(matches)
        # for match in matches:
        #     actions = match.reshape(-1)
        #     actions = var2index(players, match.reshape(-1))
        #     assert len(actions) == 2*num_per_team
        #     for action in actions:
        #         obs, reward, done, info = env.step(action)
        #     rewards.append(reward)
        objs.append(obj)
        # samples:SimpleMatchState = env.samples
        # print(env.samples.team_1, env.samples.team_2)
        env.reset()
    # print(rewards)
    # print(objs)
    print('instances:',num_instances,'avg obj:',np.sum(objs)/num_instances, sep=' ')