import json
import numpy as np
import gym
from script.strategy.utils import var2index, combinate, default_config

def run(players, num_per_team, num_matches = 1, enumerate_max = 100):
    players = sorted(players)
    tmp = []
    for i in range(num_matches):
        L = players[i*2*num_per_team:(i+1)*2*num_per_team]
        # random
        candidates = combinate(L, num_per_team, max_num=1)
        tmp.append(candidates[0])
    return np.array(tmp).reshape((num_matches, 2, num_per_team))

if __name__ == '__main__':
    from enmatch.env import *
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
        matches = run(players, num_per_team, num_matches=len(players)//num_per_team//2)
        # print(matches.shape)
        # matches = np.array([[[0,1,2],[3,4,5]]])
        # matches = np.array([[[0,1,2],[3,4,5]],[[6,7,8],[9,10,11]],[[12,13,14],[15,16,17]]])
        # print(matches.shape)
        for match in matches:
            actions = match.reshape(-1)
            actions = var2index(players, match.reshape(-1))
            assert len(actions) == 2*num_per_team
            for action in actions:
                obs, reward, done, info = env.step(action)
            rewards.append(reward)
            objs.append(info['obj'])
        env.reset()
    # print(rewards)
    # print(objs)
    print('instances:',num_instances,'avg rewards:',np.sum(rewards)/num_instances,'avg obj:',np.sum(objs)/num_instances, sep=' ')