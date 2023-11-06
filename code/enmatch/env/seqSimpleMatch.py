from functools import reduce
from operator import add
from copy import deepcopy as copy
import numpy as np
from enmatch.env.base import RecEnvBase
from enmatch.env.simpleMatch import SimpleMatchState, SimpleMatchRecEnv


class SeqSimpleMatchState(SimpleMatchState):
    def __init__(self, config, records):
        super().__init__(config, records)
        # self.page_items = config.get("page_items", 9)
        self.palyers_per_match = 2 * self.num_per_team

    @property
    def state(self):
        if self.config.get("support_rllib_mask", False):
            # location_mask = self.get_location_mask(self.location_mask, self.cur_steps % self.page_items // 3)
            return {"state": self._state, "action_mask": self.action_mask}
        elif self.config.get("support_d3rl_mask", False):
            cur_steps = np.full((self.batch_size, 1), self.cur_steps)
            page_init = self.cur_steps // self.palyers_per_match * self.palyers_per_match
            page_end = min(page_init + self.palyers_per_match - 1, self.max_steps - 1)
            masked_actions = self.prev_actions[:, page_end + 1 - self.palyers_per_match:page_end + 1]
            return {"state": self._state, "masked_actions": masked_actions, "cur_steps": cur_steps}
        else:
            return self._state

    # def get_complete_states(self):
    #     states = []
    #     for j in range(self.cur_steps):
    #         tmp = copy(self._init_state)
    #         for state, action, i in zip(self._init_state, self.prev_actions[:, j], range(len(self._init_state))):
    #             page_init = j // self.page_items * self.page_items
    #             page_end = page_init + self.page_items - 1
    #             sequence_id = j // self.page_items + 1
    #             # seq
    #             prev_expose = self.prev_actions[i, :page_init] if page_init > 0 else [0]
    #             tmp[i][1] = [tmp[i][1][0], prev_expose]
    #             # dense
    #             prev_item_feat = [
    #                 self.item_info_d[str(x)]['item_vec']
    #                 for x in self.prev_actions[i, page_init:page_end + 1]
    #             ]
    #             cur_item_feat = self.item_info_d[str(action)]['item_vec']
    #             prev_item_feat = np.array(prev_item_feat).flatten()
    #             tmp[i][2] = np.concatenate((tmp[i][2], prev_item_feat, cur_item_feat))
    #             # category
    #             cur_exposed = self.prev_actions[i, page_init:page_end + 1]
    #             tmp[i][3] = np.concatenate((tmp[i][3], [sequence_id], cur_exposed, [action]))
    #         states.append(tmp)
    #     return states

    def get_violation(self):
        tmp = np.ones((self.batch_size,), dtype=np.int)
        for step1 in range(min(self.cur_steps, self.max_steps)):
            for step2 in range(min(self.cur_steps, self.max_steps)):
                if step1!=step2:
                    duplicate_mask = (self.prev_actions[:, step1] != self.prev_actions[:, step2])
                    tmp = tmp & duplicate_mask
        return tmp

    # @property
    # def offline_reward(self):
    #     cur_step = self.cur_steps
    #     if cur_step % 9 != 0:
    #         reward = [0, ] * self.batch_size
    #     else:
    #         action = np.array([list(map(int, x.split('@')[3].split(',')[:cur_step]))
    #                            for x in self.records])
    #         price = self.get_price(action)[:, -self.page_items:]
    #         slate_label = np.array([
    #             list(map(int, x.split('@')[4].split(',')))
    #             for x in self.records
    #         ])
    #         slate_label = slate_label[:, cur_step - self.page_items:cur_step]
    #         reward = np.sum(price * slate_label, axis=1)
    #     return reward

    # @property
    # def info(self):
    #     return [{}]*self.batch_size

    def act(self, actions):
        if self.config.get("support_conti_env", False):
            location_mask = self.get_location_mask(self.location_mask,
                                                   self.cur_steps % self.page_items // 3)
            action_mask = self.action_mask & location_mask & self.special_mask
            actions = self.get_nearest_neighbor_with_mask(actions, self.action_emb, action_mask)
        self.prev_actions[:, self.cur_steps] = actions
        self.action_mask[list(range(self.batch_size)), actions] = 0
        # for i in range(self.batch_size):
        #     if len(np.intersect1d(self.prev_actions[i], self.special_items)) > 0:
        #         self.special_mask[i][self.special_items] = 0
        tmp = copy(self._init_state)
        for state, action, i in zip(self._state, actions, range(self.batch_size)):
            # all players, exists players, matched players
            tmp[i][1][self.prev_actions[i][:self.cur_steps+1]] = -1
            tmp[i][2] = self.players[i][self.prev_actions[i]]
            tmp[i][2][self.prev_actions[i] == -1] = -1
        self._state = tmp
        self.cur_steps += 1
        # if self.cur_steps % self.palyers_per_match == 0:
        #     self.action_mask = np.full((self.batch_size, self.action_size), 1, dtype=np.int)
            # self.special_mask = np.full((self.batch_size, self.action_size), 1, dtype=np.int)


class SeqSimpleMatchRecEnv(SimpleMatchRecEnv):
    """ Implements core recommendation simulator"""

    def __init__(self, config, state_cls):
        super().__init__(config, state_cls)
        # self.page_items = config.get("page_items", 9)
        self.palyers_per_match = 2 * self.num_per_team

    def forward(self, model, samples):
        step = samples.cur_steps
        reward_after_team = self.config.get('reward_after_team', True)
        if reward_after_team:
            if step % self.palyers_per_match == 0:
                # state = samples.state
                start = self.palyers_per_match * (step // self.palyers_per_match - 1)
                end = self.palyers_per_match * (step // self.palyers_per_match)
                prev_actions = samples.prev_actions[:, start:end]
                # macthes = np.reshape(prev_actions, (-1, 2, self.num_per_team))
                matches = samples.get_matches(prev_actions)
                obj = self.obj_fn(matches, samples.opt)
                reward = self.obj2reward(obj, samples.opt, self.upper_bound)
                # if self.config.get("support_rllib_mask", False) or \
                #         self.config.get("support_d3rl_mask", False):
                #     violation = samples.get_violation()
                #     # reward[violation < 0.5] = 0
            else:
                obj = np.array([0, ] * self.batch_size)
                reward = np.array([0, ] * self.batch_size)
        elif not reward_after_team and step%2 == 0:
            # 规定team2>team1, 用cur_team2-cur_team1的相反数作为奖励，规定每一步都有team2>team1；怕出现team2<team1
            prev_actions = samples.prev_actions[:, step-2:step]
            matches = samples.get_matches(prev_actions)
            tmp_obj = self.obj_fn(matches, samples.opt, use_abs=False)
            reward = self.obj2reward(tmp_obj, samples.opt, self.upper_bound)
            # 规定每一步都有team2>team1
            start = self.palyers_per_match * ((step-1) // self.palyers_per_match)
            prev_actions = samples.prev_actions[:,start:step]
            matches = samples.get_matches(prev_actions)
            tmp_obj = self.obj_fn(matches, samples.opt, use_abs=False)
            reward[tmp_obj<0] = -100000
            if step % self.palyers_per_match != 0:
                obj = np.array([0] * self.batch_size)
            else:
                obj = self.obj_fn(matches, samples.opt)
        else:
            obj = np.array([0, ] * self.batch_size)
            reward = np.array([0, ] * self.batch_size)
        for i in range(self.batch_size):
            samples.info[i]['obj'] = obj[i].tolist()
            samples.info[i]['opt'] = samples.opt[i]
        return reward.tolist()

if __name__ == '__main__':
    # from encom.simplematch.environment import SimpleMatchEnv
    batch_size = 1
    num_per_team = 3
    num_matches = 3
    num_players = 2*num_matches*num_per_team
    rewards, objs = [], []
    config = {'sample_file':'../../dataset/simplematch_18_3_100_100000_linear_opt.csv', 'batch_size':batch_size, 'support_rllib_mask':True, 'is_eval':True,
              'num_per_team':num_per_team, 'num_matches':num_matches, 'max_steps':num_players, 'upper_bound':100, 'action_size':num_players,
              'reward_after_team':False}
    recsim = SeqSimpleMatchRecEnv(config = config, state_cls=SeqSimpleMatchState)
    env = RecEnvBase(recsim)
    reward_fn = env.sim.obj_fn
    env.reset()
    num_instances = 1
    for _ in range(num_instances):
        samples:SimpleMatchState = env.samples
        # raw_state = samples.records
        players = samples.players[0]
        # print(players)
        # matches = run(players, num_per_team, reward_fn, num_matches=len(players)//num_per_team//2)
        # matches = np.array([[[0,1,2],[3,4,5]]])
        matches = np.array([[[0,1,2],[3,4,5]],[[6,7,8],[9,10,11]],[[12,13,14],[15,16,17]]])
        for match in matches:
            actions = match.reshape(-1)
            # actions = var2index(players, match.reshape(-1))
            assert len(actions) == 2*num_per_team
            for action in actions:
                obs, reward, done, info = env.step(action)
                print(obs)
                rewards.append(reward)
                objs.append(info['obj'])
                # objs.extend([x['obj'] for x in info])
            # print(samples.team_1, samples.team_2)
        env.reset()
    # print(rewards)
    print('instances:',num_instances,'avg rewards:',np.sum(rewards)/num_instances,'avg objs:',np.sum(objs)/num_instances, sep=' ')