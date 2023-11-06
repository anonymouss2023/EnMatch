import numpy as np
from copy import deepcopy as copy
from enmatch.env import RecSimBase, RecState, RecEnvBase
import json
import torch
from ortools.sat.python import cp_model
# import tensorflow as tf
# from enmatch.utils.datautil import FeatureUtil


class SimpleMatchState(RecState):
    def __init__(self, config, records):
        super().__init__(config, records)
        self.batch_size = self.config["batch_size"]
        self.num_per_team = config['num_per_team']
        self.num_matches = config['num_matches']
        self.action_size = self.config["action_size"]
        self.action_emb_size = self.config.get("action_emb_size", 32)
        self.max_steps = config['max_steps']
        self.match_assign_mode = config.get('match_assign_mode', 'alter')
        self.infos = [{} for _ in range(self.batch_size)]
        self.prev_actions = np.full((self.batch_size, self.max_steps), -1, dtype=np.int)
        self.action_mask = np.full((self.batch_size, self.action_size), 1, dtype=np.int)
        self.cur_steps = 0
        # self.item_info_d, self.action_emb = self.get_iteminfo_from_file(iteminfo_file, self.action_size)
        self.action_emb = None
        if config.get('support_onehot_action', False):
            config['action_emb_size'] = self.action_size
            self.action_emb_size = self.action_size
            self.action_emb = np.eye(self.action_size)

    @staticmethod
    def records_to_state(records):
        def fn(record):
            players = list(map(int,record.split('@')[0].split(',')))
            # all players, exists players, matched players
            return [np.array(players), np.array(players), [-1]*len(players)]
        state = list(map(fn, records))
        return state

    @property
    def state(self):
        if self.config.get("support_rllib_mask", False):
            # location_mask = self.get_location_mask(self.location_mask, self.cur_steps // 3)
            return {
                "state": self._state,
                "action_mask": self.action_mask
            }
        elif self.config.get("support_d3rl_mask", False):
            cur_steps = np.full((self.batch_size, 1), self.cur_steps)
            return {
                "state": self._state,
                "masked_actions": self.prev_actions,
                "cur_steps": cur_steps
            }
        else:
            return self._state

    @property
    def user(self):
        return [0 for x in self.records]

    @property
    def opt(self):
        return [float(x.split('@')[-1]) for x in self.records]

    @property
    def players(self):
        return np.array([list(map(int,x.split('@')[0].split(','))) for x in self.records])

    def get_matches(self, player_ind):
        tmp=self.players[np.arange(self.batch_size).reshape(-1, 1), player_ind]
        if self.match_assign_mode == 'alter':
            return tmp.reshape((self.batch_size,-1,2)).transpose((0, 2, 1))
        else:
            return tmp.reshape((self.batch_size,2,-1))

    @property
    def team_1(self):
        palyers_per_match = self.num_per_team * 2
        start = palyers_per_match * (self.cur_steps // palyers_per_match - 1)
        end = palyers_per_match * (self.cur_steps // palyers_per_match)
        return self.get_matches(self.prev_actions[:, start:end])[:,0,:]

    @property
    def team_2(self):
        palyers_per_match = self.num_per_team * 2
        start = palyers_per_match * (self.cur_steps // palyers_per_match - 1)
        end = palyers_per_match * (self.cur_steps // palyers_per_match)
        return self.get_matches(self.prev_actions[:, start:end])[:,1,:]

    # def get_complete_states(self):
    #     states = []
    #     for j in range(self.max_steps):
    #         tmp = copy(self._init_state)
    #         for state, action, i in zip(self._init_state, self.prev_actions[:, j], range(len(self._init_state))):
    #             # dense
    #             prev_item_feat = [self.item_info_d[str(x)]['item_vec'] for x in self.prev_actions[i, :]]
    #             cur_item_feat = self.item_info_d[str(action)]['item_vec']
    #             prev_item_feat = np.array(prev_item_feat).flatten()
    #             tmp[i][2] = np.concatenate((tmp[i][2], prev_item_feat, cur_item_feat))
    #             # category
    #             sequence_id = 1
    #             tmp[i][3] = np.concatenate((tmp[i][3], [sequence_id], self.prev_actions[i, :], [action]))
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
    # def offline_action(self):
    #     cur_step = self.cur_steps
    #     if cur_step < self.max_steps:
    #         if self.config.get("support_conti_env", False):
    #             action = [self.action_emb[int(x.split('@')[3].split(',')[cur_step])] for x in self.records]
    #         else:
    #             action = [int(x.split('@')[3].split(',')[cur_step]) for x in self.records]
    #     else:
    #         if self.config.get("support_conti_env", False):
    #             action = [self.action_emb[0], ] * self.batch_size
    #         else:
    #             action = [0, ] * self.batch_size
    #     return action

    # @property
    # def offline_reward(self):
    #     cur_step = self.cur_steps
    #     if cur_step < self.max_steps:
    #         reward = [0, ] * self.batch_size
    #     else:
    #         action = np.array([list(map(int, x.split('@')[3].split(','))) for x in self.records])
    #         price = self.get_price(action)
    #         slate_label = np.array([list(map(int, x.split('@')[4].split(','))) for x in self.records])
    #         reward = [sum([xx * yy for (xx, yy) in zip(x, y)]) for (x, y) in zip(price, slate_label)]
    #     return reward

    @property
    def info(self):
        return self.infos

    @staticmethod
    def get_nearest_neighbor(actions, action_emb, temperature=None):
        action_score = np.einsum('ij,kj->ik', np.array(actions), action_emb)
        best_action = np.argmax(action_score, axis=1)
        return best_action

    @staticmethod
    def get_nearest_neighbor_with_mask(actions, action_emb, action_mask, temperature=None):
        action_score = np.einsum('ij,kj->ik', np.array(actions), action_emb)
        action_score[action_mask < 0.5] = -2 ** 31
        best_action = np.argmax(action_score, axis=1)
        return best_action

    def act(self, actions):
        if self.config.get("support_conti_env", False):
            action_mask = self.action_mask
            actions = self.get_nearest_neighbor_with_mask(actions, self.action_emb, action_mask)
        self.prev_actions[:, self.cur_steps] = actions
        self.action_mask[list(range(self.batch_size)), actions] = 0
        tmp = copy(self._init_state)
        for state, action, i in zip(self._state, actions, range(len(actions))):
            # all players, exists players, matched players
            tmp[i][1][self.prev_actions[i][:self.cur_steps+1]] = -1
            tmp[i][2] = self.players[i][self.prev_actions[i]]
            tmp[i][2][self.prev_actions[i] == -1] = -1
        self._state = tmp
        self.cur_steps += 1

    def to_string(self):
        return '\n'.join(self.records)


class SimpleMatchRecEnv(RecSimBase):
    """ Implements core recommendation simulator"""

    def __init__(self, config, state_cls):
        self.max_steps = config['max_steps']
        assert self.max_steps == 2*config['num_per_team']*config['num_matches']
        self.batch_size = config['batch_size']
        self.num_per_team = config['num_per_team']
        self.num_matches = config['num_matches']
        self.upper_bound = config['upper_bound']
        # self.players_num = config['players_num']
        # self.FeatureUtil = FeatureUtil(config)
        super().__init__(config, state_cls)

    def get_model(self, config):
        # model_type = config.get('algo', 'dien')
        # model = __import__("enmatch.nets." + model_type, fromlist=['get_model']).get_model(config)
        # return model
        return None

    def obs_fn(self, state):
        if self.config.get("support_rllib_mask", False) or \
                self.config.get("support_d3rl_mask", False):
            obs = [np.ravel(x) for x in state["state"]]
        else:
            obs = [np.ravel(x) for x in state]
        if self.config.get("support_rllib_mask", False):
            action_mask = state["action_mask"]
            return [{
                "action_mask": action_mask[i],
                "obs": obs[i],
            } for i in range(self.batch_size)]
        elif self.config.get("support_d3rl_mask", False):
            masked_actions = state["masked_actions"]
            cur_steps = state["cur_steps"]
            return np.concatenate([np.array(obs), masked_actions, cur_steps], axis=-1)
        else:
            return np.array(obs)

    @staticmethod
    def obj_fn(matches, opt_obj = 0, reward_type='linear', use_abs=True):
        # config = self.config
        # reward_type = config.get('reward_type', 'linear')
        if reward_type == 'linear':
            team1 = matches[:,0,:]
            team2 = matches[:,1,:]
            if torch.is_tensor(matches):
                # reward = torch.zeros(1, device=matches.device)
                diff = torch.sum(team2,-1) - torch.sum(team1,-1)
                if use_abs:
                    diff = torch.abs(diff)
                # diff = torch.abs(team1[:,0] - team2[:,0])
                obj = diff
                # reward += torch.sum((opt_obj - diff))
                # reward += 1.0
            else:
                diff = np.sum(team2,-1) - np.sum(team1,-1)
                if use_abs:
                    diff = np.abs(diff)
                obj = diff
                # reward = (opt_obj - diff)
                # reward = 1.0 + reward
        elif reward_type == 'quadratic':
            team1 = matches[:,0,:]
            team2 = matches[:,1,:]
            if torch.is_tensor(matches):
                # reward = torch.zeros(1, device=matches.device)
                diff = torch.pow(torch.sum(team2,-1),2) -torch.pow(torch.sum(team1,-1),2)
                if use_abs:
                    diff = torch.abs(diff)
                obj = diff
                # reward += torch.sum((opt_obj - diff))
                # reward += 1.0
            else:
                diff = np.power(np.sum(team2,-1),2) -np.power(np.sum(team1,-1),2)
                if use_abs:
                    diff = np.abs(diff)
                obj = diff
                # reward = (opt_obj - diff)
                # reward = 1.0 + reward
        else:
            raise Exception
        return obj

    @staticmethod
    def obj2reward(obj, opt_obj=0, upper_bound=1):
        return 10.0-obj/upper_bound

    # @staticmethod
    # def reward2obj(reward, opt_obj=0, upper_bound=1):
    #     return -reward*upper_bound

    def forward(self, model, samples):
        step = samples.cur_steps
        reward_after_team = self.config.get('reward_after_team', True)
        if reward_after_team:
            if step < self.max_steps:
                obj = np.array([0] * self.batch_size)
                reward = np.array([0, ] * self.batch_size)
            else:
                # state = samples.state
                prev_actions = samples.prev_actions
                # macthes = np.reshape(prev_actions, (-1, 2, self.num_per_team))
                matches = samples.get_matches(prev_actions)
                obj = self.obj_fn(matches, samples.opt)
                reward = self.obj2reward(obj, samples.opt, self.upper_bound)
                # if 1:
                # # if self.config.get("support_rllib_mask", False) or \
                # #         self.config.get("support_d3rl_mask", False):
                #     violation = samples.get_violation()
                #     # reward[violation < 0.5] = 0
        elif not reward_after_team and step%2==0:
            # 规定team2>team1, 用cur_team2-cur_team1的相反数作为奖励，规定每一步都有team2>team1；怕出现team2<team1
            prev_actions = samples.prev_actions[:,step-2:step]
            # macthes = np.reshape(prev_actions, (-1, 2, self.num_per_team))
            matches = samples.get_matches(prev_actions)
            tmp_obj = self.obj_fn(matches, samples.opt, use_abs=False)
            reward = self.obj2reward(tmp_obj, samples.opt, self.upper_bound)
            # 规定每一步都有team2>team1
            prev_actions = samples.prev_actions[:,:step]
            matches = samples.get_matches(prev_actions)
            tmp_obj = self.obj_fn(matches, samples.opt, use_abs=False)
            reward[tmp_obj<0] = -100000
            if step < self.max_steps:
                obj = np.array([0] * self.batch_size)
            else:
                obj = self.obj_fn(matches, samples.opt)
        else:
            obj = np.array([0] * self.batch_size)
            reward = np.array([0, ] * self.batch_size)


        for i in range(self.batch_size):
            samples.info[i]['obj'] = obj[i].tolist()
            samples.info[i]['opt'] = samples.opt[i]
        return reward.tolist()

    @staticmethod
    def run_integer_program(players, num_per_team, num_teams = 2, max_time_in_seconds = 3600, obj_type='linear'):
        model = cp_model.CpModel()
        # const
        n = len(players)
        k = num_per_team
        player_zongping = players
        # variables
        x = []
        for i in range(n):
            t = []
            for j in range(num_teams):
                t.append(model.NewIntVar(0, 1, "x[%i,%i]" % (i, j)))
            x.append(t)

        task_zongping_diff = []
        for i in range(num_teams // 2):
            task_zongping_diff.append(model.NewIntVar(-int(1e7), int(1e7), "task_zongping_diff[%i]" % (i)))

        abs_diff = []
        for i in range(num_teams // 2):
            abs_diff.append(model.NewIntVar(0, int(1e8), "abs_diffs[%i]" % i))

        # k people
        [model.Add(sum(x[i][j] for i in range(n)) == k)
         for j in range(num_teams)]

        # at most one team
        [model.Add(sum(x[i][j] for j in range(num_teams)) <= 1)
         for i in range(n)]

        if obj_type == 'linear':
            # obj
            for j in range(0, num_teams, 2):
                model.Add(task_zongping_diff[j // 2] == sum(x[i][j] * player_zongping[i] - x[i][1 + j] * player_zongping[i] for i in range(n)))

            for j in range(num_teams // 2):
                model.AddAbsEquality(abs_diff[j], task_zongping_diff[j])
            model.Minimize(sum(abs_diff[j] for j in range(num_teams // 2)))
            # model.Maximize(sum(abs_diff[j] for j in range(env.instance.num_teams // 2)))
        else:
            # obj
            for j in range(0, num_teams, 2):
                model.Add(task_zongping_diff[j // 2] == sum(x[i][j] * player_zongping[i] - x[i][1 + j] * player_zongping[i] for i in range(n)))

            for j in range(num_teams // 2):
                model.AddMultiplicationEquality(abs_diff[j], [task_zongping_diff[j], task_zongping_diff[j]])
            model.Minimize(sum(abs_diff[j] for j in range(num_teams // 2)))

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8
        solver.parameters.max_time_in_seconds = max_time_in_seconds
        status = solver.Solve(model)
        teams = [[] for i in range(num_teams)]
        if status != cp_model.INFEASIBLE:
            opt_value = solver.ObjectiveValue()
            for i in range(n):
                for j in range(num_teams):
                    if solver.Value(x[i][j]) == 1:
                        teams[j].append(player_zongping[i])
            return np.array(teams).reshape((-1, 2, num_per_team)), int(opt_value)
        else:
            print('fail')
            raise Exception
            return ''

    @staticmethod
    def gen_opt(num_per_instance, num_per_team, instance_num, seed=1, range=(0,1000), obj_type='linear'):
        n = num_per_instance
        k = num_per_team
        instance_num = instance_num
        num_teams = n//k
        assert num_teams%2 == 0
        range = range
        np.random.seed(seed)
        tmp = []
        for i in range(instance_num):
            players = np.power(np.random.randint(range[0], range[1], n), 2)
            # mean = (range[0] + range[1])/2
            # std = (range[1] - range[0])/5
            # players = np.clip(np.random.normal(loc=mean, scale=std, size=n).astype('int'), range[0], range[1])
            # if not pre_sort:
            #     res = integer_program.run(players, k, num_teams = 2, max_time_in_seconds = 60, obj_type='linear')
            # else:
            #     players = sorted(players)
            if obj_type == 'linear':
                res = SimpleMatchRecEnv.run_integer_program(players, k, num_teams = num_teams, max_time_in_seconds = 6, obj_type='linear')
            else:
                res = SimpleMatchRecEnv.run_integer_program(players, k, num_teams = num_teams, max_time_in_seconds = 6, obj_type='quadratic')
            if len(res) > 1:
                players_str = ','.join(map(str,players))
                tmp.append(players_str+' '+json.dumps(res[0].tolist())+' '+str(res[1]))
            if len(tmp)%1000==0:
                print(len(tmp))

        with open('simplematch_%s_%s_%s_%s_%s_opt.csv'%(n, k, range[1], instance_num, obj_type), 'w') as f:
            f.write('\n'.join(tmp))




if __name__ == '__main__':
    # from encom.simplematch.environment import SimpleMatchEnv
    batch_size = 2
    num_per_team = 3
    num_matches = 1
    num_players = 2*num_matches*num_per_team
    rewards, objs = [], []
    config = {'sample_file':'../../dataset/simplematch_6_3_100_100_linear_opt.csv', 'batch_size':batch_size,'support_rllib_mask':True,
              'num_per_team':num_per_team, 'num_matches':num_matches, 'max_steps':num_players, 'upper_bound':100, 'action_size':num_players,
              'reward_after_team':False}
    recsim = SimpleMatchRecEnv(config = config, state_cls=SimpleMatchState)
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
        matches = np.array([[[0,1,2],[3,4,5]]])
        for match in matches:
            actions = match.reshape(-1)
            # actions = var2index(players, match.reshape(-1))
            assert len(actions) == 2*num_per_team
            for action in actions:
                obs, reward, done, info = env.step(action)
                print(obs)
                # print(action, players[action], info, reward)
                rewards.extend(reward)
                objs.extend([x['obj'] for x in info])
        env.reset()
    # print(rewards)
    print('instances:',num_instances,'avg rewards:',np.sum(rewards)/num_instances, 'avg objs:',np.sum(objs)/num_instances, sep=' ')