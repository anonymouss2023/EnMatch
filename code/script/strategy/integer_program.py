import json
import numpy as np
from script.strategy.utils import var2index, combinate, default_config
from ortools.sat.python import cp_model
import gym


def run(players, num_per_team, num_teams = 2, max_time_in_seconds = 3600, obj_type='linear'):
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

        # task_zongping_diff_quadratic = []
        # for i in range(num_teams // 2):
        #     task_zongping_diff_quadratic.append(model.NewIntVar(0, int(1e8), "task_zongping_diff_quadratic[%i]" % i))
        #
        # for i in range(num_teams // 2):
        #     model.AddMultiplicationEquality(task_zongping_diff_quadratic[i], [task_zongping_diff[i], task_zongping_diff[i]])

        for j in range(num_teams // 2):
            model.AddMultiplicationEquality(abs_diff[j], [task_zongping_diff[j], task_zongping_diff[j]])
        model.Minimize(sum(abs_diff[j] for j in range(num_teams // 2)))

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
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
    num_instances = 100
    for _ in range(num_instances):
        samples:SimpleMatchState = env.samples
        # raw_state = samples.records
        players = samples.players[0]
        matches, obj = run(players, num_per_team, num_teams= len(players)//num_per_team, max_time_in_seconds=2)
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
