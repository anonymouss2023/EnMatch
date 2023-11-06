from enmatch.nets.pointernet.mm_reinforce import Pipeline, PointerNet
from enmatch.env.simpleMatch import SimpleMatchRecEnv
import numpy as np
from tqdm import tqdm
import torch
import sys
from script.strategy.utils import var2index, combinate, default_config
from torch.utils.data import Dataset

device = torch.device('cpu')
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()

class MMDataset(Dataset):

    def __init__(self, data):
        super(MMDataset, self).__init__()
        data = np.array(data)
        indices = np.argsort(data, axis=1) + 1
        x = data.shape
        data = np.concatenate([data, indices], axis=1)
        self.data = torch.FloatTensor(data).reshape((x[0],2,x[1]))
        # print(self.data[0])
        self.size = len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.size


def obj_fn(a, config):
    r"""Return/Reward function.
    - input: a list; player_num * (BatchSize, Feature)
    - output reward (N,)
    """
    n = len(a)
    batch_size = a[0].size(0)
    feature_size = a[0].size(1)
    num_matches = config['num_matches']
    num_per_team = config['num_per_team']
    # a = .reshape((num_matches, 2*num_per_team, batch_size, feature_size))
    reward = torch.zeros([num_matches, batch_size], device=a[0].device)
    for i in range(num_matches):
        ind = i*2*num_per_team
        for j in range(num_per_team):
            reward[i] += (a[j+ind] - a[ind+2*num_per_team-j-1])[:,0]
    # for i in range(n//2):
    #     reward += (a[2*i] - a[2*i + 1])[:,0]
    # reward = torch.abs(reward)

    rewards = torch.abs(reward)
    # # player_num, batch_size, feature_num = a.size()
    # # print(a)
    # # raise Exception
    # a = torch.stack(a, dim=0)
    # player_num = len(a)
    # batch_size = a[0].size(0)
    # num_per_team = config['num_per_team']
    # # 0-th dim of feature_num
    # # matches = torch.reshape(a[:,:,0], shape=(batch_size, -1, num_per_team, 2))
    # # matches = torch.transpose(matches,-1,-2)
    # matches = torch.reshape(a[:,:,0], shape=(batch_size, -1, 2, num_per_team))
    # rewards = torch.zeros([batch_size,1], device=a[0].device)
    # for i in range(batch_size):
    #     rewards[i] += torch.sum(SimpleMatchRecEnv.obj_fn(matches[i], reward_type = config.get('reward_type', 'linear')))
    return torch.sum(rewards, dim=0)

if __name__ == '__main__':
    extra_config = eval(sys.argv[1]) if len(sys.argv) >= 2 else {}

    conf = {'clipping': 10,}
    conf['num_seeds'] = 1  # number of experimental seeds
    conf['device'] = device
    conf['threshold'] = 0.0
    conf['train_size'] = int(4e4)
    conf['valid_size'] = int(1e4)
    conf['embed_dim'] = 128
    conf['hidden_dim'] = 128
    conf['att_mode'] = "Dot"
    conf['n_glimpses'] = 1
    conf['num_epochs'] = 100
    conf['batch_size'] = 128
    conf['lr'] = 1e-4

    conf['num_matches'] = 1
    conf['num_per_team'] = 3
    conf['reward_type'] = 'linear'
    conf['train_file'] = '../../dataset/simplematch_6_3_100_100000_linear_opt.csv.2'
    conf['valid_file'] = '../../dataset/simplematch_6_3_100_100000_linear_opt.csv.2'

    conf = {**conf, **extra_config}

    conf['num_nodes'] = conf['num_matches'] * conf['num_per_team'] * 2
    conf['num_features'] = 2
    conf['seq_len'] = conf['num_nodes']
    conf['input_dim'] = conf['num_features']
    conf['vocab_size'] = 10000 if conf['num_per_team']<5 else 1000000

    for k, v in conf.items():
        print(k, '->', v)

    train_data = open(conf['train_file']).read().split('\n')[:-1]
    train_data = train_data[: max(len(train_data),conf['train_size'])]
    valid_data = open(conf['valid_file']).read().split('\n')[:-1][:conf['valid_size']]
    valid_data = valid_data[: max(len(valid_data),conf['valid_size'])]
    train_data = [list(map(int,x.split('@')[0].split(','))) for x in train_data]
    valid_data = [list(map(int,x.split('@')[0].split(','))) for x in valid_data]

    # (train_size, num_features, num_nodes)
    train_ds = MMDataset(train_data)
    valid_ds = MMDataset(valid_data)

    model = PointerNet(conf).to(conf['device'])
    # env = TSPEnv(conf)
    env = None
    pl = Pipeline(conf, model, env, reward_fn=lambda v: obj_fn(v, conf), upper_bound=5000.0)

    pl.fit(train_ds, train_ds, num_epochs=conf['num_epochs'], batch_size=conf['batch_size'])