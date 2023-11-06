import os
import time
import math

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# import matplotlib.pyplot as plt
# from IPython.display import clear_output

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class GraphEmbedding(nn.Module):

    def __init__(self, inp_dim, emb_dim):
        super(GraphEmbedding, self).__init__()
        self.emb_dim = emb_dim
        self.embedding = nn.Parameter(torch.FloatTensor(inp_dim, emb_dim))
        self.embedding.data.uniform_(-(1. / math.sqrt(emb_dim)), 1. / math.sqrt(emb_dim))

    def forward(self, x):
        """
        - input: x (N, F, L)
        - output: embedded (N, L, D)
        """
        batch_size, seq_len = x.size(0), x.size(2)
        embedding = self.embedding.repeat(batch_size, 1, 1)  # (N, F, D)
        embedded = []
        x = x.unsqueeze(1)  # (N, 1, F, L)
        for i in range(seq_len):
            embedded.append(torch.bmm(x[:, :, :, i].float(), embedding))  # (N, 1, D)
        embedded = torch.cat(embedded, 1)  # (N, L, D)
        return embedded


# class FloatEmbedding(nn.Module):
#     def __init__(self, L, emb_size):
#         super(FloatEmbedding, self).__init__()
#         self.linear = nn.Linear(1, 32)
#         # self.linear_2 = nn.Linear(32, 32)
#         self.leaky_relu = nn.LeakyReLU()
#         self.softmax = nn.Softmax(dim=2)
#         self.embedding = nn.Embedding(32, emb_size)
#
#     def forward(self, x):
#         N, L = x.size()
#         x = x.view(N, L, 1)
#         x = self.linear(x)
#         # x = self.linear_2(x)
#         x = self.leaky_relu(x)
#         x = x.view(N, L, 32)
#         x = self.softmax(x)
#         x = x.unsqueeze(-1) * self.embedding.weight
#         x = torch.sum(x, dim=2)
#         return x

class FloatEmbedding(nn.Module):
    def __init__(self, L, emb_size):
        super(FloatEmbedding, self).__init__()
        self.emb_size = emb_size
        self.linear = nn.Linear(1, emb_size)
        # self.linear_2 = nn.Linear(32, 32)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=2)
        self.embedding = nn.Embedding(emb_size, emb_size)

    def forward(self, x):
        x = x[:,0,:]
        N, L = x.size()
        x = x.view(N, L, 1)
        x = self.linear(x)
        # x = self.linear_2(x)
        x = self.leaky_relu(x)
        x = x.view(N, L, self.emb_size)
        x = self.softmax(x)
        x = x.unsqueeze(-1) * self.embedding.weight
        x = torch.sum(x, dim=2)
        return x


class Attention(nn.Module):
    
    def __init__(self, hid_dim, clipping=10, mode='Add'):
        super(Attention, self).__init__()
        self.clipping = clipping
        self.mode = mode
        if mode == 'Add':
            self.W_query = nn.Linear(hid_dim, hid_dim)
            self.W_ref = nn.Conv1d(hid_dim, hid_dim, 1, 1)
            self.v = nn.Parameter(torch.FloatTensor(hid_dim), requires_grad=True)
            self.v.data.uniform_(-(1. / math.sqrt(hid_dim)), 1. / math.sqrt(hid_dim))

    def forward(self, query, ref):
        """
        - input1: query (N, H)
        - input2: ref (N, L, H)
        """
        batch_size, seq_len = ref.size(0), ref.size(1)
        if self.mode == 'Add':
            ref = ref.permute(0, 2, 1)  # (N, H, L)
            query = self.W_query(query).unsqueeze(2)  # (N, H, 1)
            ref = self.W_ref(ref)  # (N, H, L)
            expanded_query = query.repeat(1, 1, seq_len)  # (N, H, L)
            v = self.v.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)  # (N, 1, H)
            logits = torch.bmm(v, torch.tanh(expanded_query + ref)).squeeze(1)  # (N, L)
        elif self.mode == 'Dot':
            query = query.unsqueeze(2)
            logits = torch.bmm(ref, query).squeeze(2)  # (N, L)
            ref = ref.permute(0, 2, 1)
        else:
            raise NotImplementedError
        if self.clipping is not None:
            logits = self.clipping * torch.tanh(logits)
        else:
            logits = logits
        return ref, logits


class PointerNet(nn.Module):

    def __init__(self, conf):
        super(PointerNet, self).__init__()
        self.seq_len = conf['seq_len']
        self.inp_dim = conf['input_dim']
        self.emb_dim = conf['embed_dim']
        self.input_emb_dim = conf['embed_dim'] - self.inp_dim
        self.hid_dim = conf['hidden_dim']
        self.att_mode = conf['att_mode']
        self.n_glimpses = conf['n_glimpses']
        self.clipping = conf['clipping']
        self.T = 1
        self.vocab_size = conf.get('vocab_size',10000.0)

        # categorical or numerical embedding
        # self.embedding = nn.Embedding(self.vocab_size, self.input_emb_dim)
        self.embedding = GraphEmbedding(self.inp_dim, self.emb_dim)
        # self.embedding = FloatEmbedding(self.seq_len, self.emb_dim).to(device)

        self.encoder = nn.LSTM(self.emb_dim, self.hid_dim, batch_first=True)
        self.decoder = nn.LSTM(self.emb_dim, self.hid_dim, batch_first=True)
        self.pointer = Attention(self.hid_dim, clipping=self.clipping, mode=self.att_mode)
        self.glimpse = Attention(self.hid_dim, clipping=False, mode=self.att_mode)

        self.decoder_inp0 = nn.Parameter(torch.FloatTensor(self.emb_dim))
        self.decoder_inp0.data.uniform_(-(1. / math.sqrt(self.emb_dim)), 1. / math.sqrt(self.emb_dim))

    def apply_mask_to_logits(self, logits, mask, indices):
        batch_size = logits.size(0)
        clone_mask = mask.clone()
        if indices is not None:
            clone_mask[[i for i in range(batch_size)], indices.data] = 1
            logits[clone_mask] = -np.inf
        return logits, clone_mask

    def forward(self, x):
        """
        - input: x (N, F, L)
        - output1: seq_probas (N, L) * L
        - output2: seq_indices (N,) * L
        """
        batch_size, seq_len = x.size(0), x.size(2)
        assert seq_len == self.seq_len

        # assert x.dim() == 3 and x.size(1) == 1
        # embedded = self.embedding(x.squeeze(1))  # (N, L, D)
        assert x.dim() == 3 and x.size(1) >= 1
        # x_tensor = torch.tensor(x.squeeze(1), dtype=torch.float, device=device)
        embedded = self.embedding(x/self.vocab_size)
        # embedded = self.embedding(x)
        # x_int = x.long()
        # x_float = torch.unsqueeze(x, dim=-1)
        # embedded = torch.cat([self.embedding(x_int), torch.FloatTensor(x_float)/self.vocab_size], dim=-1)   # (N, L, D)

        encoder_outputs, (hidden, cell) = self.encoder(embedded)  # (N, L, H), (1, N, H)

        mask = torch.zeros(batch_size, seq_len).bool()  # (N, L)
        mask = mask.to(device)

        seq_probas = []
        seq_indices = []
        indices = None

        decoder_input = self.decoder_inp0.unsqueeze(0).repeat(batch_size, 1)

        for _ in range(seq_len):

            _, (hidden, cell) = self.decoder(decoder_input.unsqueeze(1), (hidden, cell))

            query = hidden.squeeze(0)
            for _ in range(self.n_glimpses):
                ref, logits = self.glimpse(query, encoder_outputs)
                logits, mask = self.apply_mask_to_logits(logits, mask, indices)
                query = torch.bmm(ref, F.softmax(logits, dim=-1).unsqueeze(2)).squeeze(2)

            _, logits = self.pointer(query, encoder_outputs)
            logits, mask = self.apply_mask_to_logits(logits, mask, indices)
            probas = F.softmax(logits, dim=-1)
            adjusted_probas = probas.pow(1 / self.T) / probas.pow(1 / self.T).sum()
            # if self.T < 1:
            #     print(probas)
            #     print(adjusted_probas)
            #     # raise Exception
            indices = adjusted_probas.multinomial(num_samples=1).squeeze(1)
            for old_indices in seq_indices:
                if old_indices.eq(indices).data.any():
                    print('RESAMPLE!')
                    indices = probas.multinomial(num_samples=1).squeeze(1)
                    break
            decoder_input = embedded[[i for i in range(batch_size)], indices.data, :]

            seq_probas.append(probas)
            seq_indices.append(indices)

        return seq_probas, seq_indices


class Pipeline(nn.Module):

    beta = 0.1
    max_grad_norm = 2.
    
    def __init__(self, conf, model, env, reward_fn, upper_bound):
        super(Pipeline, self).__init__()
        self.model = model
        self.upper_bound = upper_bound
        # print(self.model.__class__.__name__.lower())
        self.model_name = type(self.model).__name__.lower()
        self.env = env
        self.reward_fn = reward_fn
        self.threshold = conf['threshold']
        self.optimizer = optim.Adam(model.parameters(), lr=conf['lr'])
        self.epochs = 0
        self.output_dir = conf.get('output_dir', './output')
        self.model_dir = os.path.join(self.output_dir, 'models')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)

    def forward(self, x):
        """
        - input: x (N, F, L)
        - output1: actions (N, F) * L
        - output2: action_probas (N,) * L
        - output3: indices (N,) * L
        """
        # (100, 2, 20)
        batch_size = x.size(0)
        # probas given k-1 points
        probas, indices = self.model(x)  # (N, L) * L, (N,) * L
        actions = []
        x = x.transpose(1, 2)  # (N, L, F)
        for action in indices:
            actions.append(x[[i for i in range(batch_size)], action.data, :])
        action_probas = []
        for proba, action in zip(probas, indices):
            action_probas.append(proba[[i for i in range(batch_size)], action.data])
        return actions, action_probas, indices

    def fit(self, train_ds, valid_ds, num_epochs=1, batch_size=128):
        r"""train and evaluate.
        """

        # BrokenPipeError: [Errno 32] Broken pipe
        # https://github.com/pytorch/pytorch/issues/2341
        # https://github.com/pytorch/pytorch/issues/12831
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        env_steps = 0  # a step corresponds to a episode
        total_steps = []

        steps_per_epoch = len(train_loader.dataset) // batch_size

        baseline = torch.zeros(1)
        baseline = baseline.to(device)

        for epoch in range(1, num_epochs + 1):
            train_return = []
            valid_return = []

            start_time = time.time()
            for step, batch in enumerate(train_loader):
                # batch = batch.repeat(batch_size, 1, 1)
                self.model.train()
                # print('input data', batch)
                states = Variable(batch)
                states = states.to(device)

                actions, action_probas, indices = self.forward(states)

                # batch_size, seq_len = states.size(0), states.size(2)
                env_steps += 1

                R = self.reward_fn(actions)
                R_norm = (R/self.upper_bound).to(device)

                if step == 0:
                    baseline = R_norm.mean()
                else:
                    # Exponential moving average
                    baseline = (baseline * self.beta) + ((1. - self.beta) * R_norm.mean())
                baseline = baseline.detach()
                advantage = R_norm - baseline

                log_probas = 0
                for probas in action_probas:
                    log_probas += torch.log(probas)
                log_probas[log_probas < -1000] = 0.

                reinforce = advantage * log_probas
                # print(reinforce)
                # raise Exception
                loss = reinforce.mean()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.max_grad_norm), norm_type=2)
                self.optimizer.step()

                train_return.append(R.mean().item())

                if step % 100 == 0:
                    self.model.T = 0.05
                    R = self.evaluate(valid_loader)
                    self.model.T = 1
                    valid_return.append(R)
                    total_steps.append(env_steps)
                    print('| epoch {:3d}/{:d} | step {:5d}/{:d} | train {:5.4f} | valid {:5.4f} |'.format(
                        epoch, num_epochs, step, steps_per_epoch, train_return[-1], valid_return[-1]))

            if self.threshold and valid_return[-1] < self.threshold:
                print("EARLY STOPPAGE!")
                break

            self.epochs = epoch
            elapsed = time.time() - start_time
            print('-' * 80)
            print('| epoch {:3d}/{:d} | env steps {:d} | train {:5.4f} | valid {:5.4f} | time {:5.2f}s |'.format(
                epoch, num_epochs, env_steps, np.mean(train_return), valid_return[-1], elapsed))
            print('-' * 80)
            self.save_model()

        return {
            'train return': train_return,
            'valid return': valid_return,
            'total steps': total_steps
        }

    def evaluate(self, test_loader):
        self.model.eval()
        reward = []
        for batch in test_loader:
            states = Variable(batch)
            states = states.to(device)
            actions, action_probas, indices = self.forward(states)
            R = self.reward_fn(actions)
            reward.append(R.mean().item())
        return np.array(reward).mean()

    def save_model(self, path=None):
        if path is not None:
            if path[-3:] != '.pt' and path[-4:] != '.pth':
                path = path + '.pt'
        else:
            file_name = self.model_name + '-' + str(self.epochs) + '.pt'
            path = os.path.join(self.model_dir, file_name)
        torch.save(self.model.state_dict(), path)
        print('save model to ', path)

    def load_model(self, path=None):
        if path is not None:
            if path[-3:] != '.pt' and path[-4:] != '.pth':
                path = path + '.pt'
        else:
            file_name = self.model_name + '-' + str(self.epochs) + '.pt'
            path = os.path.join(self.model_dir, file_name)
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def plot(self, epoch):
        pass


if __name__ == "__main__":

    # from config import conf
    from enmatch.nets.pointernet.tsp_env import TSPDataset, TSPEnv, reward_fn

    conf = {'clipping': 10,}
    conf['num_seeds'] = 1  # number of experimental seeds
    conf['device'] = 'cpu'

    conf['num_nodes'] = 20
    conf['num_features'] = 2
    conf['seq_len'] = conf['num_nodes']
    conf['threshold'] = 0.0

    conf['train_size'] = int(1e2)
    conf['valid_size'] = int(1e2)

    conf['input_dim'] = conf['num_features']
    conf['embed_dim'] = 128
    conf['hidden_dim'] = 128
    conf['att_mode'] = "Dot"
    conf['n_glimpses'] = 1

    conf['num_epochs'] = 1
    conf['batch_size'] = 16
    conf['lr'] = 1e-4

    for k, v in conf.items():
        print(k, '->', v)

    # (train_size, num_features, num_nodes)
    train_ds = TSPDataset(conf['num_nodes'], conf['train_size'])
    valid_ds = TSPDataset(conf['num_nodes'], conf['valid_size'])

    model = PointerNet(conf).to(device)
    # env = TSPEnv(conf)
    env = None
    pl = Pipeline(conf, model, env, reward_fn=reward_fn)

    pl.fit(train_ds, valid_ds, num_epochs=10, batch_size=128)
