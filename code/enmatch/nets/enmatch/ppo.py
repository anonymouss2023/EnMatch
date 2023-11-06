import random
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from enmatch.env.simpleMatch import SimpleMatchRecEnv

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

from enmatch.nets.enmatch.enmatch import FloatEmbedding, Attention
import math
import torch.nn.functional as F
import numpy as np
from typing import Optional, Any
from torch import Tensor


class EnNet(nn.Module):

    def __init__(self, conf):
        super(EnNet, self).__init__()
        self.conf = conf
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
        self.num_per_team = conf['num_per_team']
        self.num_matches = conf['num_matches']
        self.num_players = 2*self.num_per_team*self.num_matches

        # categorical or numerical embedding
        # self.embedding = nn.Embedding(self.vocab_size, self.input_emb_dim)
        # self.embedding = GraphEmbedding(self.inp_dim, self.emb_dim)
        self.embedding = FloatEmbedding(self.seq_len, self.emb_dim).to(device)
        assert self.emb_dim == self.hid_dim
        # self.encoder = nn.LSTM(self.emb_dim, self.hid_dim, batch_first=True)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=8, dim_feedforward=256)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        # self.decoder = nn.LSTM(self.emb_dim, self.hid_dim, batch_first=True)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.emb_dim, nhead=8, dim_feedforward=256)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        self.pointer = Attention(self.hid_dim, clipping=self.clipping, mode='Dot')
        # self.glimpse = Attention(self.hid_dim, clipping=False, mode=self.att_mode)

        # self.decoder_inp0 = nn.Parameter(torch.FloatTensor(self.emb_dim))
        self.trg_tensor = nn.Parameter(torch.zeros((2*self.num_per_team, self.emb_dim)))
        # self.decoder_inp0.data.uniform_(-(1. / math.sqrt(self.emb_dim)), 1. / math.sqrt(self.emb_dim))
        self.shuffled_indices_list = None

    def destory_repair_index(self, N, p):
        # N = 10
        # p = 0.2
        lst = list(range(N))

        shuffled_indices = set(random.sample(range(len(lst)), int(N*p)))
        # shuffled_indices = set(range(len(lst))) - fixed_indices
        shuffled_indices = dict(zip(shuffled_indices,random.sample(shuffled_indices, len(shuffled_indices))))
        # new_indices = sorted(list(fixed_indices)) + shuffled_indices
        new_lst = [j if j not in shuffled_indices else shuffled_indices[j]  for j in range(N)]
        return new_lst

    def heuristic_op(self, config, actions, indices, obj_fn, k, destory_p=[0.5]):
        """
        - input: actions (L, N, F)
        - output1: actions (k, L, N, F)
        """
        device = actions.device
        k = k//len(destory_p)*len(destory_p)
        num_matches = 1
        num_per_team = config['num_per_team']
        N, batch_size = actions.shape
        actions = actions.view(-1, 2 * num_per_team,  batch_size)
        indices = indices.view(-1, 2 * num_per_team,  batch_size)
        if self.shuffled_indices_list is None:
            shuffled_indices_list = []
            for p in destory_p:
                for _ in range(k//len(destory_p)):
                    shuffled_indices_list.append(self.destory_repair_index(2 * config['num_per_team'], p))
            self.shuffled_indices_list = torch.LongTensor(shuffled_indices_list).to(device).reshape((k,-1))
        new_actions_list = []
        # (k, num_matches, 2 * num_per_team,  batch_size) (k, num_matches, 2 * num_per_team, batch_size)
        k_actions = actions.unsqueeze(0).repeat(k,1,1,1).gather(dim=2, index=self.shuffled_indices_list[:,None,:,None].expand(-1,num_matches,-1,batch_size))
        k_indices = indices.unsqueeze(0).repeat(k,1,1,1).gather(dim=2, index=self.shuffled_indices_list[:,None,:,None].expand(-1,num_matches,-1,batch_size))
        # k_actions = new_actions
        # (k, num_matches, 2 * num_per_match, batch_size, F)
        # (batch_size, k, num_matches, 2 * num_per_match, F)
        y = k_actions.permute(3, 0, 1, 2).reshape((batch_size*k*num_matches, 2*num_per_team))
        y_obj = [obj_fn(match.reshape(-1,num_per_team,2).permute(0,2,1)) for match in y]
        y_obj = torch.stack(y_obj, dim=0)
        y_obj = y_obj.reshape((batch_size, k, num_matches))
        # (batch_size, k, num_matches, F)
        a, a_min = y_obj[:, :, :].min(dim=1) # (batch_size, num_matches)
        ori_actions = k_actions.permute(3, 0, 1, 2).reshape((batch_size, k, num_matches, 2 * num_per_team))
        ori_indices = k_indices.permute(3, 0, 1, 2).reshape((batch_size, k, num_matches, 2 * num_per_team))
        best_action = torch.gather(ori_actions, dim=1, index=a_min[:, None,:, None].expand(-1, 1, -1, ori_actions.size(3))).squeeze(1)
        best_indices = torch.gather(ori_indices, dim=1, index=a_min[:, None,:, None].expand(-1, 1, -1, ori_actions.size(3))).squeeze(1)
        return best_action.permute(1,2,0).reshape(-1,batch_size), best_indices.permute(1,2,0).reshape(-1,batch_size)

    def apply_mask_to_logits(self, logits, mask, indices):
        batch_size = logits.size(0)
        clone_mask = mask.clone()
        if indices is not None:
            clone_mask[[i for i in range(batch_size)], indices.data] = 1
        logits[clone_mask] = -np.inf
        return logits, clone_mask

    def forward(self, x):
        """
        - input: x (N, obs_size)
        - output1: actions (N, F) * L
        - output2: action_probas (N,) * L
        - output3: indices (N,) * L
        """
        # (100, 2, 20)
        batch_size = x.size(0)
        # -> (N, F, L)
        # obs (self.num_players*3,): all players, exists players, matched players
        pool = x[:, self.num_players:2*self.num_players].clone()
        mask = torch.zeros(batch_size, self.num_players).bool().to(device)  # (N, L)
        mask.masked_fill_(pool == -1, 1)
        mask = mask
        pool = torch.reshape(pool, (batch_size,1,self.num_players))
        assert pool.dim() == 3 and pool.size(1) >= 1
        # seq_len = self.seq_len
        # assert seq_len == self.seq_len
        # probas given k-1 points
        probas, indices = self.forward_seq(pool, mask)  # (N, L) * L, (N,) * L
        actions = []
        players = pool.transpose(1, 2)[:,:,0]  # (N, L)
        for action in indices:
            actions.append(players[[i for i in range(batch_size)], action.data])
        action_probas = []
        for proba, action in zip(probas, indices):
            action_probas.append(proba[[i for i in range(batch_size)], action.data])
        actions = torch.stack(actions, dim=0)
        indices = torch.stack(indices, dim=0)
        heuristic_actions, heuristic_indices = self.heuristic_op(self.conf, actions, indices, SimpleMatchRecEnv.obj_fn, k=max(100, pow(self.num_per_team,3)), destory_p=[0.4, 0.8])
        return [actions, heuristic_actions], action_probas, [indices, heuristic_indices]

    def forward_seq(self, x, mask):
        """
        - input: x (N, F, L)
        - output1: seq_probas (N, L) * L
        - output2: seq_indices (N,) * L
        """
        batch_size, seq_len = x.size(0), self.seq_len
        # assert x.dim() == 3 and x.size(1) == 1
        # embedded = self.embedding(x.squeeze(1))  # (N, L, D)

        # x_tensor = torch.tensor(x.squeeze(1), dtype=torch.float, device=device)
        # (N, F, L)
        embedded = self.embedding(x/self.vocab_size)
        # (L, N, H)
        embedded = embedded.permute(1, 0, 2)
        # embedded = self.embedding(x)
        # x_int = x.long()
        # x_float = torch.unsqueeze(x, dim=-1)
        # embedded = torch.cat([self.embedding(x_int), torch.FloatTensor(x_float)/self.vocab_size], dim=-1)   # (N, L, D)

        # encoder_outputs, (hidden, cell) = self.encoder(embedded)  # (N, L, H), (1, N, H)
        encoder_outputs = self.encoder(embedded)

        seq_probas = []
        seq_indices = []
        indices = None
        trg_input = self.trg_tensor.unsqueeze(1).repeat(1, batch_size, 1)  # (L, N, H)

        for i in range(seq_len):
            # _, (hidden, cell) = self.decoder(decoder_input.unsqueeze(1), (hidden, cell))
            query = self.decoder(encoder_outputs, trg_input.clone())[-1]
            _, logits = self.pointer(query, encoder_outputs.permute(1, 0, 2))
            logits, mask = self.apply_mask_to_logits(logits, mask, indices)
            probas = F.softmax(logits, dim=-1)
            indices = probas.multinomial(num_samples=1).squeeze(1)
            trg_input[i] = embedded[indices.data, [i for i in range(batch_size)], :]
            seq_probas.append(probas)
            seq_indices.append(indices)

        # decoder_input = self.decoder_inp0.unsqueeze(0).repeat(batch_size, 1)
        #
        # for _ in range(seq_len):
        #
        #     _, (hidden, cell) = self.decoder(decoder_input.unsqueeze(1), (hidden, cell))
        #
        #     query = hidden.squeeze(0)
        #     for _ in range(self.n_glimpses):
        #         ref, logits = self.glimpse(query, encoder_outputs)
        #         logits, mask = self.apply_mask_to_logits(logits, mask, indices)
        #         query = torch.bmm(ref, F.softmax(logits, dim=-1).unsqueeze(2)).squeeze(2)
        #
        #     _, logits = self.pointer(query, encoder_outputs)
        #     logits, mask = self.apply_mask_to_logits(logits, mask, indices)
        #     probas = F.softmax(logits, dim=-1)
        #     adjusted_probas = probas.pow(1 / self.T) / probas.pow(1 / self.T).sum()
        #     indices = adjusted_probas.multinomial(num_samples=1).squeeze(1)
        #     for old_indices in seq_indices:
        #         if old_indices.eq(indices).data.any():
        #             print('RESAMPLE!')
        #             indices = probas.multinomial(num_samples=1).squeeze(1)
        #             break
        #     decoder_input = embedded[[i for i in range(batch_size)], indices.data, :]
        #
        #     seq_probas.append(probas)
        #     seq_indices.append(indices)

        return seq_probas, seq_indices


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init, conf):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh()
            )
        else:
            self.actor = EnNet(conf)
            # self.actor = nn.Sequential(
            #     nn.Linear(state_dim, 64),
            #     nn.Tanh(),
            #     nn.Linear(64, 64),
            #     nn.Tanh(),
            #     nn.Linear(64, action_dim),
            #     nn.Softmax(dim=-1)
            # )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state, action_mask=None):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            state_val = self.critic(state)
            return action.detach(), action_logprob.detach(), state_val.detach()
        else:
            [action, heuristic_action], action_probas, [indices, heuristic_indices] = self.actor(state)
            action_logprob = 0
            for probas in action_probas:
                action_logprob += torch.log(probas)
            action_logprob[action_logprob < -1000] = 0.
            state_val = self.critic(state)
            # action:list sum(action_logprob)
            return [indices, heuristic_indices], action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
            action_logprob = dist.log_prob(action)
            dist_entropy = dist.entropy()
            state_val = self.critic(state)
        else:
            # action_probs = self.actor(state)
            [action, heuristic_action], action_probas, [indices, heuristic_indices] = self.actor(state)
            action_logprob = 0
            for probas in action_probas:
                action_logprob += torch.log(probas)
            action_logprob[action_logprob < -1000] = 0.
            state_val = self.critic(state)
            # dist = Categorical(action_probs)
            # dist_entropy = dist.entropy()
            dist_entropy = 0.5

        return action_logprob, state_val, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6, conf={}):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, conf).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, conf).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                [action, heuristic_action], action_logprob, state_val = self.policy_old.act(state)

            for i in range(int(state.size(0))):
                self.buffer.states.append(state[i])
                self.buffer.actions.append(action[i])
                self.buffer.logprobs.append(action_logprob[i])
                self.buffer.state_values.append(state_val[i])

            return action.detach().cpu().numpy().flatten(), heuristic_action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                if isinstance(state[0], dict):
                    obs = [x['obs'] for x in state]
                    obs = torch.FloatTensor(obs).to(device)
                    action_mask = [x['action_mask'] for x in state]
                    action_mask = torch.FloatTensor(action_mask).to(device)
                    [action, heuristic_action], action_logprob, state_val = self.policy_old.act(obs, action_mask)
                else:
                    obs = torch.FloatTensor(state).to(device)
                    [action, heuristic_action], action_logprob, state_val = self.policy_old.act(obs)
            action_t = torch.transpose(action,0,1)
            for i in range(int(obs.size(0))):
                self.buffer.states.append(obs[i])
                self.buffer.actions.append(action_t[i])
                self.buffer.logprobs.append(action_logprob[i])
                self.buffer.state_values.append(state_val[i])

            return action.detach().cpu().numpy().astype(np.int64), heuristic_action.detach().cpu().numpy().astype(np.int64)

    def update(self, max_ep_len, batch_size):
        buffer_rewards = np.swapaxes(np.array(self.buffer.rewards).reshape((-1, max_ep_len, batch_size)),1,2).flatten()
        buffer_is_terminals = np.swapaxes(np.array(self.buffer.is_terminals).reshape((-1, max_ep_len, batch_size)),1,2).flatten()
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(buffer_rewards), reversed(buffer_is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = np.swapaxes(np.array(rewards).reshape((-1,batch_size,max_ep_len)), 1, 2).flatten()
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))




