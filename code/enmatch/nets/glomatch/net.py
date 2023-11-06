from gym.spaces import Box
import gym
import numpy as np
from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, \
    normc_initializer
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_ops import FLOAT_MIN, FLOAT_MAX
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math




tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class MMPointer(nn.Module):

    def __init__(self, conf):
        super(MMPointer, self).__init__()

        # self.team_size = conf['team_size']
        inp_dim = conf['num_features']
        # out_dim = conf.get('num_classes', 1)
        hid_dim = conf.get('hidden_dim', 128)
        ffn_dim = conf.get('ffn_dim', hid_dim * 4)
        num_heads = conf.get('num_heads', 1)
        num_layers = conf.get('num_layers', 1)
        self.dropout = conf.get('dropout', 0.1)
        self.clip = conf.get('clip', 10)
        # self.device = conf.get('device', 'cpu')

        self.fc = nn.Linear(inp_dim, hid_dim)
        encoder_layer = TransformerEncoderLayer(hid_dim, num_heads, ffn_dim, self.dropout)
        self.team_encoder = TransformerEncoder(encoder_layer, num_layers)
        encoder_layer = TransformerEncoderLayer(hid_dim, num_heads, ffn_dim, self.dropout)
        self.pool_encoder = TransformerEncoder(encoder_layer, num_layers)

        # self.fc = nn.Linear(2 * self.team_size * self.hid_dim, 1)
        self.w1 = nn.Linear(3 * hid_dim, hid_dim)
        self.w2 = nn.Conv1d(hid_dim, hid_dim, 1, 1)
        self.v = nn.Parameter(torch.FloatTensor(hid_dim), requires_grad=True)
        self.v.data.uniform_(-(1. / math.sqrt(hid_dim)), 1. / math.sqrt(hid_dim))

    def forward(self, x):
        x_team, x_pool = x  # (N, 2T, D), (N, L, D)
        # print('x_team', x_team[0],x_pool[0])
        batch_size, team_size = x_team.size(0), int(x_team.size(1) / 2)
        batch_size, pool_size = x_pool.size(0), x_pool.size(1)

        x_team = x_team.permute(1, 0, 2)  # (2T, N, D)
        x_team = self.fc(x_team)  # (2T, N, H)
        # x1, x2 = x_team[:team_size], x_team[team_size:]
        # x1, x2 = torch.split(x_team, [team_size, team_size], dim=0)
        idx = np.arange(2 * team_size)
        idx1 = (idx % 2 == 0)
        idx2 = np.logical_not(idx1)
        x1, x2 = x_team[idx1], x_team[idx2]

        x1 = self.team_encoder(x1, mask=None)  # (T, N, H)
        h1 = torch.mean(x1, dim=0)  # (N, H)
        x2 = self.team_encoder(x2, mask=None)  # (T, N, H)
        h2 = torch.mean(x2, dim=0)  # (N, H)

        x_pool = x_pool.permute(1, 0, 2)  # (L, N, D)
        x_pool = self.fc(x_pool)  # (L, N, H)
        x_pool = self.pool_encoder(x_pool, mask=None)  # (L, N, H)
        h_pool = torch.mean(x_pool, dim=0)  # (N, H)

        query = torch.cat([h1, h2, h_pool], dim=1)  # (N, 3H)
        state = query  # (N, 3H)

        reference = x_pool.permute(1, 2, 0)  # (N, H, L)
        query = self.w1(query)  # (N, H)
        reference = self.w2(reference)  # (N, H, L)
        query = query.unsqueeze(2).repeat(1, 1, pool_size)  # (N, H, L)
        v = self.v.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)  # (N, 1, H)
        # print(query.shape, reference.shape, v.shape)

        logits = torch.bmm(v, torch.tanh(query + reference)).squeeze(1)  # (N, L)
        if self.clip:
            logits = self.clip * torch.tanh(logits)

        return logits, state

    def predict(self, x):
        pass


class PolicyNetwork(TorchModelV2, nn.Module):
    """The policy network."""

    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        # custom_conf = model_config.get("custom_model_config", {})
        activation = model_config.get("fcnet_activation")
        hiddens = model_config.get("fcnet_hiddens", [])
        no_final_linear = model_config.get("no_final_linear")
        self.model_config = model_config
        # self.num_features = model_config.get("num_features")
        self.vf_share_layers = model_config.get("vf_share_layers")
        self.free_log_std = model_config.get("free_log_std")

        # Generate free-floating bias variables for the second half of
        # the outputs.
        if self.free_log_std:
            pass

        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None

        # Create layers 0 to second-last.
        for size in hiddens[:-1]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation))
            prev_layer_size = size

        # The last layer is adjusted to be of size num_outputs, but it's a
        # layer with activation.
        if no_final_linear and num_outputs:
            pass
        # Finish the layers with the provided sizes (`hiddens`), plus -
        # iff num_outputs > 0 - a last linear layer of size num_outputs.
        else:
            if len(hiddens) > 0:
                layers.append(
                    SlimFC(
                        in_size=prev_layer_size,
                        out_size=hiddens[-1],
                        initializer=normc_initializer(1.0),
                        activation_fn=activation))
                prev_layer_size = hiddens[-1]
            if num_outputs:
                self._logits = SlimFC(
                    in_size=prev_layer_size,
                    out_size=num_outputs,
                    initializer=normc_initializer(0.01),
                    activation_fn=None)
            else:
                self.num_outputs = (
                        [int(np.product(obs_space.shape))] + hiddens[-1:])[-1]

        # Layer to add the log std vars to the state-dependent means.
        if self.free_log_std and self._logits:
            pass

        # self._hidden_layers = nn.Sequential(*layers)

        # self._hidden_layers = TSPPointer(custom_conf)
        # prev_layer_size = 2 * custom_conf['hidden_dim']

        self._hidden_layers = MMPointer(self.model_config)
        prev_layer_size = 3 * self.model_config['hidden_dim']

        self._value_branch_separate = None
        if not self.vf_share_layers:
            # Build a parallel set of hidden layers for the value net.
            pass

        self._value_branch = SlimFC(
            in_size=prev_layer_size,
            out_size=1,
            initializer=normc_initializer(1.0),
            activation_fn=None)
        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        # obs = input_dict["obs_flat"].float()
        # self._last_flat_in = obs.reshape(obs.shape[0], -1)
        # self._features = self._hidden_layers(self._last_flat_in)
        # logits = self._logits(self._features) if self._logits else self._features
        obs = input_dict["obs"].float()
        # tour = input_dict["tour"].float()
        # pool = input_dict["pool"].float()
        # # print(obs.shape, tour.shape, pool.shape)
        # self._last_flat_in = (tour, pool)
        team = input_dict["team"].float()
        pool = input_dict["pool"].float()
        # print(obs.shape, team.shape, pool.shape, 'shape')
        self._last_flat_in = (team, pool)
        logits, self._features = self._hidden_layers(self._last_flat_in)
        # print(logits.shape, self._features.shape)
        if self.free_log_std:
            pass
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            pass
        else:
            return self._value_branch(self._features).squeeze(1)


class TorchMMModel(DQNTorchModel):
    """PyTorch version of above TFTSPModel."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 true_obs_shape=(126,),  # (2 * self.team_size + self.num_players) * self.num_features
                 **kw):
        DQNTorchModel.__init__(self, obs_space, action_space, num_outputs,
                               model_config, name, **kw)

        self.action_embed_model = TorchFC(
            Box(0, 1, shape=true_obs_shape), action_space, num_outputs,
            model_config, name + "_action_embed")

        print(obs_space, action_space, num_outputs, model_config, name, true_obs_shape)

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["mask"]

        # Compute the predicted action embedding [BATCH, MAX_ACTIONS]
        action_logits, _ = self.action_embed_model({
            "obs": input_dict["obs"]["state"]
        })

        # Mask out invalid actions (use -inf to tag invalid).
        # These are then recognized by the EpsilonGreedy exploration component
        # as invalid actions that are not to be chosen.
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()

class GloModel(DQNTorchModel):
    """PyTorch version of above TFMMModel."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 true_obs_shape,
                 config,
                 # true_obs_shape=((120 + 2 * 15) * 36,),  # (120 + 2 * 3) * 19 for ball and (120 + 2 * 15) * 36 for nsh
                 **kw):
        # self.true_obs_shape = kw['true_obs_shape']
        self.true_obs_shape = true_obs_shape
        model_config = {**model_config, **config}
        # print('num_features', model_config['num_features'])
        DQNTorchModel.__init__(self, obs_space, action_space, num_outputs,
                               model_config, name, **kw)

        # self.action_embed_model = TorchFC(
        #     Box(0, 1, shape=true_obs_shape), action_space, num_outputs,
        #     model_config, name + "_action_embed")

        self.action_embed_model = PolicyNetwork(
            Box(0, 1, shape=true_obs_shape), action_space, num_outputs,
            model_config, name + "_action_embed")

        print(obs_space, action_space, num_outputs, model_config, name, true_obs_shape)

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]
        # Compute the predicted action embedding [BATCH, MAX_ACTIONS]
        # all players N, exists players N, matched players N
        obs_norm = self.model_config['obs_norm']
        num_per_team = self.model_config['num_per_team']
        num_matches = self.model_config['num_matches']
        num_players = num_per_team * num_matches * 2
        # N = input_dict["obs"]["obs"].size(1)//3
        N = num_players
        k = (action_mask[0] != 1).sum().item() // (num_per_team*2)
        start = min(num_per_team*2*k, N-num_per_team*2)+ 2*N
        end = start + num_per_team*2
        # print('N', N, input_dict["obs"]["obs"].shape,k, sep=' ')
        # print('input_dict')
        # print(k, start,end,action_mask[0],input_dict["obs"]["obs"][0],input_dict["obs"]["obs"][:,start:end].unsqueeze(-1)[0]/obs_norm, input_dict["obs"]["obs"][:,N:2*N][0].unsqueeze(-1)/obs_norm)
        # print('input_dict', input_dict["obs"]["obs"][0])
        action_logits, _ = self.action_embed_model({
            "obs": input_dict["obs"]["obs"].unsqueeze(-1)/obs_norm,
            "team": input_dict["obs"]["obs"][:,start:end].unsqueeze(-1)/obs_norm,
            "pool": input_dict["obs"]["obs"][:,N:2*N].unsqueeze(-1)/obs_norm
        })

        # Mask out invalid actions (use -inf to tag invalid).
        # These are then recognized by the EpsilonGreedy exploration component
        # as invalid actions that are not to be chosen.
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()


class GloModelTest(DQNTorchModel):
    """PyTorch version of above TFMMModel."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 true_obs_shape,
                 config,
                 # true_obs_shape=((120 + 2 * 15) * 36,),  # (120 + 2 * 3) * 19 for ball and (120 + 2 * 15) * 36 for nsh
                 **kw):
        # self.true_obs_shape = kw['true_obs_shape']
        self.true_obs_shape = true_obs_shape
        self.model_config = {**model_config, **config}
        # print('num_features', model_config['num_features'])
        DQNTorchModel.__init__(self, obs_space, action_space, num_outputs,
                               self.model_config, name, **kw)

        # self.action_embed_model = TorchFC(
        #     Box(0, 1, shape=true_obs_shape), action_space, num_outputs,
        #     model_config, name + "_action_embed")

        self.action_embed_model = PolicyNetwork(
            Box(0, 1, shape=true_obs_shape), action_space, num_outputs,
            self.model_config, name + "_action_embed")

        print(obs_space, action_space, num_outputs, model_config, name, true_obs_shape)

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["mask"]

        # Compute the predicted action embedding [BATCH, MAX_ACTIONS]
        # print('input_dict')
        # print(input_dict["obs"]["team"][0], input_dict["obs"]["pool"][0])
        action_logits, _ = self.action_embed_model({
            "obs": input_dict["obs"]["state"],
            "team": input_dict["obs"]["team"],
            "pool": input_dict["obs"]["pool"]
        })

        # Mask out invalid actions (use -inf to tag invalid).
        # These are then recognized by the EpsilonGreedy exploration component
        # as invalid actions that are not to be chosen.
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()