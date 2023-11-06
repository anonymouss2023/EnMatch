import os
import numpy as np
import gym
import torch
import ray
from copy import deepcopy
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from enmatch.utils.rllib_print import pretty_print
from enmatch.utils.rllib_vector_env import MyVectorEnvWrapper
from enmatch.utils.rllib_metrics import get_callback
from enmatch.utils.fileutil import find_newest_files
import http.client
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.pg as pg
import ray.rllib.agents.ddpg.td3 as td3
import ray.rllib.agents.impala as impala
import ray.rllib.agents.ddpg as ddpg
import ray.rllib.agents.slateq as slateq
import sys
from script.strategy.utils import default_config
from enmatch.nets.glomatch import GloModel

http.client.HTTPConnection._http_vsn = 10
http.client.HTTPConnection._http_vsn_str = 'HTTP/1.0'

def get_rl_model(algo, rllib_config):
    trainer = None
    if algo == "PPO":
        trainer = ppo.PPOTrainer(config=rllib_config, env="rllibEnv-v0")
    elif algo == "DQN":
        trainer = dqn.DQNTrainer(config=rllib_config, env="rllibEnv-v0")
    elif algo == "RAINBOW":
        trainer = dqn.DQNTrainer(config=rllib_config, env="rllibEnv-v0")
    elif algo == "A2C":
        trainer = a3c.A2CTrainer(config=rllib_config, env="rllibEnv-v0")
    elif algo == "A3C":
        trainer = a3c.A3CTrainer(config=rllib_config, env="rllibEnv-v0")
    elif algo == "PG":
        trainer = pg.PGTrainer(config=rllib_config, env="rllibEnv-v0")
    elif algo == "DDPG":
        trainer = ddpg.DDPGTrainer(config=rllib_config, env="rllibEnv-v0")
    elif algo == "TD3":
        trainer = td3.TD3Trainer(config=rllib_config, env="rllibEnv-v0")
    elif algo == "IMPALA":
        trainer = impala.ImpalaTrainer(config=rllib_config, env="rllibEnv-v0")
    elif algo == "SLATEQ":
        trainer = slateq.SlateQTrainer(config=rllib_config, env="rllibEnv-v0")
    else:
        assert algo in ("PPO", "DQN", "A2C", "A3C", "PG", "IMPALA", "TD3", "RAINBOW", "SLATEQ")
    print('trainer_default_config', trainer._default_config)
    return trainer

algo = sys.argv[1]
stage = sys.argv[2]
extra_config = eval(sys.argv[3]) if len(sys.argv) >= 4 else {}

ray.init(num_cpus=3)
print('torch.cuda.is_available()', torch.cuda.is_available())
print('get_gpu_ids', ray.get_gpu_ids())

num_per_team = default_config['num_per_team']
num_matches = default_config['num_matches']

config = {
    **default_config,
    'epoch':100000,
    'batch_size':32,
    'num_players': 2*num_per_team*num_matches,
    'num_features': 1,
    "remote_base": 'http://127.0.0.1:5000',
    'gpu':0,
    **extra_config
}
config['action_size'] = config['max_steps'] = config['num_players'] = 2 * config['num_per_team'] * config['num_matches']
print(config)
obs_size = config['num_players'] + 2*config['num_per_team']

eval_config = deepcopy(config)

ModelCatalog.register_custom_model("glomatch", GloModel)
register_env('rllibEnv-v0', lambda _: MyVectorEnvWrapper(gym.make('HttpEnv-v0', env_id=config['env'], config=config), config['batch_size']))

modelfile = algo + '_' + config['env'] + '_' + config['trial_name']
output_dir = os.environ['enmatch_output_dir']
checkpoint_dir = '%s/ray_results/%s/' % (output_dir, modelfile)
restore_dir = find_newest_files('checkpoint*', checkpoint_dir)
restore_file = find_newest_files('checkpoint*', restore_dir)
restore_file = restore_file[:restore_file.rfind('.')] \
    if '.' in restore_file.split('/')[-1] \
    else restore_file

assert "PPO" == algo

cfg = {
    "num_workers": 1,
    "use_critic": True,
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": True,
    # The GAE (lambda) parameter.
    "lambda": 1.0,
    # Initial coefficient for KL divergence.
    "kl_coeff": 0.2,
    # # Size of batches collected from each worker.
    # "rollout_fragment_length": 256,
    # # Number of timesteps collected for each SGD round. This defines the size
    # # of each SGD epoch.
    # "train_batch_size": 2048,
    # Total SGD batch size across all devices for SGD. This defines the
    # minibatch size within each epoch.
    "sgd_minibatch_size": 128,
    # Whether to shuffle sequences in the batch when training (recommended).
    "shuffle_sequences": True,
    # Number of SGD iterations in each outer loop (i.e., number of epochs to
    # execute per train batch).
    "num_sgd_iter": 1,
    # Stepsize of SGD.
    "lr": 0.0001,
    # Coefficient of the value function loss. IMPORTANT: you must tune this if
    # you set vf_share_layers=True inside your model's config.
    "vf_loss_coeff": 0.5,
    # PPO clip parameter.
    "clip_param": 0.3,
    # Clip param for the value function. Note that this is sensitive to the
    # scale of the rewards. If your expected V is large, increase this.
    "vf_clip_param": 500.0,
    # If specified, clip the global norm of gradients by this amount.
    # "grad_clip": 10.0,
    # Target value for KL divergence.
    "kl_target": 0.01,
}

rllib_config = dict(
    {
        "env": "rllibEnv-v0",
        "gamma": 1,
        "explore": True,
        "model": {
            "custom_model": "glomatch",
            "custom_model_config": {
                "true_obs_shape": (obs_size,),
                "config": {
                    'num_features': config['num_features'],
                    'hidden_dim': 128,
                    'num_per_team': config['num_per_team'],
                    'num_matches': config['num_matches'],
                    'obs_norm': 10000.0
                }
            },
            'fcnet_hiddens': [1024, 512, 256],
            'fcnet_activation': 'tanh',
        },
        "exploration_config": {
            "type": "SoftQ",
            # "temperature": 1.0,
        },
        "num_gpus": 1 if config.get('gpu', True) else 0,
        # "num_workers": 0,
        "framework": 'torch',
        # "framework": 'tfe',
        "rollout_fragment_length": config['max_steps'],
        "batch_mode": "complete_episodes",
        "train_batch_size": min(config["batch_size"] * config['max_steps'], 1024),
        "evaluation_interval": 500,
        "evaluation_num_episodes": 2048 * 4,
        "evaluation_config": {
            "explore": False
        },
        "log_level": "INFO",
        "callbacks":get_callback(batch_size=config['batch_size'])
    },
    **cfg)
print('rllib_config', rllib_config)
trainer = get_rl_model(algo.split('_')[0], rllib_config)
# trainer.register_train_result_callback(on_train_result)

if stage == 'train':
    try:
        trainer.restore(restore_file)
        print('model restore from %s' % (restore_file))
    except Exception:
        trainer = get_rl_model(algo.split('_')[0], rllib_config)
    for i in range(config["epoch"]):
        result = trainer.train()
        if (i + 1) % 50 == 0:
            print('epoch ',i)
        if (i + 1) % 100 == 0 or i == 0:
            print(pretty_print(result))
        if (i + 1) % 500 == 0:
            checkpoint = trainer.save(checkpoint_dir=checkpoint_dir)
            print("checkpoint saved at", checkpoint)

if stage == 'eval':
    eval_config = config.copy()
    eval_config['is_eval'] = True
    eval_config['batch_size'] = 2048
    eval_env = gym.make('HttpEnv-v0', env_id=eval_config['env'], config=eval_config)
    # trainer.restore(checkpoint_dir + '/checkpoint_010000/checkpoint-10000')
    trainer.restore(restore_file)
    print('model restore from %s' % (restore_file))
    episode_reward = 0
    done = False
    epoch = 4
    actions = []
    for i in range(epoch):
        obs = eval_env.reset()
        print('test batch at ', i, 'avg reward', episode_reward / eval_config['batch_size'] / (i + 0.0001))
        for _ in range(config["max_steps"]):
            obs = dict(enumerate(obs))
            action = trainer.compute_actions(obs, explore=False)
            action = np.array(list(action.values()))
            obs, reward, done, info = eval_env.step(action)
            episode_reward += sum(reward)
            actions.append(action)
    print('avg reward', episode_reward / eval_config['batch_size'] / epoch)
    eval_env.close()


ray.shutdown()
