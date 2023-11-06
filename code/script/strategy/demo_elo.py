import os
import argparse

import ray
from ray import tune
from script.strategy.elo_env import MMEnv
from enmatch.nets.glomatch import GloModelTest, TorchMMModel
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--torch", action="store_true")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=1000)
parser.add_argument("--stop-reward", type=float, default=100.0)
parser.add_argument("--stop-timesteps", type=int, default=3000000)


# os.environ['RAY_USE_MULTIPROCESSING_CPU_COUNT'] = '1'

if __name__ == "__main__":

    args = parser.parse_args()
    args.torch = True
    args.stop_iters = 2  # 4k per iter
    args.stop_timesteps = int(1e7)
    args.stop_reward = 999
    print(vars(args))

    ray.init(num_cpus=3)

    env_config = {
        'num_players': 18,
        'team_size': 3,
        'num_features': 1,
    }
    print(env_config)
    obs_size = env_config['num_players'] + 2*env_config['team_size']
    register_env("elo_env", lambda _: MMEnv(env_config))

    ModelCatalog.register_custom_model("elo_model", GloModelTest)

    if args.run == "DQN":
        cfg = {
            # TODO(ekl) we need to set these to prevent the masked values
            # from being further processed in DistributionalQModel, which
            # would mess up the masking. It is possible to support these if we
            # defined a custom DistributionalQModel that is aware of masking.
            "hiddens": [],
            "dueling": False,
            "exploration_config": {
                # The Exploration class to use.
                "type": "EpsilonGreedy",
                # Config for the Exploration class' constructor:
                "initial_epsilon": 1.0,
                "final_epsilon": 0.02,
                "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.

                # For soft_q, use:
                # "exploration_config" = {
                #   "type": "SoftQ"
                #   "temperature": [float, e.g. 1.0]
                # }
            },
            # Size of the replay buffer. Note that if async_updates is set, then
            # each worker will have a replay buffer of this size.
            "buffer_size": int(1e6)
        }
    else:
        cfg = {}

    config = dict(
        {
            "env": "elo_env",
            "model": {
                "custom_model": GloModelTest,
                "custom_model_config": {
                    "true_obs_shape": (obs_size,),
                    "config": {
                        'num_features': env_config['num_features'],
                        'hidden_dim': 128,
                     }
                },
                'fcnet_hiddens': [1024, 512, 256],
                'fcnet_activation': 'tanh',
            },
            "num_gpus": 0,
            "num_workers": 1,
            "framework": "torch" if args.torch else "tf",
        },
        **cfg)

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    results = tune.run(args.run, stop=stop, config=config, verbose=2)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
