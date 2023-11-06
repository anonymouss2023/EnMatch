from typing import Dict
import argparse
import numpy as np
import os
from typing import Dict, Optional, TYPE_CHECKING
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

def get_callback(batch_size):
    class InfoObjCallbacks(DefaultCallbacks):

        def on_episode_start(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                             policies: Dict[str, Policy], episode: Episode,
                             **kwargs) -> None:
            episode.user_data["objs"] = []

        def on_episode_step(self,
                            *,
                            worker: "RolloutWorker",
                            base_env: BaseEnv,
                            policies: Optional[Dict[str, Policy]] = None,
                            episode: Episode,
                            **kwargs) -> None:
            # Make sure this episode is ongoing.
            assert episode.length > 0, \
                "ERROR: `on_episode_step()` callback should not be called right " \
                "after env reset!"
            # for i in range(batch_size):
            obj = episode.last_info_for().get('obj', 0)
            episode.user_data["objs"].append(obj)

        def on_episode_end(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                           policies: Dict[str, Policy], episode: Episode,
                           **kwargs) -> None:
            episode.custom_metrics["obj"] = np.sum(episode.user_data["objs"])
            # episode.hist_data["objs"] = episode.user_data["objs"]
    return InfoObjCallbacks
