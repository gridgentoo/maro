# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
import time

from maro.rl import Learner

sync_mode_path = os.path.dirname(os.path.realpath(__file__))  # DQN sync mode directory
dqn_path = os.path.dirname(sync_mode_path)  # DQN directory
sys.path.insert(0, dqn_path)
sys.path.insert(0, sync_mode_path)
from general import config, log_dir
from policy_manager.policy_manager import get_policy_manager
from rollout_manager import get_rollout_manager


if __name__ == "__main__":
    learner = Learner(
        policy_manager=get_policy_manager(),
        rollout_manager=get_rollout_manager(),
        num_episodes=config["num_episodes"],
        eval_schedule=config["eval_schedule"],
        log_dir=log_dir
    )
    time.sleep(10)
    learner.run()
