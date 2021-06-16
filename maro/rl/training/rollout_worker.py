# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from multiprocessing.connection import Connection
from os import getcwd
from typing import Callable

from maro.communication import Proxy
from maro.rl.env_wrapper import AbsEnvWrapper
from maro.utils import Logger, set_seeds

from .decision_generator import AbsDecisionGenerator
from .message_enums import MsgKey, MsgTag


def rollout_worker_process(
    index: int,
    conn: Connection,
    create_env_wrapper_func: Callable[[], AbsEnvWrapper],
    create_decision_generator_func: Callable[[], AbsDecisionGenerator],
    create_eval_env_wrapper_func: Callable[[], AbsEnvWrapper],
    log_dir: str
):
    set_seeds(index)
    env_wrapper = create_env_wrapper_func()
    eval_env_wrapper = env_wrapper if not create_eval_env_wrapper_func else create_eval_env_wrapper_func() 
    decision_generator = create_decision_generator_func()
    logger = Logger("ROLLOUT_WORKER", dump_folder=log_dir)

    def collect(msg):
        ep, segment = msg["episode"], msg["segment"]
        # load policies
        if hasattr(decision_generator, "update"):
            decision_generator.update(msg["policy"])

        # update exploration parameters
        decision_generator.explore()
        if msg["exploration_step"]:
            decision_generator.exploration_step()

        if env_wrapper.state is None:
            logger.info(f"Training episode {ep}")
            env_wrapper.reset()
            env_wrapper.start()  # get initial state

        starting_step_index = env_wrapper.step_index + 1
        steps_to_go = float("inf") if msg["num_steps"] == -1 else msg["num_steps"]
        while env_wrapper.state and steps_to_go > 0:
            action = decision_generator.choose_action(env_wrapper.state, ep, env_wrapper.step_index)
            env_wrapper.step(action)
            steps_to_go -= 1

        logger.info(
            f"Roll-out finished for ep {ep}, segment {segment}"
            f"(steps {starting_step_index} - {env_wrapper.step_index})"
        )

        if hasattr(decision_generator, "store_experiences"):
            policy_names = decision_generator.store_experiences(env_wrapper.get_experiences())
            ret_exp = decision_generator.get_experiences_by_policy(policy_names)

        return_info = {
            "worker_index": index,
            "episode_end": not env_wrapper.state,
            "experiences": ret_exp,
            "env_summary": env_wrapper.summary,
            "num_steps": env_wrapper.step_index - starting_step_index + 1
        }

        conn.send(return_info)

    def evaluate(msg):
        logger.info(f"Evaluating...")
        eval_env_wrapper.reset()
        eval_env_wrapper.start()  # get initial state
        decision_generator.exploit()
        if hasattr(decision_generator, "update"):
            decision_generator.update(msg["policy"])
        while eval_env_wrapper.state:
            action = decision_generator.choose_action(
                eval_env_wrapper.state, msg["episode"], eval_env_wrapper.step_index
            )
            eval_env_wrapper.step(action)

        conn.send({"worker_id": index, "env_summary": eval_env_wrapper.summary})

    while True:
        msg = conn.recv()
        if msg["type"] == "collect":
            collect(msg)
        elif msg["type"] == "evaluate":
            evaluate(msg)
        elif msg["type"] == "quit":
            break


def rollout_worker_node(
    create_env_wrapper_func: Callable[[], AbsEnvWrapper],
    create_decision_generator_func: Callable[[], AbsDecisionGenerator],
    group: str,
    create_eval_env_wrapper_func: Callable[[], AbsEnvWrapper] = None,
    log_dir: str = getcwd(),
    **proxy_kwargs
):
    """Roll-out worker process.

    Args:
        env (AbsEnvWrapper): An ``AbsEnvWrapper`` instance to interact with a set of agents and collect experiences
            for policy training / update.
        decision_generator (AbsDecisionGenerator): Source of action decisions which could be local or remote
            depending on the implementation. 
        group (str): Group name for all roll-out workers and the roll-out manager that manages them. The roll-out
            manager process must be assigned this group name in order to form a communicating cluster.
        exploration_dict (Dict[str, AbsExploration]): A set of named exploration schemes. Defaults to None.
        agent2exploration (Dict[str, str]): Mapping from agent ID's to exploration scheme ID's. This is used to direct
            an agent's query to the correct exploration scheme. Defaults to None.
        eval_env (AbsEnvWrapper): An ``AbsEnvWrapper`` instance for policy evaluation. If None, ``env`` will be used
            as the evaluation environment. Defaults to None.
        log_dir (str): Directory to store logs in. A ``Logger`` will be created at init time and this directory
            will be used to save the log files generated by it. Defaults to the current working directory.
        proxy_kwargs: Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
            for details.
    """
    env_wrapper = create_env_wrapper_func()
    eval_env_wrapper = env_wrapper if not create_eval_env_wrapper_func else create_eval_env_wrapper_func() 
    decision_generator = create_decision_generator_func()

    proxy = Proxy(group, "rollout_worker", {"rollout_manager": 1}, **proxy_kwargs)
    logger = Logger(proxy.name, dump_folder=log_dir)

    def collect(msg):
        ep, segment = msg.body[MsgKey.EPISODE], msg.body[MsgKey.SEGMENT]
        # load policies
        if hasattr(decision_generator, "update"):
            decision_generator.update(msg.body[MsgKey.POLICY])
        # set exploration parameters
        decision_generator.explore()
        if msg.body[MsgKey.EXPLORATION_STEP]:
            decision_generator.exploration_step()

        if env_wrapper.state is None:
            logger.info(f"Training episode {msg.body[MsgKey.EPISODE]}")
            env_wrapper.reset()
            env_wrapper.start()  # get initial state

        starting_step_index = env_wrapper.step_index + 1
        steps_to_go = float("inf") if msg.body[MsgKey.NUM_STEPS] == -1 else msg.body[MsgKey.NUM_STEPS]
        while env_wrapper.state and steps_to_go > 0:
            action = decision_generator.choose_action(env_wrapper.state, ep, env_wrapper.step_index)
            env_wrapper.step(action)
            steps_to_go -= 1

        logger.info(
            f"Roll-out finished for ep {ep}, segment {segment}"
            f"(steps {starting_step_index} - {env_wrapper.step_index})"
        )

        if hasattr(decision_generator, "store_experiences"):
            policy_names = decision_generator.store_experiences(env_wrapper.get_experiences())
            ret_exp = decision_generator.get_experiences_by_policy(policy_names)

        return_info = {
            MsgKey.EPISODE_END: not env_wrapper.state,
            MsgKey.EPISODE: ep,
            MsgKey.SEGMENT: segment,
            MsgKey.EXPERIENCES: ret_exp,
            MsgKey.ENV_SUMMARY: env_wrapper.summary,
            MsgKey.NUM_STEPS: env_wrapper.step_index - starting_step_index + 1
        }

        proxy.reply(msg, tag=MsgTag.COLLECT_DONE, body=return_info)

    def evaluate(msg):
        logger.info(f"Evaluating...")
        ep = msg.body[MsgKey.EPISODE]
        eval_env_wrapper.reset()
        eval_env_wrapper.start()  # get initial state
        decision_generator.exploit()
        if hasattr(decision_generator, "update"):
            decision_generator.update(msg.body[MsgKey.POLICY])
        while eval_env_wrapper.state:
            action = decision_generator.choose_action(eval_env_wrapper.state, ep, eval_env_wrapper.step_index)
            eval_env_wrapper.step(action)

        return_info = {MsgKey.ENV_SUMMARY: eval_env_wrapper.summary, MsgKey.EPISODE: msg.body[MsgKey.EPISODE]}
        proxy.reply(msg, tag=MsgTag.EVAL_DONE, body=return_info)

    """
    The event loop handles 3 types of messages from the roll-out manager:
        1)  COLLECT, upon which the agent-environment simulation will be carried out for a specified number of steps
            and the collected experiences will be sent back to the roll-out manager;
        2)  EVAL, upon which the policies contained in the message payload will be evaluated for the entire
            duration of the evaluation environment.
        3)  EXIT, upon which it will break out of the event loop and the process will terminate.

    """
    for msg in proxy.receive():
        if msg.tag == MsgTag.EXIT:
            logger.info("Exiting...")
            proxy.close()
            break

        if msg.tag == MsgTag.COLLECT:
            collect(msg)
        elif msg.tag == MsgTag.EVAL:
            evaluate(msg)