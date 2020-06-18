#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/2/15 上午9:42
@File    : train_drl.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import logging

from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.utils import common
from tf_agents.policies import greedy_policy
from dr2.dqn20200209.train.agent import get_agent
from dr2.dqn20200209.train.env import get_env
from dr2.dqn20200209.train.policy import get_policy

logger = logging.getLogger()


class FinalTrajectoryMetric:

    def __init__(self):
        self.replay_buffer = []
        self.final_rr = []

    def __call__(self, trajectory):
        import numpy as np
        self.replay_buffer.append(trajectory)
        if trajectory.is_last():
            pct_chgs = np.ones(len(self.replay_buffer))
            for idx, _ in enumerate(self.replay_buffer):
                pct_chgs[idx] += _.reward.numpy()
            rr = pct_chgs.prod() - 1
            self.final_rr.append(rr)
            self.replay_buffer = []

    def result(self):
        import numpy as np
        return np.mean(self.final_rr)


def compute_rr(driver):
    driver.run()
    final_trajectory_rr = driver.observers[0]
    return final_trajectory_rr.result()


def train_drl(train_loop_count=20, num_eval_episodes=1, num_collect_episodes=4,
              log_interval=2, state_with_flag=True,
              eval_interval=5):
    """
    :param train_loop_count: 总体训练轮数
    :param num_eval_episodes: 评估测试次数
    :param num_collect_episodes: 数据采集次数
    :param log_interval:日志间隔
    :param state_with_flag:带 flag 标识
    :param eval_interval:评估间隔
    :return:
    """
    logger.info("Train started")
    env = get_env(state_with_flag=state_with_flag)
    agent = get_agent(env, state_with_flag=state_with_flag)
    eval_policy = agent.policy
    collect_policy = agent.collect_policy

    from tf_agents.metrics import tf_metrics
    from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
    # collect
    collect_replay_buffer = TFUniformReplayBuffer(agent.collect_data_spec, env.batch_size)
    collect_observers = [collect_replay_buffer.add_batch]
    collect_driver = DynamicEpisodeDriver(
        env, collect_policy, collect_observers, num_episodes=num_collect_episodes)
    # eval 由于历史行情相对确定,因此,获取最终汇报只需要跑一次即可
    final_trajectory_rr = FinalTrajectoryMetric()
    eval_observers = [final_trajectory_rr]
    eval_driver = DynamicEpisodeDriver(
        env, eval_policy, eval_observers, num_episodes=num_eval_episodes)

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    # agent.train = common.function(agent.train)
    # collect_driver.run = common.function(collect_driver.run)

    # Evaluate the agent's policy once before training.
    rr = compute_rr(eval_driver)
    rr_list = [rr]
    step, step_last = 0, None
    for loop_n in range(train_loop_count):

        # Collect a few steps using collect_policy and save to the replay buffer.
        logger.debug("%d/%d) collecting %d episodes", loop_n, train_loop_count, num_collect_episodes)
        collect_driver.run()

        # Sample a batch of data from the buffer and update the agent's network.
        batch_size, prefetch_count = 1024, 10
        database = iter(collect_replay_buffer.as_dataset(
            sample_batch_size=batch_size, num_steps=agent.train_sequence_length).prefetch(prefetch_count))
        train_loss = None
        for fetch_num, (experience, unused_info) in enumerate(database, start=1):
            try:
                try:
                    logger.debug(
                        "%d/%d) step=%d training %d -> %d of batch_size=%d data",
                        loop_n, train_loop_count, step, fetch_num,
                        len(experience.observation), experience.observation[0].shape[0])
                    train_loss = agent.train(experience)
                    step = agent.train_step_counter.numpy()
                except Exception as exp:
                    if isinstance(exp, KeyboardInterrupt):
                        raise exp from exp
                    logger.exception("%d/%d) train error", loop_n, train_loop_count)
                    break
            except ValueError:
                logger.exception('train error')

            if fetch_num >= prefetch_count:
                break

        logger.debug("clear buffer")
        collect_replay_buffer.clear()

        logger.info("%d/%d) step=%d", loop_n, train_loop_count, step)
        if step_last is not None and step_last == step:
            logger.warning('keep train error. stop loop.')
            break
        else:
            step_last = step

        if step % log_interval == 0:
            logger.info('%d/%d) step=%d loss=%.8f', loop_n, train_loop_count, step,
                        train_loss.loss if train_loss else None)

        if step % eval_interval == 0:
            rr = compute_rr(eval_driver)
            logger.info('%d/%d) step=%d rr = %.2f%%', loop_n, train_loop_count, step, rr * 100)
            rr_list.append(rr)

    logger.info("rr_list=%s", rr_list)
    logger.info("Train finished")


if __name__ == "__main__":
    train_drl(train_loop_count=50)