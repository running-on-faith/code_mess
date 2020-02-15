#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/2/15 上午9:42
@File    : train_drl.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import logging
from dr2.dqn20200209.train.agent import get_agent
from dr2.dqn20200209.train.env import get_env
from dr2.dqn20200209.train.policy import get_policy


logger = logging.getLogger()


def train_drl(num_episodes=2):
    env = get_env(state_with_flag=False)
    agent = get_agent(env)
    agent.initialize()
    policy = get_policy(env)

    from tf_agents.metrics import tf_metrics
    from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
    from tf_agents.drivers.py_driver import PyDriver
    avg_ret = tf_metrics.AverageReturnMetric()
    replay_buffer = []
    observers = [replay_buffer.append, avg_ret]
    driver = DynamicEpisodeDriver(
        env, policy, observers, num_episodes=num_episodes)
    # Initial driver.run will reset the environment and initialize the policy.
    final_time_step, policy_state = driver.run()
    logger.debug('final_time_step %s', final_time_step)
    logger.debug('avg_ret %f', avg_ret.result().numpy())
    logger.debug('replay_buffer length=%d', len(replay_buffer))


if __name__ == "__main__":
    train_drl()
