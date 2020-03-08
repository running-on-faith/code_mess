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


def train_drl(num_iterations=20, num_eval_episodes=2, num_collect_episodes=4,
              log_interval=2, state_with_flag=True,
              eval_interval=5):
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
        env, agent.collect_policy, collect_observers, num_episodes=num_collect_episodes)
    # eval
    final_trajectory_rr = FinalTrajectoryMetric()
    eval_observers = [final_trajectory_rr]
    eval_driver = DynamicEpisodeDriver(
        env, agent.policy, eval_observers, num_episodes=num_eval_episodes)

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    # agent.train = common.function(agent.train)
    # collect_driver.run = common.function(collect_driver.run)

    # Evaluate the agent's policy once before training.
    rr = compute_rr(eval_driver)
    rr_list = [rr]
    step_last = None
    for _ in range(num_iterations):

        # Collect a few steps using collect_policy and save to the replay buffer.
        collect_driver.run()

        # Sample a batch of data from the buffer and update the agent's network.
        batch_size = 5
        database = iter(collect_replay_buffer.as_dataset(
            sample_batch_size=batch_size, num_steps=agent.train_sequence_length).prefetch(3))
        for num, data in enumerate(database):
            try:
                experience, unused_info = data
                try:
                    train_loss = agent.train(experience)
                except:
                    logger.exception("%d loops train error", num)
                    break
            except ValueError:
                pass

        step = agent.train_step_counter.numpy()
        if step_last is not None and step_last == step:
            logger.warning('keep train error. stop loop.')
            break
        else:
            step_last = step

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % eval_interval == 0:
            rr = compute_rr(eval_driver)
            print('step = {0}: Return Rate = {1}'.format(step, rr))
            rr_list.append(rr)
            pass


if __name__ == "__main__":
    train_drl()
