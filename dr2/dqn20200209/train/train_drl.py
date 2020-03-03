#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/2/15 上午9:42
@File    : train_drl.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import logging
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


def compute_avg_return(driver):
    driver.run()
    final_trajectorys = driver.observers[0]
    return final_trajectorys.result()


def train_drl(num_iterations=20, num_eval_episodes=2, num_collect_episodes=4,
              log_interval=2,
              eval_interval=5):
    env = get_env(state_with_flag=False)
    agent = get_agent(env)
    agent.initialize()
    eval_policy = agent.policy
    collect_policy = agent.collect_policy

    from tf_agents.metrics import tf_metrics
    from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
    # collect
    collect_replay_buffer = []
    collect_observers = [collect_replay_buffer.append]
    collect_driver = DynamicEpisodeDriver(
        env, agent.collect_policy, collect_observers, num_episodes=num_collect_episodes)
    # eval
    final_trajectory = FinalTrajectoryMetric()
    eval_observers = [final_trajectory]
    eval_driver = DynamicEpisodeDriver(
        env, agent.policy, eval_observers, num_episodes=num_eval_episodes)

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)
    # collect_driver.run = common.function(collect_driver.run)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Initial driver.run will reset the environment and initialize the policy.
    # final_time_step, policy_state = collect_driver.run()
    # logger.debug('final_time_step %s', final_time_step)
    # logger.debug('avg_ret %f', avg_ret.result().numpy())
    # logger.debug('replay_buffer length=%d', len(replay_buffer))

    # Evaluate the agent's policy once before training.
    # avg_return = compute_avg_return(eval_driver)
    # returns = [avg_return]

    for _ in range(num_iterations):

        # Collect a few steps using collect_policy and save to the replay buffer.
        collect_driver.run()

        # Sample a batch of data from the buffer and update the agent's network.
        train_loss = agent.train(collect_replay_buffer)

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % eval_interval == 0:
            # avg_return = compute_avg_return(eval_driver)
            # print('step = {0}: Average Return = {1}'.format(step, avg_return))
            # returns.append(avg_return)
            pass


if __name__ == "__main__":
    train_drl()
