#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/6/23 下午10:01
@File    : replay_buffer.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import numpy as np
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer


class DeclinedTFUniformReplayBuffer(TFUniformReplayBuffer):

    def __init__(self, data_spec, batch_size, max_length=1000, reward_decline_ratio=0.5, window_size='auto'):
        """
        带收益率衰减的 replay buffer
        :param data_spec: A TensorSpec or a list/tuple/nest of TensorSpecs describing a
            single item that can be stored in this buffer.
        :param batch_size: Batch dimension of tensors when adding to buffer.
        :param max_length: The maximum number of items that can be stored in a single
            batch segment of the buffer.
        :param reward_decline_ratio: 衰减率 λ
        :param window_size: 计算衰减的窗口尺寸.数字.
            如果是'auto' 则等于 log(0.01) / log(reward_decline_ratio) 向上取整
        :return:
        """
        super().__init__(data_spec, batch_size, max_length=max_length)
        self.reward_decline_ratio = reward_decline_ratio
        if window_size == 'auto':
            self.window_size = np.ceil(np.log(0.01) / np.log(reward_decline_ratio))
        else:
            self.window_size = window_size

    def add_batch(self, items):
        super().add_batch(items)

    def clear(self):
        super().clear()


def _test_declined_replay_buffer():
    from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
    from dr2.dqn20200209.train.env import get_env
    from dr2.dqn20200209.train.agent import get_agent
    # 建立环境
    state_with_flag = True
    env = get_env(state_with_flag=state_with_flag)
    agent = get_agent(env, state_with_flag=state_with_flag)
    eval_policy = agent.policy
    collect_replay_buffer = DeclinedTFUniformReplayBuffer(agent.collect_data_spec, env.batch_size)
    eval_observers = [collect_replay_buffer.add_batch]
    eval_driver = DynamicEpisodeDriver(
        env, eval_policy, eval_observers, num_episodes=1)
    # 运行收集数据
    eval_driver.run()

    database = iter(collect_replay_buffer.as_dataset(
        sample_batch_size=10,
        num_steps=10
    ).prefetch(10))
    for fetch_num, (experience, unused_info) in enumerate(database, start=1):
        print(fetch_num)
        print(experience)


if __name__ == "__main__":
    _test_declined_replay_buffer()
