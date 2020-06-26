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
from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.trajectories import time_step as ts


class DeclinedTFUniformReplayBuffer(TFUniformReplayBuffer):

    def __init__(self, data_spec, batch_size, max_length=1000, gamma=0.5, window_size='auto'):
        """
        带收益率衰减的 replay buffer
        :param data_spec: A TensorSpec or a list/tuple/nest of TensorSpecs describing a
            single item that can be stored in this buffer.
        :param batch_size: Batch dimension of tensors when adding to buffer.
        :param max_length: The maximum number of items that can be stored in a single
            batch segment of the buffer.
        :param gamma: 衰减率 γ
        :param window_size: 计算衰减的窗口尺寸.数字.
            如果是'auto' 则等于 log(0.01) / log(reward_decline_ratio) 向上取整
        :return:
        """
        super().__init__(data_spec, batch_size, max_length=max_length)
        self.gamma = gamma
        self._trajectory_list = []
        if window_size == 'auto':
            self.window_size = int(np.ceil(np.log(0.01) / np.log(gamma)))
        else:
            self.window_size = window_size

        self._clear_before_add = True

    def add_batch(self, trajectory):
        reward = trajectory.reward
        # 更新权重
        new_trajectory_list = []
        if self.window_size is None:
            loop_count = len(self._trajectory_list) + 1
        else:
            loop_count = min(len(self._trajectory_list), self.window_size)  + 1

        # 此过程需要不断的重新创建 Trajectory 对象,效率极低但目前没有找到更有效的方法
        for num in range(1, loop_count):
            _trajectory = self._trajectory_list.pop()
            new_reward = _trajectory.reward + reward * (self.gamma ** num)
            _trajectory_new = Trajectory(
                step_type=_trajectory.step_type,
                observation=_trajectory.observation,
                action=_trajectory.action,
                policy_info=_trajectory.policy_info,
                next_step_type=_trajectory.next_step_type,
                reward=new_reward,
                discount=_trajectory.discount)
            new_trajectory_list.append(_trajectory_new)

        new_trajectory_list.reverse()
        # 加入列表
        self._trajectory_list.extend(new_trajectory_list)
        self._trajectory_list.append(trajectory)
        is_last = trajectory.is_last()
        ret = None
        if isinstance(is_last, bool):
            if trajectory.is_last():
                # 如果是最后一项, 则将列表中全部内容加入道 buffer 中
                for _trajectory in self._trajectory_list:
                    if self._clear_before_add:
                        self.clear()
                        self._clear_before_add = False

                    ret = super().add_batch(_trajectory)
            elif self.window_size is not None:
                for n in range(0, len(self._trajectory_list) - self.window_size):
                    if self._clear_before_add:
                        self.clear()
                        self._clear_before_add = False

                    ret = super().add_batch(self._trajectory_list.pop(0))

        if ret is None:
            ret = super().add_batch(trajectory)

        return ret

    def clear(self):
        super().clear()
        self._trajectory_list = []


def _test_declined_replay_buffer():
    from unittest.mock import patch, Mock
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

    def mock_step(action):
        from tf_agents.trajectories import time_step as ts
        _env = env._env.envs[0]
        if _env.last_done_state:
            return _env._reset()
        observation, rewards, _env.last_done_state = _env.market.step(action)
        # rewards = np.float64(_env.market.step_counter)
        if _env.last_done_state:
            return ts.termination(observation, rewards)
        else:
            return ts.transition(observation, rewards)

    # 运行收集数据
    # with patch.object(env, '_step', Mock(side_effect=mock_step)):
    eval_driver.run()

    database = iter(collect_replay_buffer.as_dataset(
        sample_batch_size=10,
        num_steps=20
    ).prefetch(5))
    for fetch_num, (experience, unused_info) in enumerate(database, start=1):
        print(fetch_num)
        print(experience)
        break


if __name__ == "__main__":
    _test_declined_replay_buffer()
