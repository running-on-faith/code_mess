#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/6/29 上午2:22
@File    : metrics.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from tf_agents.trajectories.trajectory import Trajectory

logger = logging.getLogger()


class FinalTrajectoryMetric:

    def __init__(self, stat_action_count=True):
        self.replay_buffer = []
        self.data_dic = defaultdict(list)
        self.stat_action_count = stat_action_count

    def __call__(self, trajectory: Trajectory):
        self.replay_buffer.append(trajectory)
        if trajectory.is_last():
            pct_chgs = np.ones(len(self.replay_buffer))
            action_counts, last_flag = np.zeros(len(self.replay_buffer)), None
            for idx, _ in enumerate(self.replay_buffer):
                pct_chgs[idx] += _.reward.numpy()
                if self.stat_action_count:
                    flag = _.observation[1].numpy()[0, 0]
                    if last_flag != flag:
                        action_counts[idx] += 1
                        last_flag = flag

            rr = pct_chgs.prod() - 1
            action_count = action_counts.sum()
            self.data_dic['rr'].append(rr)
            self.data_dic['action_count'].append(action_count)
            self.data_dic['avg_action_period'].append(len(self.replay_buffer) / action_count)
            self.replay_buffer = []

    def result(self):
        stat_dic = {
            'rr': np.mean(self.data_dic['rr']),
            'action_count': np.mean(self.data_dic['action_count']),
            'avg_action_period': np.mean(self.data_dic['avg_action_period']),
        }
        self.data_dic = defaultdict(list)
        return stat_dic


class PlotTrajectoryMatrix:

    def __init__(self, stat_action_count=True):
        self.replay_buffer = []
        self.rr_dic = {}
        self.action_dic = {}
        self.stat_action_count = stat_action_count

    def __call__(self, trajectory: Trajectory):
        self.replay_buffer.append(trajectory)
        try:
            is_last = trajectory.is_last().numpy()[0]
            if is_last:
                pct_chgs = np.ones(len(self.replay_buffer))
                action_counts, last_flag = np.zeros(len(self.replay_buffer)), None
                for idx, _ in enumerate(self.replay_buffer):
                    pct_chgs[idx] += _.reward.numpy()
                    if self.stat_action_count:
                        flag = _.observation[1].numpy()[0, 0]
                        if last_flag != flag:
                            action_counts[idx] += 1
                            last_flag = flag

                self.rr_dic[f"rr_{len(self.rr_dic)}"] = pct_chgs.cumprod() - 1
                self.action_dic[f"action_count_{len(self.action_dic)}"] = action_counts.cumsum()
                self.replay_buffer = []
                # logger.info("trajectory.is_last, len(self.replay_buffer)=%d\nrr list=%s",
                #             len(self.replay_buffer), pct_chgs.cumprod() - 1)
        except ValueError:
            pass

    def result(self):
        import matplotlib.pyplot as plt
        from datetime import datetime
        from ibats_utils.mess import datetime_2_str

        # logger.info("len(self.rr_dic)=%d\n%s", len(self.rr_dic), self.rr_dic)
        if len(self.rr_dic) > 0:
            dic = self.rr_dic.copy()
            dic.update(self.action_dic)
            rr_df = pd.DataFrame(dic)
            rr_df.plot(secondary_y=[_ for _ in rr_df.columns if _.startswith('action')])
            file_name = f"{datetime_2_str(datetime.now())}_episode_final_plot.png"
            plt.savefig(file_name)
            plt.close()
            logger.debug("file_name: %s saved", file_name)
            self.replay_buffer = []
            self.rr_dic = {}
            self.action_dic = {}
        else:
            rr_df = None
        return rr_df


def run_and_get_result(driver):
    """模拟运算，进行收益统计"""
    driver.run()
    results_dic = {}
    for matrix in driver.observers:
        result = matrix.result()
        results_dic[type(matrix)] = result

    return results_dic


def _test_matrix(matrix_class=PlotTrajectoryMatrix):
    num_episodes = 1
    state_with_flag = True
    from dr2.common.env import get_env
    from dr2.dqn20200209.train.agent import get_agent
    from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
    env = get_env(state_with_flag=state_with_flag)
    agent = get_agent(env)
    collect_policy = agent.collect_policy
    matrix = matrix_class()
    observers = [matrix]
    driver = DynamicEpisodeDriver(
        env, collect_policy, observers, num_episodes=num_episodes)

    driver.run()
    result = matrix.result()
    print(result)


if __name__ == "__main__":
    _test_matrix()
