#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/2/15 上午10:07
@File    : policy.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import os
from tf_agents.policies.random_tf_policy import RandomTFPolicy


def get_policy(env):
    tf_policy = RandomTFPolicy(action_spec=env.action_spec(),
                               time_step_spec=env.time_step_spec())
    return tf_policy


def save_policy(saver, key, base_path=None):
    """
    将 policy 参数进行保存
    可通过一下方式进行加载并使用
    saved_policy = tf.compat.v2.saved_model.load('policy_0')
    policy_state = saved_policy.get_initial_state(batch_size=3)
    time_step = ...
    while True:
      policy_step = saved_policy.action(time_step, policy_state)
      policy_state = policy_step.state
      time_step = f(policy_step.action)
    ...
    :param saver:
    :param key: 保存文件名
    :param base_path: 基础路径
    :return:
    """
    if base_path is None:
        save_path = os.path.join(os.curdir, 'model', f"{key}")
    else:
        save_path = os.path.join(base_path, 'model', f"{key}")

    saver.save(save_path)
    return save_path


if __name__ == "__main__":
    pass
