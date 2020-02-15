#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/2/15 上午10:07
@File    : policy.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
from tf_agents.policies.random_tf_policy import RandomTFPolicy


def get_policy(env):
    tf_policy = RandomTFPolicy(action_spec=env.action_spec(),
                               time_step_spec=env.time_step_spec())
    return tf_policy


if __name__ == "__main__":
    pass
