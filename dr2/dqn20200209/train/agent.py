#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/2/12 上午8:20
@File    : agent.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import logging
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment


logger = logging.getLogger()


def _get_df():
    import pandas as pd
    n_step = 60
    ohlcav_col_name_list = ["open", "high", "low", "close", "amount", "volume"]
    from ibats_common.example.data import load_data
    md_df = load_data('RB.csv', folder_path='/home/mg/github/IBATS_Common/ibats_common/example/data'
                      ).set_index('trade_date')[ohlcav_col_name_list]
    md_df.index = pd.DatetimeIndex(md_df.index)
    from ibats_common.backend.factor import get_factor, transfer_2_batch
    factors_df = get_factor(md_df, dropna=True)
    df_index, df_columns, data_arr_batch = transfer_2_batch(factors_df, n_step=n_step)
    md_df = md_df.loc[df_index, :]
    return md_df[['close', 'open']], data_arr_batch


def _get_net_4_test(train_env):
    from tf_agents.networks.q_network import QNetwork
    fc_layer_params = (100,)
    q_net = QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)
    return q_net


#@test {"skip": true}
def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]


def test_agent_run():
    import tensorflow as tf
    from tf_agents.utils import common
    from tf_agents.metrics.py_metrics import AverageReturnMetric
    from tf_agents.drivers.py_driver import PyDriver
    from dr2.dqn20200209.train.env import AccountEnv
    from tf_agents.policies.random_tf_policy import RandomTFPolicy
    from tf_agents.trajectories import trajectory

    num_iterations = 20000  # @param {type:"integer"}

    initial_collect_steps = 1000  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_max_length = 100000  # @param {type:"integer"}

    batch_size = 64  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 200  # @param {type:"integer"}

    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 1000  # @param {type:"integer"}

    md_df, data_factors = _get_df()
    train_env = tf_py_environment.TFPyEnvironment(AccountEnv(
        md_df=md_df, data_factors=data_factors, state_with_flag=False))
    eval_env = tf_py_environment.TFPyEnvironment(AccountEnv(
        md_df=md_df, data_factors=data_factors, state_with_flag=False))
    q_net = _get_net_4_test(train_env)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()

    eval_policy = agent.policy
    collect_policy = agent.collect_policy

    time_step = train_env.reset()
    random_policy = RandomTFPolicy(train_env.time_step_spec(),
                                   train_env.action_spec())
    random_policy.action(time_step)

    avg_return = compute_avg_return(eval_env, random_policy, num_eval_episodes)
    logger.debug('avg_return=%.4f', avg_return)

    replay_buffer = []

    # @test {"skip": true}
    def collect_step(environment, policy, buffer):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        buffer.append(traj)

    def collect_data(env, policy, buffer, steps):
        for _ in range(steps):
            collect_step(env, policy, buffer)

    collect_data(train_env, random_policy, replay_buffer, steps=100)

    logger.debug('replay_buffer length=%d', len(replay_buffer))
    for _ in replay_buffer:
        logger.debug(_)
        break


if __name__ == "__main__":
    test_agent_run()
