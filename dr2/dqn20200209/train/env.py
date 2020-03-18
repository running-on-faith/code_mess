#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/2/9 上午9:19
@File    : env.py
@contact : mmmaaaggg@163.com
@desc    :
环境信息，此模块稍后将被移植入 ibats_commons
"""
import functools
import pandas as pd
import numpy as np
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from dr2.dqn20200209.train.market import QuotesMarket

ACTION_LONG, ACTION_SHORT, ACTION_CLOSE, ACTION_KEEP = 0, 1, 2, 3
ACTIONS = [ACTION_LONG, ACTION_SHORT, ACTION_CLOSE, ACTION_KEEP]
ACTION_OP_LONG, ACTION_OP_SHORT, ACTION_OP_CLOSE, ACTION_OP_KEEP = 'long', 'short', 'close', 'keep'
ACTION_OPS = [ACTION_OP_LONG, ACTION_OP_SHORT, ACTION_OP_CLOSE, ACTION_OP_KEEP]
ACTION_OP_DIC = {_: ACTION_OPS[_] for _ in ACTIONS}
OP_ACTION_DIC = {ACTION_OPS[_]: _ for _ in ACTIONS}
# 内部访问的flag
_FLAG_LONG, _FLAG_SHORT, _FLAG_EMPTY = 1, -1, 0
# one-hot 模式的flag
FLAG_LONG, FLAG_SHORT, FLAG_EMPTY = _FLAG_LONG + 1, _FLAG_SHORT + 1, _FLAG_EMPTY + 1
FLAGS = [FLAG_LONG, FLAG_SHORT, FLAG_EMPTY]


class AccountEnv(PyEnvironment):

    def __init__(self, md_df: pd.DataFrame, data_factors: np.ndarray,
                 state_with_flag=True, action_kind_count=2, batch_size=None, **kwargs):
        super(AccountEnv, self).__init__()
        kwargs['state_with_flag'] = state_with_flag
        self._batch_size = batch_size
        self.market = QuotesMarket(md_df, data_factors, **kwargs)
        self._state_spec = array_spec.ArraySpec(
            shape=data_factors.shape[1:], dtype=data_factors.dtype, name='state')
        self._flag_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.float32, name='flag', minimum=0.0, maximum=1.0)
        self._rr_spec = array_spec.ArraySpec(
            shape=(1,), dtype=np.float32, name='rr')
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, name='action', minimum=0.0,
            maximum=max(ACTIONS[:action_kind_count]))
        self.last_done_state = False
        if state_with_flag:
            # self._observation_spec = {'state': self._state_spec, 'flag': self._flag_spec, 'rr': self._rr_spec}
            # _observation_spec 只能是数组形式,字典形式将会导致 encoding_network.EncodingNetwork 报错:
            # encoder(observation, step_type=step_type, network_state=network_state)
            # Key Error: 0
            # File "/home/mg/github/code_mess/venv/lib/python3.6/site-packages/
            #   tensorflow_core/python/util/nest.py", line 676,
            # in _yield_flat_up_to
            # at input_subtree = input_tree[shallow_key]
            self._observation_spec = (self._state_spec, self._flag_spec, self._rr_spec)
        else:
            self._observation_spec = self._state_spec

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def get_info(self):
        observation, rewards, done = self.market.step_ret_latest
        if done:
            return ts.termination(observation, rewards)
        else:
            return ts.transition(observation, rewards)

    def _step(self, action):
        if self.last_done_state:
            return self._reset()
        observation, rewards, self.last_done_state = self.market.step(action)
        if self.last_done_state:
            return ts.termination(observation, rewards)
        else:
            return ts.transition(observation, rewards)

    def _reset(self):
        self.last_done_state = False
        return ts.restart(self.market.reset())

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def batched(self):
        return self.batch_size is not None


@functools.lru_cache()
def _get_df():
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


def get_env(state_with_flag=True):
    from tf_agents.environments.tf_py_environment import TFPyEnvironment
    md_df, data_factors = _get_df()
    env = TFPyEnvironment(AccountEnv(
        md_df=md_df, data_factors=data_factors, state_with_flag=state_with_flag))
    return env


def account_env_test():
    from tf_agents.metrics.py_metrics import AverageReturnMetric
    from tf_agents.drivers.py_driver import PyDriver

    md_df, data_factors = _get_df()
    env = AccountEnv(md_df=md_df, data_factors=data_factors, state_with_flag=False)

    use_list = True
    if use_list:
        from tf_agents.policies.random_py_policy import RandomPyPolicy
        policy = RandomPyPolicy(time_step_spec=env.time_step_spec(), action_spec=env.action_spec())
        replay_buffer = []
        metric = AverageReturnMetric()
        observers = [replay_buffer.append, metric]
        driver = PyDriver(
            env, policy, observers, max_steps=1000, max_episodes=10)
        initial_time_step = env.reset()
        final_time_step, _ = driver.run(initial_time_step)
        print(f'Replay Buffer length = {len(replay_buffer)}:')
        for traj in replay_buffer:
            print(traj)
            break
    else:
        from tf_agents.policies.random_tf_policy import RandomTFPolicy
        from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
        from tf_agents.utils.nest_utils import batch_nested_array
        from tf_agents.environments.tf_py_environment import TFPyEnvironment
        from tf_agents.trajectories import trajectory
        env = TFPyEnvironment(env)  # TFPyEnvironment 会自动将 env 置为 batched=True batch_size = 1
        policy = RandomTFPolicy(time_step_spec=env.time_step_spec(), action_spec=env.action_spec())
        replay_buffer = TFUniformReplayBuffer(policy.trajectory_spec, env.batch_size)
        metric = AverageReturnMetric()
        # observers = [lambda x: replay_buffer.add_batch(batch_nested_array(x)), metric]
        observers = [replay_buffer.add_batch, metric]
        driver = PyDriver(
            env, policy, observers, max_steps=1000, max_episodes=10)
        initial_time_step = env.reset()
        final_time_step, _ = driver.run(initial_time_step)  # 这里会出错，××ReplayBuffer无法与当前 env 连用
        print(f'Replay Buffer length = {replay_buffer.size}:')
        for traj in replay_buffer.as_dataset(3, num_steps=2):
            print(traj)
            break

    # Replay Buffer length = 1000:
    # Trajectory(
    #  step_type=array(1, dtype=int32),
    #  observation={'state': array([[ 4.48054187e+03,  4.52411822e+03,  4.46154706e+03, ...,
    #          6.62600325e+01, -6.99948793e+00,  9.86842105e-01],
    #        [ 4.50847543e+03,  4.51741416e+03,  4.46825111e+03, ...,
    #          6.33523308e+01, -1.95985662e+01,  9.74025974e-01],
    #        [ 4.48724593e+03,  4.50400606e+03,  4.47607250e+03, ...,
    #          6.41403572e+01, -9.46577617e+00,  9.74358974e-01],
    #        ...,
    #        [ 4.08121714e+03,  4.10001950e+03,  4.04471845e+03, ...,
    #          3.75567764e+01, -8.93312399e+01,  2.33082707e-01],
    #        [ 4.04250641e+03,  4.08453521e+03,  4.03918835e+03, ...,
    #          3.78037117e+01, -8.68095329e+01,  2.91044776e-01],
    #        [ 4.06905091e+03,  4.15642657e+03,  4.06794489e+03, ...,
    #          5.07109403e+01, -6.80287485e+01,  3.92592593e-01]]), 'flag': array([1])},
    #  action=array([0], dtype=int32),
    #  policy_info=(),
    #  next_step_type=array(1, dtype=int32),
    #  reward=array(0.17562035, dtype=float32),
    #  discount=array(1., dtype=float32))

    print('Average Return: ', metric.result())
    #     Average Return:  0.0


if __name__ == "__main__":
    account_env_test()
