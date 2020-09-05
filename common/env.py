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
from ibats_common.example import get_trade_date_series, get_delivery_date_series
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from ibats_common.backend.rl.emulator.market2 import QuotesMarket
from drl import DATA_FOLDER_PATH

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
                 state_with_flag=True, action_kind_count=2, batch_size=None,
                 is_continuous_action=False, long_holding_punish=0,
                 **kwargs):
        super(AccountEnv, self).__init__()
        kwargs['state_with_flag'] = state_with_flag
        kwargs['long_holding_punish'] = long_holding_punish
        self._batch_size = batch_size
        self.market = QuotesMarket(md_df, data_factors, **kwargs)
        self.is_continuous_action = is_continuous_action
        self.action_kind_count = action_kind_count
        self._state_spec = array_spec.ArraySpec(
            shape=data_factors.shape[1:], dtype=data_factors.dtype, name='state')
        self._flag_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.float32, name='flag', minimum=0.0, maximum=1.0)
        self._rr_spec = array_spec.ArraySpec(
            shape=(1,), dtype=np.float32, name='rr')
        self._holding_period_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.float32, name='holding_period', minimum=0.0)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32 if not is_continuous_action else np.float32,
            name='action', minimum=0.0, maximum=max(ACTIONS[:action_kind_count]))
        self.last_done_state = False
        _observation_spec = [self._state_spec]
        # 添加 flag、rr 结构
        if state_with_flag:
            _observation_spec.extend([self._flag_spec, self._rr_spec])

        # 添加 持仓周期数 结构
        if long_holding_punish>0:
            _observation_spec.append(self._holding_period_spec)

        # 如果只有一个数据，则无需数组形式
        if len(_observation_spec) > 1:
            self._observation_spec = tuple(_observation_spec)
        else:
            self._observation_spec = _observation_spec[0]

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
        if self.is_continuous_action:
            action = int(np.round(action))

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
def load_md(instrument_type):
    from ibats_common.example.data import load_data
    ohlcav_col_name_list = ["open", "high", "low", "close", "amount", "volume"]
    md_df = load_data(f'{instrument_type}.csv', folder_path=DATA_FOLDER_PATH
                      ).set_index('trade_date')[ohlcav_col_name_list]
    md_df.index = pd.DatetimeIndex(md_df.index)
    return md_df


@functools.lru_cache()
def _get_df():
    from ibats_common.backend.factor import get_factor, transfer_2_batch
    n_step = 60
    instrument_type = 'RB'
    md_df = load_md(instrument_type)
    trade_date_series = get_trade_date_series(DATA_FOLDER_PATH)
    delivery_date_series = get_delivery_date_series(instrument_type, DATA_FOLDER_PATH)
    factors_df = get_factor(md_df, trade_date_series, delivery_date_series, dropna=True)
    df_index, df_columns, data_arr_batch = transfer_2_batch(factors_df, n_step=n_step)
    md_df = md_df.loc[df_index, :]
    return md_df[['close', 'open']], data_arr_batch


@functools.lru_cache()
def _get_mock_df(period_len=40):
    n_step = 60
    ohlcav_col_name_list = ["open", "high", "low", "close", "amount", "volume"]
    instrument_type = 'RB'
    from ibats_common.example.data import load_data
    trade_date_series = get_trade_date_series(DATA_FOLDER_PATH)
    delivery_date_series = get_delivery_date_series(instrument_type, DATA_FOLDER_PATH)
    md_df = load_data(f'{instrument_type}.csv', folder_path=DATA_FOLDER_PATH
                      ).set_index('trade_date')[ohlcav_col_name_list]
    md_df.index = pd.DatetimeIndex(md_df.index)
    data_len = md_df.shape[0]
    # 每40天为一个 2π 周期
    open_price = close_price = np.sin(np.linspace(0, data_len * np.pi / period_len, data_len)) * 100 + 1000
    high_price = open_price + 10
    low_price = open_price - 10
    md_df["open"] = open_price
    md_df["high"] = high_price
    md_df["low"] = low_price
    md_df["close"] = close_price
    from ibats_common.backend.factor import get_factor, transfer_2_batch
    factors_df = get_factor(md_df, trade_date_series, delivery_date_series, dropna=True)
    df_index, df_columns, data_arr_batch = transfer_2_batch(factors_df, n_step=n_step)
    md_df = md_df.loc[df_index, :]
    return md_df[['close', 'open']], data_arr_batch


def get_env(state_with_flag=True, is_continuous_action=False, is_mock_data=False, **kwargs):
    from tf_agents.environments.tf_py_environment import TFPyEnvironment
    if is_mock_data:
        md_df, data_factors = _get_mock_df()
    else:
        md_df, data_factors = _get_df()

    env = TFPyEnvironment(AccountEnv(
        md_df=md_df, data_factors=data_factors, state_with_flag=state_with_flag,
        is_continuous_action=is_continuous_action, **kwargs))
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
        from tf_agents.environments.tf_py_environment import TFPyEnvironment
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


def _test_get_mock_df():
    import matplotlib.pyplot as plt
    md_df, data_arr_batch = _get_mock_df()
    plt.plot(md_df['close'])
    plt.show()


if __name__ == "__main__":
    # account_env_test()
    _test_get_mock_df()
