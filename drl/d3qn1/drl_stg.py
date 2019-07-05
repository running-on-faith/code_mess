#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 19-7-5 上午9:45
@File    : drl_stg.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import os
import re
from datetime import datetime, timedelta

import numpy as np
from ibats_utils.mess import copy_module_file_to, date_2_str, get_last, datetime_2_str

from ibats_common import module_root_path
from ibats_common.backend.factor import get_factor, transfer_2_batch
from ibats_common.backend.rl.emulator.account import Account
from ibats_common.common import BacktestTradeMode, ContextKey, CalcMode
from ibats_common.example import get_trade_date_series, get_delivery_date_series
from ibats_common.example.drl.d3qn1.agent.main import Agent
from ibats_common.strategy import StgBase
from ibats_common.strategy_handler import strategy_handler_factory
from ibats_local_trader.agent.md_agent import *
from ibats_local_trader.agent.td_agent import *

logger = logging.getLogger(__name__)


class DRLStg(StgBase):

    def __init__(self, instrument_type, unit=1):
        super().__init__()
        self.unit = unit
        self.instrument_type = instrument_type
        # 模型运行所需参数
        self.enable_load_model_if_exist = True
        self.retrain_period = 30  # 60 每隔60天重新训练一次，0 则为不进行重新训练
        self.trade_date_last_train = None
        self.do_nothing_on_min_bar = False  # 仅供调试使用
        # 模型训练所需参数
        self._factor_df = None
        self._batch_factors = None
        self._data_factors_latest_date = None
        self._md_df = None
        self._env = None
        self._agent = None
        self.num_episodes = 500
        self.target_step_size = 128
        self.train_step_size = 32
        self.show_log_pre_n_loop = 50
        self.benchmark_cagr = 0.05
        # 模型因子构建所需参数
        self.input_size = 38
        self.n_step = 60
        self.ohlcav_col_name_list = ["open", "high", "low", "close", "amount", "volume"]
        self.trade_date_series = get_trade_date_series()
        self.delivery_date_series = get_delivery_date_series(instrument_type)
        # 模型保存路径相关参数
        # 是否使用现有模型进行操作，如果是记得修改以下下方的路径
        # enable_load_model_if_exist 将会在调用 self.load_model_if_exist 时进行检查
        # 如果该字段为 False，调用 load_model_if_exist 时依然可以传入参数的方式加载已有模型
        # 该字段与 self.load_model_if_exist 函数的 enable_load_model_if_exist参数是 “or” 的关系
        self.enable_load_model_if_exist = False
        if self.enable_load_model_if_exist:
            self.base_folder_path = folder_path = os.path.join(module_root_path, 'tf_saves_2019-06-27_16_24_34')
        else:
            datetime_str = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
            self.base_folder_path = folder_path = os.path.join(module_root_path, f'tf_saves_{datetime_str}')
        # 备份文件
        file_path = copy_module_file_to(self.__class__, folder_path)
        self.logger.debug('文件已经备份到：%s', file_path)
        # 设置模型备份目录
        self.model_folder_path = model_folder_path = os.path.join(folder_path, 'model_tfls')
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path, exist_ok=True)
        # 设置日至备份目录
        self.tensorboard_dir = tensorboard_dir = os.path.join(folder_path, f'tensorboard_logs')
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)

    def get_model(self, rebuild_model=False) -> Agent:
        """
        建立模型前必须 get_factor 调用完毕，需要用到 self.input_size
        :param rebuild_model:
        :return:
        """
        if self._agent is None or rebuild_model:
            if self._agent is not None:
                self._agent.close()
            self._agent = Agent(input_shape=self.input_size)
        return self._agent

    def get_factor(self, md_df):
        trade_date_latest = md_df.index[-1]
        if self._data_factors_latest_date is None or self._data_factors_latest_date < trade_date_latest:
            factors_df = get_factor(
                md_df,
                ohlcav_col_name_list=self.ohlcav_col_name_list,
                trade_date_series=self.trade_date_series,
                delivery_date_series=self.delivery_date_series,
                dropna=True)
            df_index, df_columns, data_arr_batch = transfer_2_batch(factors_df, n_step=self.n_step)
            shape = [data_arr_batch.shape[0], 5, int(self.n_step / 5), data_arr_batch.shape[2]]
            self._batch_factors = np.transpose(data_arr_batch.reshape(shape), [0, 2, 3, 1])
            self._data_factors_latest_date = trade_date_latest
            self._factor_df = factors_df.loc[df_index, :]
            self._md_df = md_df.loc[df_index, :]
            if self.input_size is None or self.input_size != self._batch_factors.shape[1]:
                self.input_size = self._batch_factors.shape[1]
                self.logger.warning("set input_size: %d", self.input_size)

        return self._md_df, self._factor_df, self._batch_factors

    def get_date_file_path_pair_list(self):
        """
        目录结构：
        tf_saves_2019-06-05_16_21_39
          *   model_tfls
          *       *   2012-12-31
          *       *       *   checkpoint
          *       *       *   model_dqn_0.tfl.data-00000-of-00001
          *       *       *   model_dqn_0.tfl.index
          *       *       *   model_dqn_0.tfl.meta
          *       *   2013-02-28
          *       *       *   checkpoint
          *       *       *   model_dqn_3.tfl.data-00000-of-00001
          *       *       *   model_dqn_3.tfl.index
          *       *       *   model_dqn_3.tfl.meta
          *   tensorboard_logs
          *       *   2012-12-31_496_20190605_184316
          *       *       *   events.out.tfevents.1559731396.mg-ubuntu64
          *       *   2013-02-28_496_20190605_184716
          *       *       *   events.out.tfevents.1559731396.mg-ubuntu64
        :return:
        """
        # 获取全部文件名
        pattern = re.compile(r'model_dqn_\d+.tfl')
        date_file_path_pair_list, model_name_set = [], set()
        for folder_name in os.listdir(self.model_folder_path):
            folder_path = os.path.join(self.model_folder_path, folder_name)
            if os.path.isdir(folder_path):
                try:
                    # 获取 trade_date_last_train
                    key = str_2_date(folder_name)
                    for file_name in os.listdir(folder_path):
                        # 对下列有效文件名，匹配结果："model_dqn_0.tfl"
                        # model_dqn_0.tfl.data-00000-of-00001
                        # model_dqn_0.tfl.index
                        # model_dqn_0.tfl.meta
                        m = pattern.search(file_name)
                        if m is None:
                            continue
                        model_name = m.group()
                        if key in model_name_set:
                            continue
                        model_name_set.add(key)
                        # 获取 model folder_path
                        file_path = os.path.join(folder_path, model_name)

                        date_file_path_pair_list.append([key, file_path])
                except:
                    pass

        return date_file_path_pair_list

    def load_model_if_exist(self, trade_date, enable_load_model_if_exist=False):
        """
        将模型导出到文件
        目录结构：
        tf_saves_2019-06-05_16_21_39
          *   model_tfls
          *       *   2012-12-31
          *       *       *   checkpoint
          *       *       *   model_dqn_0.tfl.data-00000-of-00001
          *       *       *   model_dqn_0.tfl.index
          *       *       *   model_dqn_0.tfl.meta
          *       *   2013-02-28
          *       *       *   checkpoint
          *       *       *   model_dqn_3.tfl.data-00000-of-00001
          *       *       *   model_dqn_3.tfl.index
          *       *       *   model_dqn_3.tfl.meta
          *   tensorboard_logs
          *       *   2012-12-31_496[1]_20190605_184316
          *       *       *   events.out.tfevents.1559731396.mg-ubuntu64
          *       *   2013-02-28_496[1]_20190605_184716
          *       *       *   events.out.tfevents.1559731396.mg-ubuntu64
        :param enable_load_model_if_exist:
        :param trade_date:
        :return:
        """
        if self.enable_load_model_if_exist or enable_load_model_if_exist:
            # 获取小于等于当期交易日的最大的一个交易日对应的文件名
            min_available_date = str_2_date(trade_date) - timedelta(days=self.retrain_period)
            self.logger.debug('尝试加载现有模型，[%s - %s] %d 天', min_available_date, trade_date, self.retrain_period)
            date_file_path_pair_list = [_ for _ in self.get_date_file_path_pair_list() if _[0] >= min_available_date]
            if len(date_file_path_pair_list) > 0:
                # 按日期排序
                date_file_path_pair_list.sort(key=lambda x: x[0])
                # 获取小于等于当期交易日的最大的一个交易日对应的文件名
                # file_path = get_last(date_file_path_pair_list, lambda x: x[0] <= trade_date, lambda x: x[1])
                trade_date = str_2_date(trade_date)
                ret = get_last(date_file_path_pair_list, lambda x: x[0] <= trade_date)
                if ret is not None:
                    key, folder_path = ret
                    if folder_path is not None:
                        model = self.get_model(rebuild_model=True)  # 这句话是必须的，需要实现建立模型才可以加载
                        model.restore_model(folder_path)
                        self.trade_date_last_train = key
                        self.logger.info("加载模型成功。trade_date_last_train: %s load from path: %s", key, folder_path)
                        return True

        return False

    def train(self, md_df, batch_factors, has_try_n_times=None, show_plot=False) -> pd.DataFrame:
        """

        :param md_df:
        :param batch_factors:
        :param has_try_n_times:
        :param show_plot:
        :return: pd.DataFrame columns:["value", "reward", "cash", "action"]
        """
        agent = self._agent
        env = Account(md_df, data_factors=batch_factors)

        episodes_train = []
        global_step = 0
        for episode in range(self.num_episodes):
            state = env.reset()
            episode_step = 0
            while True:
                global_step += 1
                episode_step += 1

                action = agent.choose_action_stochastic(state)
                next_state, reward, done = env.step(action)
                agent.update_cache(state, action, reward, next_state, done)
                state = next_state

                if global_step % self.target_step_size == 0 or done:
                    agent.update_target()
                    # print('global_step=%d, episode_step=%d, agent.update_target()' % (global_step, episode_step))

                if episode_step % self.train_step_size == 0 or done:
                    agent.update_eval()
                    # print('global_step=%d, episode_step=%d, agent.update_eval()' % (global_step, episode_step))

                    if done:
                        # print("episode=%d, data_observation.shape[0]=%d, env.A.total_value=%f" % (
                        #     episode, env.A.data_observation.shape[0], env.A.total_value))
                        if episode % self.show_log_pre_n_loop == 0 or episode == self.num_episodes - 1:
                            episodes_train.append(env.plot_data())
                            if has_try_n_times is None or has_try_n_times == 1:
                                print("episode=%d, data_observation.shape[0]=%d, env.A.total_value=%f" % (
                                    episode, env.A.data_observation.shape[0], env.A.total_value))
                            else:
                                print("round=%d, episode=%d, env.A.total_value=%f" % (
                                    has_try_n_times, episode, env.A.total_value))
                        break

        if show_plot:
            import matplotlib.pyplot as plt
            reward_df = env.plot_data()
            value_s = reward_df.iloc[:, 0]
            value_s.plot()  # figsize=(16, 6)
            plt.suptitle(datetime_2_str(datetime.now()))
            plt.show()
            value_df = pd.DataFrame({num: df['value']
                                     for num, df in enumerate(episodes_train, start=1)
                                     if df.shape[0] > 0})
            value_df.plot()
            plt.suptitle(datetime_2_str(datetime.now()))
            plt.show()
        else:
            reward_df = env.plot_data()

        path = agent.save_model()
        print('model save to path:', path)
        agent.close()
        return reward_df

    def save_model(self, trade_date, key=None):
        """
        将模型导出到文件
        :param trade_date:
        :param key:
        :return:
        """
        folder_path = os.path.join(self.model_folder_path, date_2_str(trade_date))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(
            folder_path,
            f"model_dqn_{key}.tfl" if key is not None else f"model_dqn_0.tfl")
        self._agent.save_model(path=file_path)
        self.logger.info("模型训练截止日期： %s 保存到: %s", self.trade_date_last_train, file_path)
        return file_path

    def load_train_test(self, indexed_df, enable_load_model, rebuild_model=False, enable_train_if_load_not_suss=True,
                        enable_train_even_load_succ=False, enable_predict_test=False):
        if rebuild_model or self._agent is None:
            self.get_factor(indexed_df)
            self.get_model(rebuild_model=True)

        trade_date = str_2_date(indexed_df.index[-1])
        # 加载模型
        if enable_load_model:
            is_load = self.load_model_if_exist(trade_date)
        else:
            is_load = False

        if enable_train_even_load_succ or (enable_train_if_load_not_suss and not is_load):
            md_df, factor_df, batch_factors = self.get_factor(indexed_df)
            num = 0
            while True:
                num += 1
                if num > 1:
                    # 重新构建模型，原来模型训练失败，需要重新构建模型
                    self.get_model(rebuild_model=True)
                # 训练模型
                reward_df = self.train(md_df, batch_factors, has_try_n_times=num)
                stats = reward_df['value'].calc_stats()
                total_return = stats['total_return']
                if total_return < 0:
                    self.logger.warning('第 %d 次训练，样本内收益率 rr=%.2f%% 过低，重新采样训练',
                                        num, total_return * 100)
                    continue
                cagr = stats['cagr']
                if np.isnan(cagr) or cagr < self.benchmark_cagr:
                    self.logger.warning('第 %d 次训练，cagr=%.2f，重新采样训练',
                                        num, cagr * 100, self.benchmark_cagr * 100)
                    continue
                else:
                    break

            self.save_model(trade_date)
            self.trade_date_last_train = trade_date
        else:
            factor_df = get_factor(indexed_df, ohlcav_col_name_list=self.ohlcav_col_name_list,
                                   trade_date_series=self.trade_date_series,
                                   delivery_date_series=self.delivery_date_series)
            # train_acc, val_acc = self.valid_model_acc(factor_df)

        # enable_test 默认为 False
        # self.valid_model_acc(factor_df) 以及完全取代 self.predict_test
        # self.predict_test 仅用于内部测试使用
        # if enable_predict_test:
        #     self.predict_test(factor_df)

        return factor_df

    def on_prepare_min1(self, md_df, context):
        if md_df is None:
            return
        indexed_df = md_df.set_index('trade_date').drop('instrument_type', axis=1)
        indexed_df.index = pd.DatetimeIndex(indexed_df.index)

        self.load_train_test(indexed_df, enable_load_model=self.enable_load_model_if_exist)

    def predict_latest(self):
        """
        对最新状态进行预测
        :return: action: 0 long, 1, short, 0, close
        """
        env = Account(self._md_df, data_factors=self._batch_factors)
        latest_state = env.latest_state()
        # TODO: 加入强化学习机制，目前只做action计算，没有进行状态更新
        action = self._agent.choose_action_deterministic(latest_state)
        return action

    def on_min1(self, md_df, context):
        if self.do_nothing_on_min_bar:  # 仅供调试使用
            return

        # 数据整理
        indexed_df = md_df.set_index('trade_date').drop('instrument_type', axis=1)
        indexed_df.index = pd.DatetimeIndex(indexed_df.index)
        # 获取最新交易日
        trade_date = str_2_date(indexed_df.index[-1])
        days_after_last_train = (trade_date - self.trade_date_last_train).days
        if self.retrain_period is not None and 0 < self.retrain_period < days_after_last_train:
            # 重新训练
            self.logger.info('当前日期 %s 距离上一次训练 %s 已经过去 %d 天，重新训练',
                             trade_date, self.trade_date_last_train, days_after_last_train)
            self.load_train_test(indexed_df, rebuild_model=True,
                                 enable_load_model=self.enable_load_model_if_exist)
        else:
            self.get_factor(indexed_df)

        # 预测
        action = self.predict_latest()
        is_empty, is_buy, is_sell = action == 2, action == 0, action == 1
        # logger.info('%s is_buy=%s, is_sell=%s', trade_date, str(is_buy), str(is_sell))
        close = md_df['close'].iloc[-1]
        instrument_id = context[ContextKey.instrument_id_list][0]
        if is_buy:  # is_buy
            # position_date_pos_info_dic = self.get_position(instrument_id)
            # no_target_position = True
            # if position_date_pos_info_dic is not None:
            #     for position_date, pos_info in position_date_pos_info_dic.items():
            #         if pos_info.position == 0:
            #             continue
            #         direction = pos_info.direction
            #         if direction == Direction.Short:
            #             self.close_short(instrument_id, close, pos_info.position)
            #         elif direction == Direction.Long:
            #             no_target_position = False
            # if no_target_position:
            #     self.open_long(instrument_id, close, self.unit)
            # else:
            #     self.logger.debug("%s %s     %.2f holding", self.trade_agent.curr_timestamp, instrument_id, close)
            self.keep_long(instrument_id, close, self.unit)

        if is_sell:  # is_sell
            # position_date_pos_info_dic = self.get_position(instrument_id)
            # no_holding_target_position = True
            # if position_date_pos_info_dic is not None:
            #     for position_date, pos_info in position_date_pos_info_dic.items():
            #         if pos_info.position == 0:
            #             continue
            #         direction = pos_info.direction
            #         if direction == Direction.Long:
            #             self.close_long(instrument_id, close, pos_info.position)
            #         elif direction == Direction.Short:
            #             no_holding_target_position = False
            # if no_holding_target_position:
            #     self.open_short(instrument_id, close, self.unit)
            # else:
            #     self.logger.debug("%s %s     %.2f holding", self.trade_agent.curr_timestamp, instrument_id, close)
            self.keep_short(instrument_id, close, self.unit)

        if is_empty:
            # position_date_pos_info_dic = self.get_position(instrument_id)
            # if position_date_pos_info_dic is not None:
            #     for position_date, pos_info in position_date_pos_info_dic.items():
            #         if pos_info.position == 0:
            #             continue
            #         direction = pos_info.direction
            #         if direction == Direction.Long:
            #             self.close_long(instrument_id, close, pos_info.position)
            #         elif direction == Direction.Short:
            #             self.close_short(instrument_id, close, pos_info.position)
            self.keep_empty(instrument_id, close)


def _test_use(is_plot):
    from ibats_common import module_root_path
    import os
    # 参数设置
    run_mode = RunMode.Backtest_FixPercent
    calc_mode = CalcMode.Normal
    strategy_params = {'unit': 1}
    md_agent_params_list = [{
        'md_period': PeriodType.Min1,
        'instrument_id_list': ['RB'],
        'datetime_key': 'trade_date',
        'init_md_date_from': '1995-1-1',  # 行情初始化加载历史数据，供策略分析预加载使用
        'init_md_date_to': '2010-1-1',
        # 'C:\GitHub\IBATS_Common\ibats_common\example\ru_price2.csv'
        'file_path': os.path.abspath(os.path.join(module_root_path, 'example', 'data', 'RB.csv')),
        'symbol_key': 'instrument_type',
    }]
    if run_mode == RunMode.Realtime:
        trade_agent_params = {
        }
        strategy_handler_param = {
        }
    elif run_mode == RunMode.Backtest:
        trade_agent_params = {
            'trade_mode': BacktestTradeMode.Order_2_Deal,
            'init_cash': 1000000,
            "calc_mode": calc_mode,
        }
        strategy_handler_param = {
            'date_from': '2010-1-1',  # 策略回测历史数据，回测指定时间段的历史行情
            'date_to': '2018-10-18',
        }
    else:
        # RunMode.Backtest_FixPercent
        trade_agent_params = {
            'trade_mode': BacktestTradeMode.Order_2_Deal,
            "calc_mode": calc_mode,
        }
        strategy_handler_param = {
            'date_from': '2010-1-1',  # 策略回测历史数据，回测指定时间段的历史行情
            'date_to': '2018-10-18',
        }

    # 初始化策略处理器
    stghandler = strategy_handler_factory(
        stg_class=DRLStg,
        strategy_params=strategy_params,
        md_agent_params_list=md_agent_params_list,
        exchange_name=ExchangeName.LocalFile,
        run_mode=run_mode,
        trade_agent_params=trade_agent_params,
        strategy_handler_param=strategy_handler_param,
    )
    stghandler.start()
    time.sleep(10)
    stghandler.keep_running = False
    stghandler.join()
    stg_run_id = stghandler.stg_run_id
    logging.info("执行结束 stg_run_id = %d", stg_run_id)

    if is_plot:
        # from ibats_common.analysis.plot_db import show_order, show_cash_and_margin, show_rr_with_md
        # from ibats_common.analysis.summary import summary_rr
        # show_order(stg_run_id)
        # df = show_cash_and_margin(stg_run_id)
        # sum_df, symbol_rr_dic, save_file_path_dic = show_rr_with_md(stg_run_id)
        # for symbol, rr_df in symbol_rr_dic.items():
        #     col_transfer_dic = {'return': rr_df.columns}
        #     summary_rr(rr_df, figure_4_each_col=True, col_transfer_dic=col_transfer_dic)
        from ibats_common.analysis.summary import summary_stg_2_docx
        from ibats_utils.mess import open_file_with_system_app
        file_path = summary_stg_2_docx(stg_run_id)
        if file_path is not None:
            open_file_with_system_app(file_path)

    return stg_run_id


if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG, format=config.LOG_FORMAT)
    _test_use(is_plot=True)
