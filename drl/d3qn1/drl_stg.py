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

import ffn
import numpy as np
from ibats_utils.mess import date_2_str, get_last, datetime_2_str, copy_file_to, get_module_file_path, copy_folder_to

from ibats_common import module_root_path
from ibats_common.analysis.plot import show_drl_accuracy
from ibats_common.analysis.summary import summary_release_2_docx
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
logger.debug('import %s', ffn)


class DRLStg(StgBase):

    def __init__(self, instrument_type, unit=1):
        super().__init__()
        self.unit = unit
        self.instrument_type = instrument_type
        # 模型运行所需参数
        self.enable_load_model_if_exist = True
        self.retrain_period = 60  # 60 每隔60天重新训练一次，0 则为不进行重新训练
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
        self.train_step_size = 64
        self.show_log_pre_n_loop = 50
        self.benchmark_cagr = None  # 0.05              # 如果为空则不检查此项
        self.benchmark_total_return = 0.00  # 如果为空则不检查此项
        # 模型因子构建所需参数
        self.input_shape = [None, 12, 93, 5]
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
            self.base_folder_path = folder_path = os.path.join(module_root_path, 'tf_saves_2019-07-08_12_32_12')
        else:
            datetime_str = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
            self.base_folder_path = folder_path = os.path.join(module_root_path, f'tf_saves_{datetime_str}')
        # 备份文件(将策略文件所在目录全部拷贝到备份目录下)
        file_path = get_module_file_path(self.__class__)
        source_folder_path = os.path.split(file_path)[0]
        file_path = copy_folder_to(source_folder_path, folder_path)
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
                try:
                    self._agent.close()
                except:
                    self.logger.exception('agent.close() exception')
            self._agent = Agent(input_shape=self.input_shape)
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
            if self.input_shape is None or self.input_shape[1:] != self._batch_factors.shape[1:]:
                self.input_shape = self._batch_factors.shape
                self.logger.warning("set input_size: %s", self.input_shape)

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

    def load_model_if_exist(self, trade_date, enable_load_model_if_exist=False, rebuild_model=True):
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
                        model = self.get_model(rebuild_model=rebuild_model)  # 这句话是必须的，需要实现建立模型才可以加载
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
        self._env = env = Account(md_df, data_factors=batch_factors)

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
                                self.logger.debug("episode=%d, data_observation.shape[0]=%d, env.A.total_value=%f",
                                                  episode, env.A.data_observation.shape[0], env.A.total_value)
                            else:
                                self.logger.debug("round=%d, episode=%d, env.A.total_value=%f",
                                                  has_try_n_times, episode, env.A.total_value)
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

        # path = agent.save_model()
        # self.logger.info('model save to path:%s', path)
        # agent.close()
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
            rebuild_model_when_load = False
        else:
            rebuild_model_when_load = True

        trade_date = str_2_date(indexed_df.index[-1])
        # 加载模型
        if enable_load_model:
            is_load = self.load_model_if_exist(trade_date, rebuild_model=rebuild_model_when_load)
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
                total_return = stats.total_return
                if self.benchmark_total_return is not None and total_return < self.benchmark_total_return:
                    self.logger.warning('第 %d 次训练，样本内收益率 rr=%.2f%% 过低，重新采样训练',
                                        num, total_return * 100)
                    continue
                cagr = stats.cagr
                if self.benchmark_cagr is not None and (np.isnan(cagr) or cagr < self.benchmark_cagr):
                    self.logger.warning('第 %d 次训练，cagr=%.2f < %.2f，重新采样训练',
                                        num, cagr * 100, self.benchmark_cagr * 100)
                    continue
                else:
                    self.logger.debug('第 %d 次训练 rr=%.2f%%，cagr=%.2f，重新采样训练',
                                      num, total_return * 100, cagr * 100)
                    break

            self.save_model(trade_date)
            self.trade_date_last_train = trade_date
        else:
            md_df, factor_df, batch_factors = self.get_factor(indexed_df)

        return factor_df

    def on_prepare_min1(self, md_df, context):
        if md_df is None:
            return
        indexed_df = md_df.set_index('trade_date').drop('instrument_type', axis=1)
        indexed_df.index = pd.DatetimeIndex(indexed_df.index)

        self.load_train_test(indexed_df, enable_load_model=self.enable_load_model_if_exist)

    def predict_latest_action(self):
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
        action = self.predict_latest_action()
        is_empty, is_buy, is_sell = action == 2, action == 0, action == 1
        # logger.info('%s is_buy=%s, is_sell=%s', trade_date, str(is_buy), str(is_sell))
        close = md_df['close'].iloc[-1]
        instrument_id = context[ContextKey.instrument_id_list][0]
        if is_buy:  # is_buy
            self.keep_long(instrument_id, close, self.unit)

        if is_sell:  # is_sell
            self.keep_short(instrument_id, close, self.unit)

        if is_empty:
            self.keep_empty(instrument_id, close)

    def predict(self):
        agent = self._agent
        env = Account(self._md_df, data_factors=self._batch_factors)
        episodes_train = []

        state = env.reset()
        episode_step = 0
        while True:
            episode_step += 1

            action = agent.choose_action_deterministic(state)
            next_state, reward, done = env.step(action)
            # agent.update_cache(state, action, reward, next_state, done)
            state = next_state

            if done:
                # print("episode=%d, data_observation.shape[0]=%d, env.A.total_value=%f" % (
                #     episode, env.A.data_observation.shape[0], env.A.total_value))
                episodes_train.append(env.plot_data())
                self.logger.debug("data_observation.shape[0]=%d, env.A.total_value=%f",
                                  env.A.data_observation.shape[0], env.A.total_value)
                break

        reward_df = env.plot_data()
        return reward_df

    def calc_real_label(self) -> np.ndarray:
        close_s = self._env.A.data_close
        fee = self._env.A.fee
        rr = close_s.to_returns()
        rr_shift = rr.shift(-1).fillna(0)
        real_label = np.zeros(close_s.shape)
        real_label[rr_shift > fee] = 1
        real_label[rr_shift < -fee] = 2
        return real_label

    def on_min1_release(self, md_df):
        """
        增加模型对未来数据预测成功率走势图展示
        :param md_df:
        :return:
        """
        if md_df is None or md_df.shape[0] == 0:
            self.logger.warning('md_df is None or shape[0] == 0')
            return
        else:
            self.logger.debug('md_df.shape= %s', md_df.shape)

        # 获取各个模型训练时间点及路径
        date_file_path_pair_list = self.get_date_file_path_pair_list()
        if len(date_file_path_pair_list) > 0:
            # 按日期排序
            date_file_path_pair_list.sort(key=lambda x: x[0])

        # 建立数据集
        indexed_df = md_df.set_index('trade_date').drop('instrument_type', axis=1)
        trade_date_end = indexed_df.index[-1]
        indexed_df, factor_df, batch_factors = self.get_factor(indexed_df)
        trade_date_index = indexed_df.index
        data_len = indexed_df.shape[0]
        if data_len == 0:
            self.logger.warning('ys 长度为0，请检查是否存在数据错误')
            return
        # trade_date2_list 对日期做一次 shift(-1) 操作
        trade_date2_list = [_[0] for _ in date_file_path_pair_list][1:]
        trade_date2_list.append(None)
        # 预测结果
        self.logger.info("按日期分段验证检验预测结果")
        real_label_s = self.calc_real_label()
        action_concat_s, real_ys_tot, img_meta_dic_list = [], [], []
        # 根据模型 trade_date_last_train 进行分段预测，并将结果记录到 pred_ys
        for num, ((trade_date_last_train, file_path, predict_test_random_state), trade_date_next) in enumerate(zip(
                date_file_path_pair_list, trade_date2_list)):
            # 以模型训练日期为基准，后面的数据作为验证集数据（样本外数据）
            # 获取有效的日期范围 from - to
            range_from_arr = trade_date_index >= pd.to_datetime(trade_date_last_train)
            range_from_len = len(range_from_arr)
            if range_from_len == 0:  # range_from_len 应该与 trade_date_list_count 等长度，所以这个条件应该永远不会满足
                self.logger.error('总共%d条数据，%s 开始后面没有可验证数据', data_len, trade_date_last_train)
                continue
            true_count = sum(range_from_arr)
            self.logger.debug("len(range_from)=%d, True Count=%d", len(range_from_arr), true_count)
            if true_count == 0:
                self.logger.warning('总共%d条数据，%s 开始后面没有可验证数据', data_len, trade_date_last_train)
                continue
            # 自 trade_date_last_train 起的所有有效日期
            trade_date_list_sub = trade_date_index[range_from_arr]

            # 获取 in_range，作为 range_from, range_to 的交集
            if trade_date_next is None:
                in_range_arr = None
                in_range_count = true_count
            else:
                in_range_arr = trade_date_list_sub < pd.to_datetime(trade_date_next)
                in_range_count = sum(in_range_arr)
                if in_range_count == 0:
                    self.logger.warning('总共%d条数据，[%s - %s) 之间没有可用数据',
                                        data_len, trade_date_last_train, trade_date_next)
                    continue
                else:
                    self.logger.debug('总共%d条数据，[%s - %s) 之间有 %d 条数据将被验证 model path:%s',
                                      data_len, trade_date_last_train, trade_date_next, in_range_count, file_path)

            # 获取当前时段对应的 xs
            # 进行验证时，对 range_from 开始的全部数据进行预测，按照 range_to 为分界线分区着色显示
            close_df = indexed_df.loc[trade_date_list_sub, 'close']

            # 加载模型
            is_load = self.load_model_if_exist(trade_date_last_train, enable_load_model_if_exist=True)
            if not is_load:
                self.logger.error('%s 模型加载失败：%s', trade_date_last_train, file_path)
                continue
            # 预测
            reward_df = self.predict()
            action_s = reward_df['action']
            if in_range_arr is not None and in_range_count > 0:
                action_concat_s.extend(action_s[in_range_arr])
            else:
                action_concat_s.extend(action_s[trade_date_list_sub])

            # 为每一个时段单独验证成功率，以当前模型为基准，验证后面全部历史数据成功率走势
            if trade_date_next is None:
                split_point_list = None
            else:
                split_point_list = [close_df.index[0], trade_date_next, close_df.index[-1]]

            img_file_path = show_drl_accuracy(real_label_s, action_s, close_df, split_point_list)
            img_meta_dic_list.append({
                'img_file_path': img_file_path,
                'trade_date_last_train': trade_date_last_train,
                'module_file_path': file_path,
                'predict_test_random_state': predict_test_random_state,
                'split_point_list': split_point_list,
                'in_range_count': in_range_count,
                'trade_date_end': trade_date_end,
            })

        # action_concat_s = np.array(action_concat_s)
        trade_date_last_train_first = pd.to_datetime(date_file_path_pair_list[0][0])
        split_point_list = [_[0] for _ in date_file_path_pair_list]
        split_point_list.append(trade_date_index[-1])
        # 获取 real_ys
        close_df = indexed_df.loc[trade_date_index[trade_date_index >= trade_date_last_train_first], 'close']
        img_file_path = show_drl_accuracy(real_label_s, action_concat_s, close_df, split_point_list)

        img_meta_dic_list.append({
            'img_file_path': img_file_path,
            'trade_date_last_train': trade_date_last_train_first,
            'module_file_path': date_file_path_pair_list[0][1],
            'predict_test_random_state': date_file_path_pair_list[0][2],
            'split_point_list': split_point_list,
            'in_range_count': close_df.shape[0],
            'trade_date_end': trade_date_end,
        })
        is_output_docx = True
        if is_output_docx:
            title = f"[{self.stg_run_id}] predict accuracy trend report"
            file_path = summary_release_2_docx(title, img_meta_dic_list)
            copy_file_to(file_path, self.base_folder_path)


def _test_use(is_plot):
    from ibats_common import module_root_path
    import os
    instrument_type = 'RB'
    # 参数设置
    run_mode = RunMode.Backtest_FixPercent
    calc_mode = CalcMode.Normal
    strategy_params = {'instrument_type': instrument_type, 'unit': 1}
    md_agent_params_list = [{
        'md_period': PeriodType.Min1,
        'instrument_id_list': [instrument_type],
        'datetime_key': 'trade_date',
        'init_md_date_from': '1995-1-1',  # 行情初始化加载历史数据，供策略分析预加载使用
        'init_md_date_to': '2013-1-1',
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
            'date_from': '2013-1-1',  # 策略回测历史数据，回测指定时间段的历史行情
            'date_to': '2018-10-18',
        }
    else:
        # RunMode.Backtest_FixPercent
        trade_agent_params = {
            'trade_mode': BacktestTradeMode.Order_2_Deal,
            "calc_mode": calc_mode,
        }
        strategy_handler_param = {
            'date_from': '2013-1-1',  # 策略回测历史数据，回测指定时间段的历史行情
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
