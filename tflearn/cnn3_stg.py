#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 19-5-30 上午8:40
@File    : rnn3_stg.py
@contact : mmmaaaggg@163.com
@desc    : CNN 三分类预测
pip3 install tensorflow sklearn tflearn
2019-06-06
1）  模型及日志目录结构调整如下：
    tf_saves_2019-06-05_16_21_39
      *   model_tfls
      *       *   2012-12-31
      *       *       *   checkpoint
      *       *       *   model_-54_51.tfl.data-00000-of-00001
      *       *       *   model_-54_51.tfl.index
      *       *       *   model_-54_51.tfl.meta
      *       *   2013-02-28
      *       *       *   checkpoint
      *       *       *   model_-54_51.tfl.data-00000-of-00001
      *       *       *   model_-54_51.tfl.index
      *       *       *   model_-54_51.tfl.meta
      *   tensorboard_logs
      *       *   2012-12-31_496[1]_20190605_184316
      *       *       *   events.out.tfevents.1559731396.mg-ubuntu64        datetime_str = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    self.base_folder_path = folder_path = os.path.join(get_report_folder_path(self.stg_run_id), f'tf_saves_{datetime_str}')

      *       *   2013-02-28_496[1]_20190605_184716
      *       *       *   events.out.tfevents.1559731396.mg-ubuntu64

2）  调整restore功能
3）  增加模型对未来数据预测成功率走势图
2019-06-12
对原始数据的 OHLCA 乘以一个因子以扩充样本数据，相应的调整 get_factor 方法及相关代码
2019-07-01
继承自 AIStgBase 类
"""
import os
import random
from collections import defaultdict
from datetime import datetime

import numpy as np
import tflearn
from tflearn import conv_2d, max_pool_2d, local_response_normalization, fully_connected, dropout

from ibats_common.backend.mess import get_report_folder_path
from ibats_common.backend.factor import get_factor
from ibats_common.backend.label import calc_label3
from ibats_common.common import BacktestTradeMode, ContextKey, Direction, CalcMode
from ibats_common.example import AIStgBase
from ibats_common.example.data import get_trade_date_series, get_delivery_date_series
from ibats_common.strategy_handler import strategy_handler_factory
from ibats_local_trader.agent.md_agent import *
from ibats_local_trader.agent.td_agent import *

logger = logging.getLogger(__name__)


class AIStg(AIStgBase):

    def __init__(self, instrument_type, unit=1):
        super().__init__(instrument_type, unit=1)
        self.unit = unit
        # 模型相关参数
        self.input_size = 38
        self.batch_size = 128
        self.n_step = 60
        self.output_size = 3
        self.n_hidden_units = 10
        self.lr = 0.006
        # 模型训练，及数据集相关参数
        self._model = None
        self._session = None
        self.train_validation_rate = 0.8
        self.xs_train, self.xs_validation, self.ys_train, self.ys_validation = None, None, None, None
        self.label_func_max_rr = 0.0051
        self.label_func_min_rr = -0.0054
        self.max_future = 3
        self.predict_test_random_state = None
        self.n_epoch = 48
        self.retrain_period = 60
        self.validation_accuracy_base_line = 0.55  # 0.6    # 如果为 None，则不进行 validation 成功率检查
        self.over_fitting_train_acc = 0.9  # 过拟合训练集成功率，如果为None则不进行判断
        # 其他辅助信息
        self.trade_date_series = get_trade_date_series()
        self.delivery_date_series = get_delivery_date_series(instrument_type)
        self.tensorboard_verbose = 3
        # 模型保存路径相关参数
        # 是否使用现有模型进行操作，如果是记得修改以下下方的路径
        # enable_load_model_if_exist 将会在调用 self.load_model_if_exist 时进行检查
        # 如果该字段为 False，调用 load_model_if_exist 时依然可以传入参数的方式加载已有模型
        # 该字段与 self.load_model_if_exist 函数的 enable_load_model_if_exist参数是 “or” 的关系
        self.enable_load_model_if_exist = False
        if self.enable_load_model_if_exist:
            self.base_folder_path = folder_path = os.path.join(
                get_report_folder_path(self.stg_run_id), f'tf_saves_2019-06-16_09_33_30')
        else:
            datetime_str = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
            self.base_folder_path = folder_path = os.path.join(
                get_report_folder_path(self.stg_run_id), f'tf_saves_{datetime_str}')

        self.model_folder_path = model_folder_path = os.path.join(folder_path, 'model_tfls')
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path, exist_ok=True)
        self.tensorboard_dir = tensorboard_dir = os.path.join(folder_path, f'tensorboard_logs')
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        self.trade_date_last_train = None
        self.trade_date_acc_list = defaultdict(lambda: [0.0, 0.0])
        self.do_nothing_on_min_bar = False  # 仅供调试使用
        # 用于记录 open,high,low,close,amount 的 column 位置
        self.ohlcav_col_name_list = ["open", "high", "low", "close", "amount", "volume"]

    def get_x_y(self, factor_df):
        factors, trade_date_index = self.get_factor_array(factor_df)
        price_arr = factors[:, 0]
        self.input_size = factors.shape[1]
        # ys_all = self.calc_y_against_future_data(price_arr, -0.01, 0.01)
        # ys_all = self.calc_y_against_future_data(price_arr, -self.classify_wave_rate, self.classify_wave_rate)
        ys_all = calc_label3(price_arr, self.label_func_min_rr, self.label_func_max_rr,
                             max_future=self.max_future, one_hot=True)
        range_from = self.n_step
        range_to = factors.shape[0]

        xs = np.zeros((range_to - range_from, self.n_step, self.input_size, 1))
        for num, index in enumerate(range(range_from, range_to)):
            xs[num, :, :, :] = factors[(index - self.n_step):index, :, np.newaxis]

        ys = ys_all[range_from:range_to, :]

        return xs, ys, trade_date_index[range_from:range_to]

    def get_batch_xs(self, factors: np.ndarray, index=None):
        """
        取 batch_xs
        :param factors:
        :param index: 样本起始坐标，如果为None，则默认取尾部一组样本
        :return:
        """
        if index is None:
            index = factors.shape[0] - 1

        batch_xs = np.zeros((1, self.n_step, self.input_size, 1))
        batch_xs[0, :, :, :] = factors[(index - self.n_step + 1):(index + 1), :, np.newaxis]

        return batch_xs

    def get_batch_by_random(self, factors: np.ndarray, labels: np.ndarray):
        """
        够在一系列输入输出数据集
        xs： 两条同频率，不同位移的sin曲线
        ys_value： 目标是一条cos曲线
        ys: ys_value 未来涨跌标识
        i_s：X 序列
        """
        xs = np.zeros((self.batch_size, self.n_step, self.input_size))
        ys = np.zeros((self.batch_size, self.output_size))
        # available_batch_size, num = 0, 0
        samples_index = random.sample(range(self.n_step - 1, factors.shape[0] - 1), self.batch_size)
        examples_index_list = []
        for available_batch_size, index in enumerate(samples_index):
            tmp = factors[(index - self.n_step + 1):(index + 1), :]
            if tmp.shape[0] < self.n_step:
                break
            xs[available_batch_size, :, :] = tmp
            ys[available_batch_size, :] = labels[index, :]
            examples_index_list.append(index)
            if available_batch_size + 1 >= self.batch_size:
                available_batch_size += 1
                break

        # returned xs, ys_value and shape (batch, step, input)
        return xs, ys, examples_index_list

    def _build_model(self) -> tflearn.models.DNN:
        # Network building
        net = tflearn.input_data([None, self.n_step, self.input_size, 1])  # input_size=39
        net = tflearn.layers.normalization.batch_normalization(net)
        # layer1
        net = conv_2d(net, 32, 3, activation='relu', name='Conv2D_1')
        net = max_pool_2d(net, 3, strides=2)
        net = local_response_normalization(net)
        # layer2
        net = conv_2d(net, 64, 3, activation='relu', name='Conv2D_2')
        net = max_pool_2d(net, 3, strides=2)
        net = local_response_normalization(net)

        # layer3
        net = conv_2d(net, 128, 3, activation='relu', name='Conv2D_3')
        net = max_pool_2d(net, 3, strides=2)
        net = local_response_normalization(net)

        # layer4
        net = conv_2d(net, 256, 3, activation='relu', name='Conv2D_4')
        net = max_pool_2d(net, 3, strides=2)
        net = local_response_normalization(net)

        net = fully_connected(net, 1024, activation='tanh', name='fc1')
        net = dropout(net, 0.4, name='Dropout1')
        net = fully_connected(net, 1024, activation='tanh', name='fc2')
        net = dropout(net, 0.4, name='Dropout2')
        net = fully_connected(net, self.output_size, activation='softmax')

        net = tflearn.regression(net, optimizer='momentum',
                                 learning_rate=0.001,
                                 loss='categorical_crossentropy')

        # Training
        _model = tflearn.DNN(net, tensorboard_verbose=self.tensorboard_verbose,
                             # checkpoint_path=self.checkpoint_path,
                             tensorboard_dir=self.tensorboard_dir)
        return _model

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
            factor_df = self.load_train_test(indexed_df, rebuild_model=True,
                                             enable_load_model=self.enable_load_model_if_exist)
        else:
            factor_df = get_factor(indexed_df, ohlcav_col_name_list=self.ohlcav_col_name_list,
                                   trade_date_series=self.trade_date_series,
                                   delivery_date_series=self.delivery_date_series)

        # 预测
        pred_mark = self.predict_latest(factor_df)
        is_holding, is_buy, is_sell = pred_mark == 0, pred_mark == 1, pred_mark == 2
        # self.logger.info('%s is_buy=%s, is_sell=%s', trade_date, str(is_buy), str(is_sell))
        close = md_df['close'].iloc[-1]
        instrument_id = context[ContextKey.instrument_id_list][0]
        if is_buy:  # is_buy
            position_date_pos_info_dic = self.get_position(instrument_id)
            no_target_position = True
            if position_date_pos_info_dic is not None:
                for position_date, pos_info in position_date_pos_info_dic.items():
                    direction = pos_info.direction
                    if direction == Direction.Short:
                        self.close_short(instrument_id, close, pos_info.position)
                    elif direction == Direction.Long:
                        no_target_position = False
            if no_target_position:
                self.open_long(instrument_id, close, self.unit)
            else:
                self.logger.debug("%s %s     %.2f holding", self.trade_agent.curr_timestamp, instrument_id, close)

        if is_sell:  # is_sell
            position_date_pos_info_dic = self.get_position(instrument_id)
            no_holding_target_position = True
            if position_date_pos_info_dic is not None:
                for position_date, pos_info in position_date_pos_info_dic.items():
                    direction = pos_info.direction
                    if direction == Direction.Long:
                        self.close_long(instrument_id, close, pos_info.position)
                    elif direction == Direction.Short:
                        no_holding_target_position = False
            if no_holding_target_position:
                self.open_short(instrument_id, close, self.unit)
            else:
                self.logger.debug("%s %s     %.2f holding", self.trade_agent.curr_timestamp, instrument_id, close)

        if is_holding:
            self.logger.debug("%s %s * * %.2f holding", self.trade_agent.curr_timestamp, instrument_id, close)


def _test_use(is_plot):
    from ibats_common.backend.mess import get_folder_path
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
        'file_path': os.path.abspath(os.path.join(
            get_folder_path('example', create_if_not_found=False), 'data', f'{instrument_type}.csv')),
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
            'init_cash': 10000,
            "calc_mode": calc_mode,
        }
        strategy_handler_param = {
            'date_from': '2013-1-1',  # 策略回测历史数据，回测指定时间段的历史行情
            'date_to': '2018-10-18',
        }
    else:
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
        stg_class=AIStg,
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
        from ibats_common.analysis.summary import summary_stg_2_docx
        from ibats_utils.mess import open_file_with_system_app
        file_path = summary_stg_2_docx(stg_run_id, enable_clean_cache=False)
        if file_path is not None:
            open_file_with_system_app(file_path)

    return stg_run_id


if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG, format=config.LOG_FORMAT)
    is_plot = True
    _test_use(is_plot)
