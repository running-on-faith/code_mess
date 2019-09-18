#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2019-9-17 19:45
@File    : d3qn_reply_stg.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import multiprocessing
from collections import defaultdict

import ffn
from ibats_common.backend.factor import get_factor, transfer_2_batch
from ibats_common.backend.rl.emulator.account import Account
from ibats_common.common import BacktestTradeMode, ContextKey, CalcMode
from ibats_common.example import get_trade_date_series, get_delivery_date_series
from ibats_common.strategy import StgBase
from ibats_common.strategy_handler import strategy_handler_factory
from ibats_local_trader.agent.md_agent import *
from ibats_local_trader.agent.td_agent import *
from ibats_utils.mess import date_2_str, get_last
from collections import Counter
from drl.d3qn_replay_2019_08_25.agent.main import Agent, get_agent

logger = logging.getLogger(__name__)
logger.debug('import %s', ffn)


def predict_action(agent: Agent, inputs):
    return agent.choose_action_deterministic(inputs)


class DRL_LSTM_Stg(StgBase):

    def __init__(self, instrument_type, model_file_csv_path, unit=1):
        super().__init__()
        self.unit = unit
        self.instrument_type = instrument_type
        # 模型运行所需参数
        self.model_file_csv_path = model_file_csv_path  # 模型路径.csv 文件路径
        self.date_file_path_list_dic = defaultdict(list)
        self.model_date_list = []
        self.retrain_period = 60  # 60 每隔60天重新训练一次，0 则为不进行重新训练
        self._model_date_curr = None  # 当前预测使用的模型日期
        self.do_nothing_on_min_bar = False  # 仅供调试使用
        self._env = None
        self._agent_list = None  # 当前模型 Agent list
        self._pool = multiprocessing.Pool(multiprocessing.cpu_count())
        # 模型因子构建所需参数
        self.input_shape = [None, 12, 93, 5]
        self.action_size = 2  # close, long, short, keep 如果是3的话，则没有keep
        self.n_step = 60
        self.trade_date_series = get_trade_date_series()
        self.delivery_date_series = get_delivery_date_series(instrument_type)

    def load_model_list(self):
        df = pd.read_csv(self.model_file_csv_path)
        self.model_date_list = []
        for _, data_df in df.groupby('date'):
            self.date_file_path_list_dic[_] = list(data_df['file_path'])
            self.model_date_list.append(_)
        self.model_date_list.sort()

    def on_prepare_min1(self, md_df, context):
        # 加载模型列表
        self.load_model_list()

    @property
    def agent_param_iter(self):
        latest_state = self._env.latest_state()
        for agent in self._agent_list:
            yield agent, latest_state

    def predict_latest_action(self, indexed_df):
        """
        对最新状态进行预测
        :return: action: 0 long, 1, short, 0, close
        """
        # 获取最新交易日
        trade_date_latest = str_2_date(indexed_df.index[-1])
        # 获取因子
        factors_df = get_factor(
            indexed_df,
            trade_date_series=self.trade_date_series,
            delivery_date_series=self.delivery_date_series,
            dropna=True)
        df_index, df_columns, data_arr_batch = transfer_2_batch(factors_df, n_step=self.n_step)
        # 构建环境
        self._env = Account(indexed_df, data_factors=data_arr_batch, state_with_flag=True)
        model_date = get_last(self.model_date_list, lambda x: x < trade_date_latest)
        if model_date is None:
            raise ValueError(f'{date_2_str(trade_date_latest)} 以前没有有效的模型，最小的模型日期未 '
                             f'{date_2_str(min(self.model_date_list))}')
        if self._model_date_curr is None or self._model_date_curr != model_date:
            self._model_date_curr = model_date
            model_path_list = self.date_file_path_list_dic[model_date]
            # 构建 Agent list
            self._agent_list = []
            for model_path in model_path_list:
                agent = get_agent()
                agent.restore_model(model_path)
                self._agent_list.append(agent)

        # 多进程进行模型预测
        results = self._pool.map(lambda _agent, _inputs: _agent.choose_action_deterministic(_inputs),
                                 self.agent_param_iter)
        action_count_dic = Counter(results)
        action = action_count_dic.most_common(1)[0]
        self.logger.debug('%s action=%d, 各个 action 次数', date_2_str(trade_date_latest), action, action_count_dic)
        return action

    def on_min1(self, md_df, context):
        if self.do_nothing_on_min_bar:  # 仅供调试使用
            return

        # 数据整理
        indexed_df = md_df.set_index('trade_date').drop('instrument_type', axis=1)
        indexed_df.index = pd.DatetimeIndex(indexed_df.index)
        # 预测最新动作
        action = self.predict_latest_action(indexed_df)
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


def _test_use(is_plot):
    from drl import DATA_FOLDER_PATH
    import os
    instrument_type, backtest_date_from, backtest_date_to = 'RB', '2013-05-14', '2018-10-18'
    model_file_csv_path = ''
    # 参数设置
    run_mode = RunMode.Backtest_FixPercent
    calc_mode = CalcMode.Normal
    strategy_params = {'instrument_type': instrument_type, 'unit': 1}
    md_agent_params_list = [{
        'md_period': PeriodType.Min1,
        'instrument_id_list': [instrument_type],
        'datetime_key': 'trade_date',
        'init_md_date_from': '1995-1-1',  # 行情初始化加载历史数据，供策略分析预加载使用
        'init_md_date_to': backtest_date_from,
        'file_path': os.path.abspath(os.path.join(DATA_FOLDER_PATH, 'RB.csv')),
        'symbol_key': 'instrument_type',
    }]
    if run_mode == RunMode.Realtime:
        trade_agent_params = {
            "model_file_csv_path": model_file_csv_path,
        }
        strategy_handler_param = {
        }
    elif run_mode == RunMode.Backtest:
        trade_agent_params = {
            'trade_mode': BacktestTradeMode.Order_2_Deal,
            'init_cash': 1000000,
            "calc_mode": calc_mode,
            "model_file_csv_path": model_file_csv_path,
        }
        strategy_handler_param = {
            'date_from': backtest_date_from,  # 策略回测历史数据，回测指定时间段的历史行情
            'date_to': backtest_date_to,
        }
    else:
        # RunMode.Backtest_FixPercent
        trade_agent_params = {
            'trade_mode': BacktestTradeMode.Order_2_Deal,
            "calc_mode": calc_mode,
            "model_file_csv_path": model_file_csv_path,
        }
        strategy_handler_param = {
            'date_from': backtest_date_from,  # 策略回测历史数据，回测指定时间段的历史行情
            'date_to': backtest_date_to,
        }

    # 初始化策略处理器
    stghandler = strategy_handler_factory(
        stg_class=DRL_LSTM_Stg,
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
        file_path = summary_stg_2_docx(stg_run_id)
        if file_path is not None:
            open_file_with_system_app(file_path)

    return stg_run_id


if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG, format=config.LOG_FORMAT)
    _test_use(is_plot=True)
