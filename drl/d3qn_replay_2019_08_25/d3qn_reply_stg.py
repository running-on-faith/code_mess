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
from collections import Counter
from collections import defaultdict

import ffn
from config import config
from ibats_common.backend.factor import get_factor, transfer_2_batch
from ibats_common.backend.rl.emulator.account import Account
from ibats_common.common import BacktestTradeMode, ContextKey, CalcMode
from ibats_common.example import get_trade_date_series, get_delivery_date_series
from ibats_common.example.data import OHLCAV_COL_NAME_LIST
from ibats_common.strategy import StgBase
from ibats_common.strategy_handler import strategy_handler_factory
from ibats_local_trader.agent.md_agent import *
from ibats_local_trader.agent.td_agent import *
from ibats_utils.mess import date_2_str, get_last

from drl import DATA_FOLDER_PATH
from drl.d3qn_replay_2019_08_25.agent.main import get_agent

logger = logging.getLogger(__name__)
logger.debug('import %s', ffn)


def predict(model_path, inputs, agent):
    """加载模型，加载权重，预测action"""
    agent.restore_model(model_path)
    # logger.info('agent predict state.shape=%s, action=%s', inputs[0].shape, inputs[1])
    # print('agent predict state.shape=%s, action=%s' % (inputs[0].shape, inputs[1]))
    # return 1
    return agent.choose_action_deterministic(inputs)


class DRLStg(StgBase):

    def __init__(self, instrument_type, model_file_csv_path, unit=1, pool_worker_num=multiprocessing.cpu_count()):
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
        # 供进程池使用
        self.pool_worker_num = pool_worker_num
        self._pool, self.task_kwargs_queue, self.task_result_queue = None, None, None
        # 供串行执行使用
        self.agent = None
        # 模型因子构建所需参数
        self.n_step = 60
        self.trade_date_series = get_trade_date_series(DATA_FOLDER_PATH)
        self.delivery_date_series = get_delivery_date_series(instrument_type, DATA_FOLDER_PATH)
        self._last_action = 0

    def load_model_list(self):
        df = pd.read_csv(self.model_file_csv_path, parse_dates=['date'])
        self.model_date_list, self.date_file_path_list_dic = [], {}
        for _, data_df in df.groupby('date'):
            self.date_file_path_list_dic[_] = list(data_df['file_path'])
            self.model_date_list.append(_)
        self.model_date_list.sort()

    def on_prepare_min1(self, md_df, context):
        # 加载模型列表
        self.load_model_list()
        if self.pool_worker_num is not None and self.pool_worker_num > 0:
            # 建立进程池
            self._pool = multiprocessing.Pool(self.pool_worker_num)
            self.task_kwargs_queue, self.task_result_queue = multiprocessing.JoinableQueue(), multiprocessing.Queue()


        def predict_worker(queues):
            """加载模型，加载权重，预测action"""
            import queue
            task_kwargs_queue, task_result_queue = queues
            agent = None
            while True:
                try:
                    # model_path, inputs, agent=None, input_shape=None
                    model_path, inputs, input_shape = task_kwargs_queue.get(True, 5)
                except queue.Empty:
                    continue
                try:
                    if agent is None:
                        agent = get_agent(input_shape=input_shape)
                        logger.debug('建立模型 input_shape=%s', input_shape)

                    # 加载模型参数
                    agent.restore_model(model_path)
                    result = agent.choose_action_deterministic(inputs)
                    task_result_queue.put(result)
                except:
                    logger.exception("%s, input_shape=%s 模型执行异常", model_path, input_shape)
                    task_result_queue.put(None)
                finally:
                    task_kwargs_queue.task_done()

        self._pool.imap(predict_worker, [(self.task_kwargs_queue, self.task_result_queue)
                                         for _ in range(self.pool_worker_num)])

    def predict_param_iter(self, model_path_list, shape):
        latest_state = self._env.latest_state()
        for model_path in model_path_list:
            yield model_path, latest_state, None, shape

    def predict_latest_action(self, indexed_df):
        """
        对最新状态进行预测
        :return: action: 0 long, 1, short, 0, close
        """
        # 获取最新交易日
        trade_date_latest = pd.to_datetime(indexed_df.index[-1])
        # 获取因子
        # 2019-09-19 当期训练模型没有 trade_date_series， delivery_date_series 两种因子，因此预测时需要注释掉
        factors_df = get_factor(
            indexed_df,
            # trade_date_series=self.trade_date_series,
            # delivery_date_series=self.delivery_date_series,
            dropna=True)
        df_index, df_columns, batch_factors = transfer_2_batch(factors_df, n_step=self.n_step)
        # 构建环境
        self._env = Account(indexed_df, data_factors=batch_factors, state_with_flag=True)
        # 设置最新的 action
        self._env.step(self._last_action)
        model_date = get_last(self.model_date_list, lambda x: x < trade_date_latest)
        if model_date is None:
            raise ValueError(f'{date_2_str(trade_date_latest)} 以前没有有效的模型，最小的模型日期未 '
                             f'{date_2_str(min(self.model_date_list))}')
        if self._model_date_curr is None or self._model_date_curr != model_date:
            self._model_date_curr = model_date
        model_path_list = self.date_file_path_list_dic[model_date]

        # 多进程进行模型预测
        if self._pool is None:
            # 串行执行
            latest_state = self._env.latest_state()
            if self.agent is None:
                self.agent = get_agent(input_shape=batch_factors.shape)
            results = [predict(model_path, latest_state, self.agent)
                       for model_path in model_path_list]
        else:
            # 多进程执行
            task_count, finished_count, results = 0, 0, []
            for _ in self.predict_param_iter(model_path_list, batch_factors.shape):
                self.task_kwargs_queue.put(_)
                task_count += 1
            while True:
                result = self.task_result_queue.get()
                if result is not None:
                    results.append(result)
                finished_count += 1
                if finished_count == task_count:
                    break

        action_count_dic = Counter(results)
        action = action_count_dic.most_common(1)[0][0]
        self.logger.debug('%s action=%d, 各个 action 次数 %s', date_2_str(trade_date_latest), action, action_count_dic)
        return action

    def on_min1(self, md_df, context):
        if self.do_nothing_on_min_bar:  # 仅供调试使用
            return

        # 数据整理
        indexed_df = md_df.set_index('trade_date')[OHLCAV_COL_NAME_LIST]
        indexed_df.index = pd.DatetimeIndex(indexed_df.index)
        # 预测最新动作
        action = self.predict_latest_action(indexed_df)
        close = md_df['close'].iloc[-1]
        instrument_id = context[ContextKey.instrument_id_list][0]
        if action == 1:  # is_buy
            self.keep_long(instrument_id, close, self.unit)

        elif action == 2:  # is_sell
            self.keep_short(instrument_id, close, self.unit)

        elif action == 0:
            self.keep_empty(instrument_id, close)

        self._last_action = action


def _test_use(is_plot):
    from drl import DATA_FOLDER_PATH
    import os
    instrument_type, backtest_date_from, backtest_date_to = 'RB', '2013-05-14', '2018-10-18'
    from ibats_utils.mess import is_windows_os
    if is_windows_os():
        model_file_csv_path = r'D:\WSPych\code_mess\drl\drl_off_example\d3qn_replay_2019_08_25\output\available_model_path.csv'
    else:
        model_file_csv_path = r'/home/mg/github/code_mess/drl/drl_off_example/d3qn_replay_2019_08_25/output/available_model_path.csv'
    # 参数设置
    run_mode = RunMode.Backtest_FixPercent
    calc_mode = CalcMode.Normal
    strategy_params = {'instrument_type': instrument_type,
                       'unit': 1,
                       "model_file_csv_path": model_file_csv_path,
                       "pool_worker_num": 2,
                       }
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
            'date_from': backtest_date_from,  # 策略回测历史数据，回测指定时间段的历史行情
            'date_to': backtest_date_to,
        }
    else:
        # RunMode.Backtest_FixPercent
        trade_agent_params = {
            'trade_mode': BacktestTradeMode.Order_2_Deal,
            "calc_mode": calc_mode,
        }
        strategy_handler_param = {
            'date_from': backtest_date_from,  # 策略回测历史数据，回测指定时间段的历史行情
            'date_to': backtest_date_to,
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
        from ibats_common.analysis.summary import summary_stg_2_docx
        from ibats_utils.mess import open_file_with_system_app
        file_path = summary_stg_2_docx(stg_run_id)
        if file_path is not None:
            open_file_with_system_app(file_path)

    return stg_run_id


if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG, format=config.LOG_FORMAT)
    _test_use(is_plot=True)
