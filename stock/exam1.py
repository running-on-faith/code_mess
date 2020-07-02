import pandas as pd
import numpy as np
import ffn
import os
import matplotlib
import empyrical
# matplotlib.use('agg')
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# DIR_PATH = r'C:\Users\26559\Downloads\弗居投资笔试 20200630\Data'
DIR_PATH = r'/home/mg/Downloads/Data'


def get_df(dir_path=r'/home/mg/Downloads/Data'):
    """获取数据"""
    factor_file_path = os.path.join(dir_path, "FACTOR.csv")
    rr_future_file_path = os.path.join(dir_path, "FMRTN1W.csv")
    factor_df = pd.read_csv(factor_file_path, index_col=[0])
    factor_df.columns = pd.to_datetime(factor_df.columns)
    rr_future_df = pd.read_csv(rr_future_file_path, index_col=[0])
    rr_future_df.columns = pd.to_datetime(rr_future_df.columns)
    return factor_df, rr_future_df


def backtest(factor_df, rr_future_df):
    separate_count = 5
    # As the order said:
    # You have 1 dollar at the start. You want to invest equally, thus initially it is 0.01 dollar for each stock.
    # So we can only hold 50 stocks max for each direction
    max_position_each_direction = 50
    # 每一只股票固定占比 1%
    fix_position = 0.01
    portfolio_rr = defaultdict(dict)
    # 按日回测
    long_each_period_dic, short_each_period_dic, rr_dic = OrderedDict(), OrderedDict(), OrderedDict()
    stocks_fractile_dic = defaultdict(list)
    for idx in range(factor_df.shape[1]):
        x = factor_df.iloc[:, idx:idx + 1]
        x_no_nan = x.dropna().index
        y = rr_future_df.iloc[:, idx]
        y_no_nan = y.dropna().index
        no_nan = x_no_nan & y_no_nan
        if len(no_nan) == 0:
            continue
        # 取有效数据
        xs = x.loc[no_nan]
        ys = y.loc[no_nan]
        ts = pd.to_datetime(ys.name)
        # rank 逆排序
        rank = xs.rank(ascending=False).iloc[:, 0]
        count_per_rank = len(rank) // separate_count
        long_idxs = rank <= min(max_position_each_direction, count_per_rank)
        long_each_period_dic[ts] = xs[long_idxs].dropna()
        short_idxs = (count_per_rank * (separate_count - 1) < rank) & \
                     (rank <= count_per_rank * (separate_count - 1) + min(max_position_each_direction, count_per_rank))
        short_each_period_dic[ts] = xs[short_idxs].dropna()
        rr_dic[ts] = ys[long_idxs].sum() * fix_position - ys[short_idxs].sum() * fix_position
        # 收集每个 20% quantile 的股票列表，以及 portfolio rr
        for _ in range(separate_count):
            quantile_idxs = (count_per_rank * _ < rank) & (rank <= count_per_rank * (_ + 1))
            stocks_fractile_dic[ts].append(set(xs[quantile_idxs].dropna().index))
            portfolio_rr[f'antile{_ + 1}'][ts] = ys[quantile_idxs].dropna().mean()

    portfolio_rr["S"] = rr_dic

    return long_each_period_dic, short_each_period_dic, portfolio_rr, stocks_fractile_dic


def plot_cum_rr(rr_dic):
    rr_df = pd.DataFrame([rr_dic], index=['stg1_rr']).T
    rr_df['Wealth Curve'] = (rr_df['stg1_rr'] + 1).cumprod()
    rr_df[['Wealth Curve']].plot(grid=True, title="Wealth Curve")
    plt.savefig('Wealth Curve(Portfolio Cum RR).png')
    plt.close()


def plot_coverage_quantile(long_each_period_dic: dict, short_each_period_dic: dict, stocks_fractile_dic: dict):
    coverage_dic = {
        key: (len(long_each_period_dic[key].index) + len(short_each_period_dic[key].index)) /
             (len(stocks_fractile_dic[key][0]) + len(stocks_fractile_dic[key][0]))
        for key in long_each_period_dic.keys()
    }
    coverage_df = pd.DataFrame({"Coverage of Quantile": coverage_dic})
    coverage_df.index = pd.to_datetime(coverage_df.index)
    coverage_df.sort_index(inplace=True)
    coverage_df.plot(kind='bar')
    plt.savefig('Coverage of Quantile.png')
    plt.close()


def plot_bar_year_avg(coverage_df, bar_title, file_name, avg_title='12-month moving average'):
    ax = coverage_df.rolling('365D').mean().rename(columns={bar_title: avg_title}).plot(color='r')
    ax2 = plt.twiny(ax)
    coverage_df.plot(y=bar_title, kind='bar', ax=ax2, title=bar_title)
    # plot 图标显示存在问题，稍后解决
    plt.legend(["b", "g"], labels=[bar_title, avg_title], loc=0)
    plt.savefig(file_name)
    plt.close()


def plot_coverage(factor_df):
    coverage_df = pd.DataFrame({
        'Coverage': {pd.to_datetime(key): x.dropna().shape[0] for key, x in factor_df.items()}})
    plot_bar_year_avg(coverage_df, 'Coverage', 'Coverage.png')


def plot_ann(portfolio_rr):
    rr_df = pd.DataFrame(portfolio_rr).sort_index()
    cum_rr_df = (rr_df + 1).cumprod()
    portfolio_stat_dic = {_: cum_rr_df[_].calc_perf_stats() for _ in cum_rr_df.columns}
    # Annualized Returns
    cagr_df = pd.DataFrame([{key: _.cagr for key, _ in portfolio_stat_dic.items()}],
                           index=['Annualized Returns']).T
    cagr_df.plot(kind='bar', title='Annualized Returns')
    plt.savefig('Annualized Returns.png')
    plt.close()
    # Annualized Volatility
    vol_df = pd.DataFrame([{key: _.yearly_vol for key, _ in portfolio_stat_dic.items()}],
                          index=["Annualized Volatility"]).T
    vol_df.plot(kind='bar', title='Annualized Volatility')
    plt.savefig('Annualized Volatility.png')
    plt.close()
    # Portfolio IR (Information Ratio)
    vol_df = pd.DataFrame([{key: _.calc_sharpe(0) for key, _ in cum_rr_df.items()}],
                          index=["sharpe ratio"]).T
    vol_df.plot(kind='bar', title='sharpe ratio')
    plt.savefig('Sharpe Ratio（没有用指数进行对标，因此没有求信息比率，用夏普比率替代一下）')
    plt.close()
    # Portfolio Returns
    portfolio_return = cum_rr_df.iloc[-1, :].rename("Portfolio Returns")
    portfolio_return.plot(kind='bar', title='Portfolio Returns')
    plt.savefig('Portfolio Returns.png')
    # Quantile Returns
    cum_rr_df.resample('1M').last().plot(title='Quantile Returns')
    plt.savefig('Quantile Returns.png')
    plt.close()
    # Sortino Ratio
    ratio_df = pd.DataFrame(empyrical.sortino_ratio(rr_df), columns=["Sortino Ratio"])
    ratio_df.index = rr_df.columns
    ratio_df.plot(kind='bar', title='Sortino Ratio')
    plt.savefig("Sortino Ratio.png")
    plt.close()


def factor_turnover(long_each_period_dic, short_each_period_dic):
    last_long, last_short = None, None
    turnover_dic = {}
    for key in long_each_period_dic.keys():
        long_pos = set(long_each_period_dic[key].index)
        short_pos = set(short_each_period_dic[key].index)
        if last_long is not None and last_short is not None:
            turnover_long = (len(long_pos) - len(last_long & long_pos)) / len(long_pos)
            turnover_short = (len(short_pos) - len(last_short & short_pos)) / len(short_pos)
            turnover = turnover_long + turnover_short
            turnover_dic[key] = turnover

        # 更新上一个状态
        last_long, last_short = long_pos, short_pos

    turnover_df = pd.DataFrame([turnover_dic], index=['Factor turnover']).T
    turnover_df.index = pd.to_datetime(turnover_df.index)
    turnover_df.sort_index(inplace=True)
    plot_bar_year_avg(turnover_df, 'Factor turnover', 'Factor turnover.png')


def info_coefficient(factor_df: pd.DataFrame, rr_future_df: pd.DataFrame):
    factor_m_df = factor_df.T.resample('1M').last().T
    rr_future_m_df = rr_future_df.T.resample('1M').last().T
    info_coefficient_dic = {}
    for key in factor_m_df.columns:
        factor_s = factor_m_df[key]
        if key not in rr_future_m_df:
            continue
        rr_future_s = rr_future_m_df[key]
        no_nan = factor_s.dropna().index & rr_future_s.dropna().index
        factor_s = factor_s[no_nan]
        rr_future_s = rr_future_s[no_nan]
        factor_rank = factor_s.rank()
        rr_future_rank = rr_future_s.rank()
        rank_df = pd.DataFrame({"factor": factor_rank, "rr_future": rr_future_rank})
        info_coefficient_dic[key]=rank_df.corr(method="pearson").iloc[1,0]

    info_coefficient_df = pd.DataFrame([info_coefficient_dic],
                                       index=['Spearson rank IC (information coefficient)']).T
    info_coefficient_df.plot(title='Spearson rank IC')
    plt.savefig('Spearson rank IC.png')
    plt.close()


def time_series_spread(portfolio_rr):
    portfolio_rr_df = pd.DataFrame({_: portfolio_rr[_] for _ in ['antile1', 'antile5']})
    spread_s = portfolio_rr_df['antile1'] - portfolio_rr_df['antile5']
    spread_s.name = 'Time Series Spread'
    pd.DataFrame(spread_s).plot(title='Time Series Spread')
    plt.savefig('Time Series Spread.png')
    plt.close()


def analyse(factor_df, rr_future_df, long_each_period_dic, short_each_period_dic, portfolio_rr, stocks_fractile_dic):
    rr_dic = portfolio_rr['S']

    # 计算 Coverage
    # Coverage: This shows the number of stocks covered by the strategy (Factor data) in all quintile (meaning
    # 20% fractile) portfolios, i.e. the number of stocks that have quant factors in each period. 12-month moving
    # average is usually drawn to show smooth information.
    plot_coverage(factor_df)

    # Annualized Returns, Annualized Volatility, Portfolio IR (Information Ratio), Portfolio Returns,
    # Quantile Returns, Sortino Ratio
    plot_ann(portfolio_rr)

    # Factor turnover
    factor_turnover(long_each_period_dic, short_each_period_dic)

    # Serial Correlation
    # plot_serial_corr(factor_df)

    # Spearson rank IC (information coefficient)
    info_coefficient(factor_df, rr_future_df)
    # Time Series Spread
    time_series_spread(portfolio_rr)
    # Wealth Curve
    plot_cum_rr(rr_dic)


def backtest_and_analyse():
    # 获取数据
    factor_df, rr_future_df = get_df(DIR_PATH)
    # 回测
    _ = backtest(factor_df, rr_future_df)
    # 分析
    analyse(factor_df, rr_future_df, *_)


if __name__ == '__main__':
    backtest_and_analyse()
