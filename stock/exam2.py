import pandas as pd
import numpy as np
import ffn
import os
import matplotlib
import empyrical
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# DIR_PATH = r'C:\Users\26559\Downloads\弗居投资笔试 20200630\Data'
DIR_PATH = r'/home/mg/Downloads/Data'


def get_df(dir_path=r'/home/mg/Downloads/Data'):
    """获取数据"""
    file_path = os.path.join(DIR_PATH, "returns_20181228.csv")
    data_df = pd.read_csv(file_path, index_col=[0], parse_dates=[0])
    return data_df


def factor_adf(price_df):
    """因子ADF"""
    corr_dic = {}
    for n in [5]:
        factor_df = (price_df / price_df.shift(n) - 1).dropna()
        corr_s = pd.Series(np.diag(factor_df.T.corr(), 1), index=factor_df.index[1:])
        corr_dic[f"mom_{n}"] = corr_s
        corr_dic[f"mom_{n} Avg"] = corr_s.rolling(10).mean()

    corr_df = pd.DataFrame(corr_dic).dropna()
    """
    plot图可以看出,
    """
    corr_df.plot()


def factor_ic(price_df):
    """因子IC"""
    corr_dic = {}
    for n in [5, 10, 20]:
        factor_df = (price_df / price_df.shift(n) - 1).dropna()
        rr_df = (price_df.shift(-1) / price_df - 1).dropna()
        idxs = factor_df.index & rr_df.index
        factor_df = factor_df.loc[idxs, :]
        rr_df = rr_df.loc[idxs, :]
        corr_dic[f"mom_{n}"] = {idx: factor_df.loc[idx].corr(rr_df.loc[idx]) for idx in idxs}

    corr_df = pd.DataFrame(corr_dic).dropna()
    corr_df.plot(grid=True)
    plt.savefig("factor_ic.png")
    plt.close()


def mean_reversion(price_df):
    n = 5
    mean_df = price_df.rolling(n).mean()
    std_df = price_df.rolling(n).std()
    diff_df = ((price_df - mean_df) / std_df).dropna()
    mean_average_reversion = diff_df.std().mean()
    print("mean_average_reversion:", mean_average_reversion)
    return mean_average_reversion


def answer():
    data_df = get_df(DIR_PATH)
    price_df = (data_df / 100 + 1).cumprod()
    price_df.quantile([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95], axis=1).T.plot(title='Quantile Returns')
    plt.savefig('Quantile Returns.png')
    plt.close()
    # adf
    factor_adf(price_df)
    # 因子相关性
    factor_ic(price_df)
    # plt.close('all')
    mean_reversion(price_df)


if __name__ == '__main__':
    answer()
