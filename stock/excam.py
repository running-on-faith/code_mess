import pandas as pd
import numpy as np
import os
from sklearn import linear_model, decomposition, ensemble, preprocessing, isotonic, metrics, svm

# DIR_PATH = r'C:\Users\26559\Downloads\弗居投资笔试 20200630\Data'
DIR_PATH = r'/home/mg/Downloads/Data'


def get_df(dir_path=r'/home/mg/Downloads/Data'):
    """获取数据"""
    factor_file_path = os.path.join(dir_path, "FACTOR.csv")
    rr_future_file_path = os.path.join(dir_path, "FMRTN1W.csv")
    factor_df = pd.read_csv(factor_file_path, index_col=[0])
    rr_future_df = pd.read_csv(rr_future_file_path, index_col=[0])
    return factor_df, rr_future_df


def backtest(factor_df: pd.DataFrame, rr_future_df):
    long_each_week_dic, short_each_week_dic = {}, {}
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
        rank = xs.rank()
        count_per_rank = len(rank) // 5
        break
        long_each_week_dic[ys.name] = xs[rank <= count_per_rank].dropna()
        ys[rank <= count_per_rank].dropna()



if __name__ == '__main__':
    backtest()
