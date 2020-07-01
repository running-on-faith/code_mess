import pandas as pd
import numpy as np
import ffn
import os
import matplotlib
import empyrical
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

DIR_PATH = r'C:\Users\26559\Downloads\弗居投资笔试 20200630\Data'


# DIR_PATH = r'/home/mg/Downloads/Data'


def get_df(dir_path=r'/home/mg/Downloads/Data'):
    """获取数据"""
    file_path = os.path.join(DIR_PATH, "returns_20181228.csv")
    data_df = pd.read_csv(file_path, index_col=[0], parse_dates=[0])
    return data_df


if __name__ == '__main__':
    data_df = get_df(DIR_PATH)
    price_df = (data_df/100 + 1).cumprod()
    factor_df = (price_df - price_df.shift(10)).dropna()
    price_df.quantile([0.1, 0.25, 0.5, 0.75, 0.9], axis=1).T.plot(title='Quantile Returns')
    plt.savefig('Quantile Returns.png')
    plt.close()
    factor = (data_df.rolling(10).mean() - data_df).dropna()
