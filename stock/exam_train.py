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


def get_xy(factor_df: pd.DataFrame, rr_future_df: pd.DataFrame):
    """生成 训练数据 xs, 训练标签 ys"""
    # factor_df.shape, rr_future_df.shape
    xs_list, ys_list = [], []
    separate_count = 5
    # 按每天收益率数值 quantile 进行标记
    for idx in range(factor_df.shape[1]):
        x = factor_df.iloc[idx:idx + 1, :].T
        x_no_nan = x.dropna().index
        y = rr_future_df.iloc[idx, :].T
        y_no_nan = y.dropna().index
        no_nan = x_no_nan & y_no_nan
        if len(no_nan) == 0:
            continue
        # 取有效数据
        xs = x.loc[no_nan]
        ys = y.loc[no_nan]
        # 对 y 进行标记
        ys_rank = ys.rank()
        count_per_rank = ys.shape[0] // separate_count
        for n in range(separate_count):
            ys[((count_per_rank * n) < ys_rank) & (ys_rank <= (count_per_rank * (n + 1)))] = n

        # 加入列表
        xs_list.append(xs)
        ys_list.append(ys)

    # 合并列表,形成统一的x,y序列
    xs = np.concatenate(xs_list)
    ys = np.concatenate(ys_list).astype(int)
    return xs, ys


def train_xy(xs, ys):
    """训练并返回 分类器 clf 以及 正则化类"""
    mm_scaler = preprocessing.MinMaxScaler()
    xs = mm_scaler.fit_transform(xs)

    # 用AdaBoost训练
    # Train classifier
    clf = ensemble.AdaBoostClassifier(n_estimators=150)  # n_estimators controls how many weak classifiers are fi
    clf.fit(xs, ys)
    print(f"accuracy_score: {metrics.accuracy_score(clf.predict(xs), ys)*100:.2f}%")
    return clf, mm_scaler


def backtest(factor_df: pd.DataFrame, rr_future_df: pd.DataFrame, clf, scaler):
    for idx in range(factor_df.shape[1]):
        x = factor_df.iloc[idx:idx + 1, :].T
        x_no_nan = x.dropna().index
        y = rr_future_df.iloc[idx, :].T
        y_no_nan = y.dropna().index
        no_nan = x_no_nan & y_no_nan
        if len(no_nan) == 0:
            continue
        # 取有效数据
        xs = x.loc[no_nan]
        ys = y.loc[no_nan]
        ys_pred = clf.predict(xs)




def train_and_backtest():
    factor_df, rr_future_df = get_df(DIR_PATH)
    classifier, scaler = train_xy(*get_xy(factor_df, rr_future_df))
    backtest(factor_df, rr_future_df, classifier, scaler)


if __name__ == '__main__':
    train_and_backtest()
