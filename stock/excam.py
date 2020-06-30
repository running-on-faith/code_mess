import pandas as pd
import numpy as np
import os
from sklearn import linear_model, decomposition, ensemble, preprocessing, isotonic, metrics, svm


dir_path = r'C:\Users\26559\Downloads\弗居投资笔试 20200630\Data'
factor_file_path = os.path.join(dir_path, "FACTOR.csv")
rr_future_file_path = os.path.join(dir_path, "FMRTN1W.csv")

factor_df = pd.read_csv(factor_file_path, index_col=[0])
rr_future_df = pd.read_csv(rr_future_file_path, index_col=[0])


def get_xy(factor_df: pd.DataFrame, rr_future_df: pd.DataFrame):
    # factor_df.shape, rr_future_df.shape
    xs_list, ys_list = [], []
    for idx in range(factor_df.shape[0]):
        x = factor_df.iloc[idx:idx+1, :].T
        x_no_nan = x.dropna().index
        y = rr_future_df.iloc[idx, :].T
        y_no_nan = y.dropna().index
        no_nan = x_no_nan & y_no_nan
        if len(no_nan) == 0:
            continue
        xs_list.append(x.loc[no_nan].to_numpy())
        ys_list.append(y.loc[no_nan].to_numpy())

    xs = np.concatenate(xs_list)
    ys = np.concatenate(ys_list)
    return xs, ys


# 用AdaBoost训练
# Train classifier
scaler = preprocessing.MinMaxScaler()
clf = ensemble.AdaBoostClassifier(n_estimators=150) # n_estimators controls how many weak classifiers are fi

xs = scaler.transform(xs)
clf.fit(xs, ys)
clf.predict(xs) == ys
metrics.accuracy_score(clf.predict(xs), ys)

svr_model=svm.SVR()
svr_model.fit(xs, ys)
svr_model.predict(xs)
var = (svr_model.predict(xs) - ys).var()
metrics.accuracy_score(svr_model.predict(xs)>0, ys>0)