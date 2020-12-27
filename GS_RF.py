import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from time import time


def GS_RF(ds, max_ds=np.arange(15, 20), max_fs=np.arange(3, 7)):
    x_train, x_test, y_train, y_test = train_test_split(ds[:, :-1], ds[:, -1], test_size=0.3)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    n_features = x_train.shape[1]

    best_tst = 0
    total = 10 * len(max_ds) * len(max_fs)
    i = 0
    for n_est in range(10, 101, 10):
        for max_d in max_ds:
            for max_f in max_fs:
                rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, max_features=max_f)  # , min_samples_split=2,
                rf = rf.fit(x_train, y_train)
                # rf_tra_acc = rf.score(x_train, y_train)
                rf_tst_acc = rf.score(x_test, y_test)
                if (rf_tst_acc > best_tst):
                    best_tst = rf_tst_acc
                    best_tra = rf.score(x_train, y_train)
                    best_n_est = n_est
                    best_max_d = max_d
                    best_max_f = max_f
                i += 1
                print(i, 'out of', total, 'done')

    print('best: %f (%f)' % (best_tst, best_tra), ', for n_estimators=%d, max_depth=%d, & max_features=%d' % (best_n_est, best_max_d, best_max_f))


# df = pd.read_csv('wine.dat', sep=',', header=None)
# # df_tra = df.reindex(np.random.permutation(df.index))
# ds = df.values
# # ds = ds.astype('float32')
# scaler = MinMaxScaler(feature_range=(0, 1))
# ds[:, :-1] = scaler.fit_transform(ds[:, :-1])
# # print(ds.shape)
# # print(np.var(ds[:, :-1]))

# t0 = time()
# GS_RF(ds, max_ds=np.arange(15, 20), max_fs=np.arange(3, 7))
# t1 = time()
# print(t1-t0, '(s)')
