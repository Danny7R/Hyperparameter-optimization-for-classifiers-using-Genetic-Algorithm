import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from time import time


def GS_SVM(ds, gammas=np.arange(1, 10, 0.1)):
    x_train, x_test, y_train, y_test = train_test_split(ds[:, :-1], ds[:, -1], test_size=0.3)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    best_tst = 0
    kernels = ['rbf', 'sigmoid']  # , 'linear', 'poly'
    total = len(kernels) * len(gammas)
    i = 0
    for kernel in kernels:
        for gamma in gammas:  # gamma = 1/(n_features * np.var(X))
            svm = SVC(kernel=kernel, gamma=gamma)
            svm.fit(x_train, y_train)
            svm_tra_acc = svm.score(x_train, y_train)
            svm_tst_acc = svm.score(x_test, y_test)
            if (svm_tst_acc > best_tst):
                best_tst = svm_tst_acc
                best_tra = svm_tra_acc
                best_kernel = kernel
                best_gamma = gamma
            i += 1
            print(i, 'out of', total, 'done')

    print('best: %f (%f)' % (best_tst, best_tra), ', for kernel=', best_kernel, '& gamma=', best_gamma)


# df = pd.read_csv('spambase.dat', sep=',', header=None)
# # df_tra = df.reindex(np.random.permutation(df.index))
# ds = df.values
# ds = ds.astype('float32')
# scaler = MinMaxScaler(feature_range=(0, 1))
# ds[:, :-1] = scaler.fit_transform(ds[:, :-1])
# print(ds.shape)
# # print(np.var(ds[:, :-1]))

# t0 = time()
# GS_SVM(ds)
# t1 = time()
# print(t1-t0, '(s)')
