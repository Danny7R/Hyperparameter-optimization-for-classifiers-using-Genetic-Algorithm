import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from time import time


def GA_SVM(ds, pop_size=10, selection_size=6, Pr=0.3, num_iter=20, max_count=4, runs=1):
    n_features = ds.shape[1] - 1
    var = np.var(ds[:, :-1])
    x_train, x_test, y_train, y_test = train_test_split(ds[:, :-1], ds[:, -1], test_size=0.3)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    kernels = ['rbf', 'sigmoid']  # , 'linear', 'poly']

    def acc(a):
        svm_tst_acc = np.zeros(a.shape[0])
        for i in range(a.shape[0]):
            svm = SVC(kernel=kernels[int(a[i, 0])], gamma=a[i, 1])
            svm.fit(x_train, y_train)
            # svm_tra_acc = svm.score(x_train, y_train)
            svm_tst_acc[i] = svm.score(x_test, y_test)
            print(i+1, 'out of', a.shape[0], 'done')
        return svm_tst_acc

    best_acc = 0
    for run in range(runs):
        counter = 0
        x1 = np.random.choice(len(kernels), size=(pop_size, 1), replace=True)  # , p=[0.6, 0.3, 0.07, 0.03])
        x2 = np.random.gamma(shape=2/(n_features*var) + 1, scale=0.5, size=(pop_size, 1))  # .reshape((pop_size, 2))
        x = np.concatenate((x1, x2), axis=1)
        x_acc = acc(x)
        sorted_indices = np.argsort(x_acc)
        x = x[sorted_indices]
        x_acc = x_acc[sorted_indices]

        for itr in range(num_iter):
            fitness = x_acc - np.min(x_acc) + 0.01  #
            fitness = fitness / np.sum(fitness)  # so that every value is between 0 and 1 and the sum of all is 1
            # selection
            selected_indices = np.random.choice(pop_size, size=selection_size, replace=True, p=fitness)
            # mutation
            parents = np.copy(x[selected_indices])
            rnd1 = np.random.choice(len(kernels), size=(selection_size, 1), replace=True)  # , p=[0.6, 0.3, 0.07, 0.03])
            rnd2 = np.random.gamma(shape=1/(n_features*var) + 1, scale=1, size=(selection_size, 1))  # .reshape((pop_size, 2))
            randoms = np.concatenate((rnd1, rnd2), axis=1)
            # we want mutation to happen with a probability, not always. so, we will choose between the old value and
            # the new one according to that probability.
            mask = np.random.choice([False, True], size=parents.shape, replace=True, p=[1-Pr, Pr])
            parents[mask] = randoms[mask]
            # cross over
            np.random.shuffle(parents)
            children = np.zeros_like(parents)
            children[:int(parents.shape[0]/2)] = np.append(parents[:int(parents.shape[0]/2), :int(parents.shape[1]/2)],
                                                           parents[int(parents.shape[0]/2):, int(parents.shape[1]/2):], axis=1)
            children[int(parents.shape[0]/2):] = np.append(parents[int(parents.shape[0]/2):, :int(parents.shape[1]/2)],
                                                           parents[:int(parents.shape[0]/2), int(parents.shape[1]/2):], axis=1)
            x[:children.shape[0]] = children
            children_acc = acc(children)
            x_acc[:len(children_acc)] = children_acc
            sorted_indices = np.argsort(x_acc)
            x = x[sorted_indices]
            x_acc = x_acc[sorted_indices]

            counter += 1
            if(x_acc[-1] > best_acc):
                best = x[-1]
                best_acc = x_acc[-1]
                counter = 0

            if(counter >= max_count):
                break

    print('best:', best_acc, ', for kernel=', kernels[int(best[0])], '& gamma=', best[1], ', after', itr+1, 'iterations')


# df = pd.read_csv('spambase.dat', sep=',', header=None)
# # df_tra = df.reindex(np.random.permutation(df.index))
# ds = df.values
# ds = ds.astype('float32')
# scaler = MinMaxScaler(feature_range=(0, 1))
# ds[:, :-1] = scaler.fit_transform(ds[:, :-1])
# print(ds.shape)
# # x_train, x_test, y_train, y_test = train_test_split(ds[:, :-1], ds[:, -1], test_size=0.3)
# # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# # svm = SVC(kernel='rbf', gamma='auto')
# # svm.fit(x_train, y_train)
# # svm_tra_acc = svm.score(x_train, y_train)
# # svm_tst_acc = svm.score(x_test, y_test)
# # print(svm_tra_acc, svm_tst_acc)

# t0 = time()
# GA_SVM(ds)
# t1 = time()
# print(t1-t0, '(s)')
