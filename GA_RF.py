import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from time import time


def GA_RF(ds, pop_size=10, selection_size=6, Pr=0.3, num_iter=20, max_count=4, runs=1):
    x_train, x_test, y_train, y_test = train_test_split(ds[:, :-1], ds[:, -1], test_size=0.3)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    n_features = x_train.shape[1]

    def acc(a):
        rf_tst_acc = np.zeros(a.shape[0])
        for i in range(a.shape[0]):
            rf = RandomForestClassifier(n_estimators=x[i, 0], max_depth=x[i, 1], max_features=x[i, 2])
            rf = rf.fit(x_train, y_train)
            # rf_tra_acc = rf.score(x_train, y_train)
            rf_tst_acc[i] = rf.score(x_test, y_test)
            print(i+1, 'out of', a.shape[0], 'done')
        return rf_tst_acc

    best_acc = 0
    for run in range(runs):
        counter = 0
        x1 = np.random.choice(range(10, 101, 10), size=(pop_size, 1), replace=True)
        x2 = np.random.poisson(lam=int(14 + n_features/20), size=(pop_size, 1))
        x3 = np.random.poisson(lam=int(np.sqrt(n_features)), size=(pop_size, 1))
        x3[x3 > n_features] = n_features
        x3[x3 == 0] = 1
        x = np.concatenate((x1, x2, x3), axis=1)
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
            rnd1 = np.random.choice(range(10, 101, 10), size=(selection_size, 1), replace=True)
            rnd2 = np.random.poisson(lam=int(14 + n_features/20), size=(selection_size, 1))
            rnd3 = np.random.poisson(lam=int(np.sqrt(n_features)), size=(selection_size, 1))
            rnd3[rnd3 > n_features] = n_features
            rnd3[rnd3 == 0] = 1
            randoms = np.concatenate((rnd1, rnd2, rnd3), axis=1)
            # we want mutation to happen with a probability, not always. so, we will choose between the old value and
            # the new one according to that probability.
            mask = np.random.choice([False, True], size=parents.shape, replace=True, p=[1-Pr, Pr])
            parents[mask] = randoms[mask]
            # cross over
            c0 = parents[:, 0].reshape(-1, 1)
            c1 = parents[:, 1].reshape(-1, 1)
            c2 = parents[:, 2].reshape(-1, 1)
            np.random.shuffle(c0)
            np.random.shuffle(c1)
            np.random.shuffle(c2)
            children = np.concatenate((c0, c1, c2), axis=1)
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

    print('best:', best_acc, ', for n_estimators=%d, max_depth=%d, & max_features=%d' % (best[0], best[1], best[2]), ', after', itr+1, 'iterations')


# df = pd.read_csv('spambase.dat', sep=',', header=None)
# # df_tra = df.reindex(np.random.permutation(df.index))
# ds = df.values
# # ds = ds.astype('float32')
# scaler = MinMaxScaler(feature_range=(0, 1))
# ds[:, :-1] = scaler.fit_transform(ds[:, :-1])
# # print(ds.shape)
# # print(np.var(ds[:, :-1]))

# t0 = time()
# GA_RF(ds)
# t1 = time()
# print(t1-t0, '(s)')
