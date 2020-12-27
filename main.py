import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from time import time
from GA_SVM import GA_SVM
from GS_SVM import GS_SVM
from GA_RF import GA_RF
from GS_RF import GS_RF


## choose among these 5 datasets:

# df = pd.read_csv('Data/spambase.dat', sep=',', header=None)
# df = pd.read_csv('Data/banana.dat', sep=',', header=None)
# df = pd.read_csv('Data/wine.dat', sep=',', header=None)
# df = pd.read_csv('Data/ionosphere.dat', sep=',', header=None)
df = pd.read_csv('Data/iris.dat', sep=',', header=None)

## preparing the data

df = df.reindex(np.random.permutation(df.index))
ds = df.values
# ds = ds.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
ds[:, :-1] = scaler.fit_transform(ds[:, :-1])
print(ds.shape)
# var = np.var(ds[:, :-1])
# print(var)

## the SVM classifier
t0 = time()
GA_SVM(ds, pop_size=10, selection_size=6, Pr=0.3, num_iter=20, max_count=4, runs=1)  # Genetic Algorithm
t1 = time()
print(t1-t0, '(s)')

t0 = time()
GS_SVM(ds, gammas=np.arange(1, 9, 0.4))  # Grid Search
t1 = time()
print(t1-t0, '(s)')

## the Random Forest classifier
t0 = time()
GA_RF(ds, pop_size=10, selection_size=6, Pr=0.3, num_iter=20, max_count=4, runs=1)  # Genetic Algorithm
t1 = time()
print(t1-t0, '(s)')

t0 = time()
GS_RF(ds, max_ds=np.arange(10, 15), max_fs=np.arange(1, 4))  # Grid Search
t1 = time()
print(t1-t0, '(s)')
