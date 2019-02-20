import numpy as np
import pandas as pa
import matplotlib.pyplot as plt
from RNN_ import Rnn_
COLUMNS_WL = ['workload', 'next_workload']
LABEL_WL = "next_workload"


def train():
    data = pa.read_csv('../data_new_new.csv')
    workload = data['workload']
    next_w = data['next_workload']
    wl_ = pa.concat([workload, next_w], axis=1)
    wl = wl_.values[:5800, :]
    wl_rnn = Rnn_(name='wl', data=wl)
    wl_rnn.train(64, 200, wl, 'wl_rnn_data1_new_conv_20.npz', len(COLUMNS_WL) - 1)


def predict():
    data = pa.read_csv('../data_new_new.csv')
    workload = data['workload']
    next_w = data['next_workload']
    wl_ = pa.concat([workload, next_w], axis=1)
    wl = wl_.values[-256: , :]
    wl_rnn = Rnn_(name='wl', data=wl)
    pre = np.around(wl_rnn.predict(wl, 'wl_rnn_data1_new_conv_20.npz', 64))
    re = wl[:, -1:]
    a = pre - re
    a = np.abs(a)
    print(np.mean(a / re))
    plt.ylabel('workload')
    plt.xlabel('time')
    plt.plot(list(range(len(pre))), pre, color='b', label='the predicted workload')
    plt.plot(list(range(len(re))), re, color='r', label='the real workload')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # train()
    predict()
