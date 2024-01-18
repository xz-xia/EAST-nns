import random
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def data_split(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.Random(0).shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

paths = []#数据所处炮号地址
val_data, train_data_full = data_split(paths[:], 0.1, shuffle=True)
test_data, train_data = data_split(train_data_full, 0.1, shuffle=True)
#数据文件格式为.npy
def read_npy(file):
    input = []
    output = []
    for path in os.listdir(file):
        if os.path.splitext(path)[-1] == '.npy':
            if path != 'PSIRZ.npy':
                input.append(np.load(os.path.join(file,path),allow_pickle=True))
            else:
                output.append(np.load(os.path.join(file,path),allow_pickle=True))
    return input,output
test_input = []
test_output = []
for i in test_data:
    test_input.append(read_npy(i)[0])
    test_output.append(read_npy(i)[1])
#处理输出数据
for i,ar in enumerate(test_output):
    test_output[i]=np.array(ar).reshape(-1,129,129)
test_output = np.concatenate(test_output,axis=0)
#处理输入数据
for j,arr in enumerate(test_input):
    arr[0]=arr[0].reshape(-1,1)
    arr[3]=arr[3].reshape(-1,1)
    test_input[j]=np.concatenate(arr,axis=1)
test_input = np.concatenate(test_input,axis=0)