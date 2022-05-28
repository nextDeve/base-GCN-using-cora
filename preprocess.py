import pandas as pd
import numpy as np
import json
import random

random.seed(123456)


def preprocess():
    # 读取数据
    label = pd.read_csv('./label.csv', names=['id', 'label'])
    label['id'] = label['id'].astype(str)
    label['label'] = label['label'].astype(str)
    # 获取label缺失的数据
    ids_with_missing_label = []
    for i in range(len(label)):
        if pd.isna(label.iloc[i]['label']):
            ids_with_missing_label.append(label.iloc[i]['id'])
            label = label[label['id'] != label.iloc[i]['id']]
    # 读取特征与拓扑图
    # 读取特征并去掉label缺失的数据
    attrs = []
    with open("./attr.csv", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            split = line.split(',')
            if split[0] in ids_with_missing_label:
                continue
            attr = [0] * 1433
            for i in split[1:]:
                attr[int(i)] = 1
            attrs.append(attr)
    # 读取拓扑图并去掉与之相关的边
    edges = [[], []]
    with open("./adj_list.csv", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            split = line.split(',')
            no = split[0]
            if no in ids_with_missing_label:
                continue
            for node in split[1:]:
                if node not in ids_with_missing_label:
                    # 有向图
                    edges[0].append(int(no))
                    edges[1].append(int(node))
    # 处理label
    unique = label['label'].unique()
    label_to_num = {}
    for i, u in enumerate(unique):
        label_to_num[u] = i
    label_encode = []
    for i in range(len(label)):
        label_encode.append(label_to_num[label.iloc[i]['label']])
    # 生成训练集、测试集、验证集mask矩阵
    train_mask = [False] * 2708
    val_mask = [False] * 2708
    test_mask = [False] * 2708
    index = list(range(2708))
    train_size, test_size = int(2708 * 0.7), int(2708 * 0.2)
    train_index = random.sample(index, train_size)
    for i in train_index:
        train_mask[i] = True
        index.remove(i)
    test_index = random.sample(index, test_size)
    for i in test_index:
        test_mask[i] = True
        index.remove(i)
    for i in index:
        val_mask[i] = True
    return np.array(attrs), np.array(edges), np.array(label_encode), train_mask, val_mask, test_mask
