from preprocess import preprocess
import torch
from train import train_gcn, train_svm, train_ann
from dataset import GCNDataset
import matplotlib.pyplot as plt
import time

num_class, num_node_features = 7, 1433
attr, edges, label, train_mask, val_mask, test_mask = preprocess()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = GCNDataset(attr, edges, label, train_mask, val_mask, test_mask, device)
times = []

# 训练svm
start = time.time()
svm_res = train_svm(attr[train_mask], label[train_mask], attr[test_mask], label[test_mask])
end = time.time()
times.append(end - start)
# 训练gcn
start = time.time()
gcn_res = train_gcn(data, num_class, num_node_features, device)
end = time.time()
times.append(end - start)
# 训练ann
start = time.time()
ann_res = train_ann(attr, label, train_mask, val_mask, test_mask, num_node_features, num_class, device)
end = time.time()
times.append(end - start)

font_label = {'family': 'Times New Roman', 'weight': 'normal', 'size': 24, }
font_legend = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18, }

plt.figure(figsize=(12, 8), dpi=500)
x = range(len(gcn_res[0]))
plt.plot(x, gcn_res[0], label='train_loss')
plt.plot(x, gcn_res[1], label='val_loss')
plt.xlabel('Epoch', font_label)
plt.ylabel('NLL_LOSS', font_label)
plt.legend(prop=font_legend, loc='upper right')
plt.savefig('./img/gcn_loss.svg', bbox_inches='tight', pad_inches=0)

plt.figure(figsize=(12, 8), dpi=500)
x = range(len(gcn_res[0]))
plt.plot(x, gcn_res[2], label='train_acc')
plt.plot(x, gcn_res[3], label='val_acc')
plt.xlabel('Epoch', font_label)
plt.ylabel('ACC', font_label)
plt.legend(prop=font_legend, loc='upper right')
plt.savefig('./img/gcn_acc.svg', bbox_inches='tight', pad_inches=0)

plt.figure(figsize=(12, 8), dpi=500)
x = range(len(ann_res[0]))
plt.plot(x, ann_res[0], label='train_loss')
plt.plot(x, ann_res[1], label='val_loss')
plt.xlabel('Epoch', font_label)
plt.ylabel('Cross Entropy Loss', font_label)
plt.legend(prop=font_legend, loc='upper right')
plt.savefig('./img/fnn_loss.svg', bbox_inches='tight', pad_inches=0)

plt.figure(figsize=(12, 8), dpi=500)
x = range(len(ann_res[0]))
plt.plot(x, ann_res[2], label='train_acc')
plt.plot(x, ann_res[3], label='val_acc')
plt.xlabel('EPOCH', font_label, labelpad=50)
plt.ylabel('ACC', font_label)
plt.legend(prop=font_legend, loc='upper right')
plt.savefig('./img/fnn_acc.svg', bbox_inches='tight', pad_inches=0)

print("测试集正确率(20%):\tSVM:{}\tGCN:{}\tANN:{}".format(svm_res, gcn_res[4], ann_res[4]))
print("训练耗时(s):\tSVM:{}\tGCN:{}\tANN:{}".format(times[0], times[1], times[2]))
