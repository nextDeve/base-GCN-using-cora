import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier


class GCN(nn.Module):
    def __init__(self, num_node_features, classes):
        super(GCN, self).__init__()
        self.conV1 = GCNConv(num_node_features, 16)
        self.conV2 = GCNConv(16, classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conV1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conV2(x, edge_index)
        return F.log_softmax(x, dim=1)


class ANN(torch.nn.Module):
    def __init__(self, num_node_features, classes):
        super(ANN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_node_features, 2048),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, classes)
        )

    def forward(self, x):
        hidden = self.layers(x)
        return hidden


class SVM:
    def __init__(self):
        self.clf = OneVsRestClassifier(SVC(probability=True))

    def fit(self, x, y):
        self.clf.fit(x, y)

    def predict(self, x):
        return self.clf.predict_proba(x)
