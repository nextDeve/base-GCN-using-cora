import torch
from torch.utils.data import Dataset
import numpy as np


class GCNDataset(torch.nn.Module):

    def __init__(self, x, edge_index, label, train_mask, val_mask, test_mask, device):
        super(GCNDataset, self).__init__()
        self.x = torch.tensor(x, dtype=torch.float).to(device)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
        self.y = torch.tensor(label, dtype=torch.long).to(device)
        self.train_mask = torch.tensor(train_mask, dtype=torch.bool).to(device)
        self.val_mask = torch.tensor(val_mask, dtype=torch.bool).to(device)
        self.test_mask = torch.tensor(test_mask, dtype=torch.bool).to(device)


class ANNDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)
        self.len = len(self.x)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.x[item], self.y[item]
