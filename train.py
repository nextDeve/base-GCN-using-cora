from models import GCN, ANN, SVM
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import ANNDataset


def train_gcn(data, num_class, num_node_features, device):
    model = GCN(num_node_features, num_class).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    model.train()
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    print('start GCN training...')
    for epoch in range(2000):
        optimizer.zero_grad()
        out = model(data)
        loss_train = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        train_loss.append(loss_train.item())
        loss_train.backward()
        optimizer.step()

        _, pred = out.max(dim=1)
        correct = float(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
        acc_train = correct / data.train_mask.sum().item()
        train_acc.append(acc_train)

        with torch.no_grad():
            model.eval()
            out = model(data)
            loss_val = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
            val_loss.append(loss_val.item())

            _, pred = out.max(dim=1)
            correct = float(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
            acc_val = correct / data.val_mask.sum().item()
            val_acc.append(acc_val)
        if epoch % 100 == 0:
            print('Train GCN,Epoch:{},Train Loss:{:.2f} Acc:{:.2f},Val Loss:{:.2f} Acc:{:.2f}'.format(
                epoch + 1, loss_train.item(), acc_train, loss_val.item(), acc_val))
    with torch.no_grad():
        model.eval()
        _, pred = model(data).max(dim=1)
        correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        acc = correct / data.test_mask.sum().item()
    return [train_loss, val_loss, train_acc, val_acc, acc]


def train_svm(x, y, test_x, test_y):
    model = SVM()
    print('start SVM training...')
    model.fit(x, y)
    _, pred = torch.tensor(model.predict(test_x), dtype=torch.float).max(dim=1)
    test_y = torch.tensor(test_y, dtype=torch.long)
    correct = float(pred.eq(test_y).sum().item())
    acc = correct / test_y.shape[0]
    return acc


def train_ann(attr, label, train_mask, val_mask, test_mask, num_node_features, num_class, device):
    batch_size = 32
    epochs = 30

    model = ANN(num_node_features, num_class).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=5e-4)

    train_dataset = ANNDataset(attr[train_mask], label[train_mask])
    val_dataset = ANNDataset(attr[val_mask], label[val_mask])
    test_dataset = ANNDataset(attr[test_mask], label[test_mask])

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    print('start ANN training...')
    model.train()
    for epoch in range(epochs):
        train_epoch_loss, val_epoch_loss = 0, 0
        batch_num_train, batch_num_val = 0, 0
        correct_train, correct_val = 0, 0
        for b, data in enumerate(train_loader):
            batch_num_train = b + 1
            x, y = data[0].to(device), data[1].to(device)
            out = model(x)
            loss = F.cross_entropy(out, y.long())
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()
            _, pred = out.max(dim=1)
            correct_train += float(pred.eq(y).sum().item())
        train_epoch_loss /= batch_num_train
        with torch.no_grad():
            model.eval()
            for b, data in enumerate(val_loader):
                batch_num_val = b + 1
                x, y = data[0].to(device), data[1].to(device)
                out = model(x)
                loss = F.cross_entropy(out, y.long())
                val_epoch_loss += loss.item()
                _, pred = out.max(dim=1)
                correct_val += float(pred.eq(y).sum().item())
            val_epoch_loss /= batch_num_val
        train_loss.append(train_epoch_loss), val_loss.append(val_epoch_loss)
        acc_train = correct_train / (batch_num_train * batch_size)
        acc_val = correct_val / (batch_num_val * batch_size)
        train_acc.append(acc_train)
        val_acc.append(acc_val)
        print('Train ANN,Epoch:{},Train Loss:{:.2f} Acc:{:.2f},Val Loss:{:.2f} Acc:{:.2f}'.format(
            epoch + 1, train_epoch_loss, acc_train, val_epoch_loss, acc_val))
    correct = 0
    num = 0
    with torch.no_grad():
        model.eval()
        for data in test_loader:
            x, y = data[0].to(device), data[1].to(device)
            out = model(x)
            _, pred = out.max(dim=1)
            correct += float(pred.eq(y).sum().item())
            num += batch_size
    acc = correct / num
    return [train_loss, val_loss, train_acc, val_acc, acc]
