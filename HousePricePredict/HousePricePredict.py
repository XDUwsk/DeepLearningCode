import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
import time
from torch.utils.data import DataLoader, Dataset

train_data = pd.read_csv('../data/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('../data/house-prices-advanced-regression-techniques/test.csv')

print(train_data.shape)
print(test_data.shape)

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True)

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)


class self_dataset(Dataset):
    def __init__(self, data, label=None):
        self.data = data
        if label is not None:
            self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        if self.label is not None:
            labels = self.label[index]
            return data,labels
        return data


train_set = self_dataset(train_features[:int(len(train_features) * 4 / 5)], train_labels[:int(len(train_features) * 4 / 5)])
val_set = self_dataset(train_features[int(len(train_features) * 4 / 5):], train_labels[int(len(train_features) * 4 / 5):])
test_label=torch.rand(size=[len(test_features),1]).fill_(1)
test_set=self_dataset(test_features,test_label)


batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_loader=DataLoader(test_set,batch_size=batch_size)

class Predict(nn.Module):
    def __i  nit__(self):
        super(Predict, self).__init__()
        self.line1 = nn.Linear(all_features.shape[1], 1)

    def forward(self, x):
        out = self.line1(x)

        return out


model = Predict()

loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
num_epoch = 100

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()
    for i, data in enumerate(train_loader):
        if data[0].shape[0] < batch_size:
            break
        optimizer.zero_grad()
        train_pred = model(data[0])
        batch_loss = loss(train_pred, data[1])
        batch_loss.backward()
        optimizer.step()

        train_acc += torch.mean(torch.abs(data[1] / train_pred))
        train_loss += batch_loss.item()


    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if data[0].shape[0] < batch_size:
                break
            val_pred = model(data[0])
            batch_loss = loss(val_pred, data[1])
            val_acc += torch.mean(torch.abs(data[1] / train_pred))
            val_loss += batch_loss.item()

    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
          (epoch + 1, num_epoch, time.time() - epoch_start_time, \
           train_acc / train_set.__len__(), train_loss / train_set.__len__(), val_acc / val_set.__len__(),
           val_loss / val_set.__len__()))


model.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model(data[0])
        test_label = test_pred.data.numpy()
        for y in test_label:
            prediction.append(y)

with open("predict.csv", 'w') as f:
    f.write('Id,SalePrice\n')
    for i, y in enumerate(prediction):
        f.write('{},{}\n'.format(i+1461, y.item()))



