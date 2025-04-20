from pathlib import Path
import pickle
import gzip
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
from torch import optim

DATA_PATH = Path("data")
PATH = DATA_PATH/"mnist"
FILENAME = "mnist.pkl.gz"
bs = 64
dropout_Prob = 0.5
lr = 0.001

PATH.mkdir(parents=True, exist_ok=True)     #允许创建父目录，目录已经存在不抛出异常

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")#x_train, y_train存储训练数据与标签、#x_valid, y_valid存储验证数据标签

x_train, y_train, x_valid, y_valid = map(torch.as_tensor, (x_train, y_train, x_valid, y_valid))     #map(迭代函数，迭代内容)

loss_func = F.cross_entropy     #调用交叉熵函数作为损失函数

def model(xb):
    """计算WX+b函数"""
    return xb.mm(weights) + bias

xb = x_train[0:bs]
yb = y_train[0:bs]
weights = torch.randn([784, 10], dtype = torch.float,  requires_grad = True)    #随机初始化w，（维度，数据类型，是否更新）
bias = torch.zeros(10, requires_grad=True)      #初始化b=0

class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 256)
        self.out = nn.Linear(256, 10)
        self.dropout = nn.Dropout(dropout_Prob)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x

net_test = Mnist_NN()

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

def loss_batch(model,loss_func,xb,yb, opt = None):
    loss =  loss_func(model(xb),yb)     #计算损失

    if opt is not None:
        loss.backward()     #计算反向传播梯度
        opt.step()          #更新参数
        opt.zero_grad()     #清除参数累计

    return loss.item(),len(xb)      #返回loss的具体数值而不是tensor格式


def fit(steps, model, loss_func, opt, train_dl, valid_dl):
    for step in range(steps):

        model.train()           #模型训练模式
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()            #模型验证模式
        with torch.no_grad():   #禁止梯度操作
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])      #解包操作  [a for i in range(1, 11)]会生成一个列表  loss_batch函数不传入优化器，默认为只计算损失，不更新参数

        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)     #数组相乘再相加，除以总数，计算平均损失
        print('当前step:' + str(step), '验证集损失：' + str(val_loss))

        correct = 0             #计算验证集准确率
        total = 0
        for xb, yb in valid_dl:
            outputs = model(xb)                                 #取出模型输出结果
            _, predicted = torch.max(outputs.data, 1)           #取出模型输出结果最大的对应的序号
            total += yb.size(0)                                 #计算总数

            correct += (predicted == yb).sum().item()           #计算正确的数量
        print(str(100 * correct / total) + "%")

def get_model(lr=0.001):
    model = Mnist_NN()
    return model, optim.Adam(model.parameters(), lr)    #实例化模型，优化器

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(20, model, loss_func, opt, train_dl, valid_dl)