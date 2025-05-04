import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import numpy as np

#定义超参数
input_size = 28         #图像总尺寸
num_classes = 10        #标签的种类数
num_epochs = 10          #训练的总循环周期
batch_size = 64         #一个撮（批次）的大小，64张图片
learning_rate = 0.001   #定义学习率

# 训练集
train_dataset = datasets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

# 测试集
test_dataset = datasets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# 构建batch数据
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,          #图片输入通道数
                out_channels=16,        #输出的通道数，卷积核的数量
                kernel_size=5,          #卷积核大小
                stride=1,               #步长大小
                padding=2               #周围补像素数量，为了保证输入输出的大小相同，输出为（16*28*28）
            ),
        nn.ReLU(),                      #激活函数relu
        nn.MaxPool2d(kernel_size=2)     #2*2池化操作，输出为（16*14*14）
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),  #设置第二层中的卷积参数,输出为（32*14*14）
            nn.ReLU(),
            nn.MaxPool2d(2)             #池化操作，输出为（32*7*7）
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2), nn.ReLU(),  #输出为（64*7*7）
            nn.ReLU()
        )
        self.out = nn.Linear(64 * 7 * 7, 10)        #全连接层得到结果

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)           #将矩阵转化为（batch_size,len)维度，其中len维度通过计算自动获得
        output = self.out(x)
        return output

def accuracy(predictions, labels):          #计算准确率函数
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)

net = CNN()         #实例化模型
criterion = nn.CrossEntropyLoss() #损失函数调用交叉熵函数
optimizer = optim.Adam(net.parameters(), lr=learning_rate) #设置优化器

#开始训练
for epoch in range(num_epochs):
    #保留当前epoch结果
    train_rights = []
    for batch_idx, (data, target) in enumerate(train_loader):
        net.train()
        output = net(data)                  #计算模型输出
        loss = criterion(output, target)    #计算损失函数
        optimizer.zero_grad()               #梯度清零
        loss.backward()                     #计算反向传播
        optimizer.step()                    #更新参数
        right = accuracy(output, target)    #计算准确率
        train_rights.append(right)          #记录当前batch准确率

        if batch_idx % 100 == 0:            #每经过100个batch测试一下准确率
            net.eval()                      #验证模式
            val_rights = []

            for (data, target) in test_loader:      #取出数据与标签
                output = net(data)                  #计算模型输出
                right = accuracy(output, target)    #计算准确率
                val_rights.append(right)

            train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
            val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))

            print('当前epoch: {} [{}/{} ({:.0f}%)]\t损失: {:.6f}\t训练集准确率: {:.2f}%\t测试集正确率: {:.2f}%'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                loss.data,
                       100. * train_r[0].numpy() / train_r[1],
                       100. * val_r[0]/ val_r[1]))
