import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import warnings
from sklearn import preprocessing

from sympy.physics.units import years

warnings.filterwarnings("ignore")
import datetime

features = pd.read_csv('temps.csv')     #读取数据
learning_rate = 0.001
losses = []
epoch_size = 1000
id = 0              #选择使用的模型，1为手动设置的；0为使用框架设置的

years = features['year']
months = features['month']
days = features['day']                  #分别取出年月日的数据

dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]        #准备数据

# 准备画图
# 指定默认风格
plt.style.use('fivethirtyeight')

features = pd.get_dummies(features)                     #独热编码，将数据中星期一列转化为对应的01序列
labels = np.array(features['actual'])                   #在数据中取出标签
features= features.drop('actual', axis = 1)       #在特征中去掉标签数据

features = np.array(features)                           #将数据转化为对应的格式

input_features = preprocessing.StandardScaler().fit_transform(features)     #标准化数据，减去均值除以方差

if id == 1:
    #构建网络模型（一步一步复杂方法）
    x = torch.tensor(input_features, dtype = float)
    y = torch.tensor(labels, dtype = float)
    # 权重参数初始化
    weights = torch.randn((14, 128), dtype = float, requires_grad = True)
    biases = torch.randn(128, dtype = float, requires_grad = True)
    weights2 = torch.randn((128, 1), dtype = float, requires_grad = True)
    biases2 = torch.randn(1, dtype = float, requires_grad = True)

    for epoch in range(epoch_size):
        hidden = x.mm(weights) + biases                     #隐层1
        hidden = torch.relu(hidden)                         #激活函数relu
        predictions = hidden.mm(weights2) + biases2         #计算隐层2
        loss = torch.mean((predictions - y) ** 2)           #自定义激活函数，两者差的绝对值
        losses.append(loss.data.numpy())                    #存储损失

        if epoch % 100 == 0:
            print("loss:" ,loss.item())

        loss.backward()

        #更新参数
        weights.data.add_(- learning_rate * weights.grad.data)
        biases.data.add_(- learning_rate * biases.grad.data)
        weights2.data.add_(- learning_rate * weights2.grad.data)
        biases2.data.add_(- learning_rate * biases2.grad.data)

        # 每次迭代都得记得清空
        weights.grad.data.zero_()
        biases.grad.data.zero_()
        weights2.grad.data.zero_()
        biases2.grad.data.zero_()

elif id == 0:
    #采用框架提供的构建方法
    input_size = input_features.shape[1]        #input_features.shape = [样本数量，每个样本尺寸]
    hidden_size = 128                           #设置中间隐层大小
    output_size = 1                             #最终的输出值大小
    batch_size = 16                             #一个batch的大小
    my_nn = torch.nn.Sequential(                #
        torch.nn.Linear(input_size, hidden_size),#定义一个全连接层
        torch.nn.Sigmoid(),                     #激活函数采用sigmoid函数
        torch.nn.Linear(hidden_size, output_size),#定义输出层
    )
    cost = torch.nn.MSELoss(reduction='mean')   #调用损失函数，为均方损失函数，loss(x,y) = (x-y)**2
    optimizer = torch.optim.Adam(my_nn.parameters(), lr=0.001)      #定义优化器
    losses = []
    for epoch in range(epoch_size):
        batch_loss = []
        for start in range(0, len(input_features), batch_size):         #计算样本batch起始标号
            end = start + batch_size if start + batch_size < len(input_features) else len(input_features)       #计算样本end标号
            xx = torch.tensor(input_features[start:end], dtype=torch.float, requires_grad=True)
            yy = torch.tensor(labels[start:end], dtype = torch.float, requires_grad = True)
            prediction = my_nn(xx)              #计算模型输出预测值
            loss = cost(prediction, yy)         #计算损失函数
            optimizer.zero_grad()

            loss.backward(retain_graph=True)
            optimizer.step()
            batch_loss.append(loss.data.numpy())

        # 打印损失
        if epoch % 100 == 0:
            losses.append(np.mean(batch_loss))
            print(epoch, np.mean(batch_loss))
