# 01_神经网络_手写字体分类任务
## 数据
数据为维度[28, 28, 1]手写字体灰度图，已经完成打包结构化
## 模型
- 两层简单的全连接神经网络，

- 第一层输入784，输出128，采用relu激活函数。

- 第二层输入128，输出256，采用relu激活函数。

- 输出层输入256，输出10，均为全链接层。

- 损失函数使用交叉熵函数F.cross_entropy

- 所有数据随机初始化
- 设置随机dropout
## 参数
- 通过dropout_Prob参数设置dropout
- lr参数设置学习率
## 代码
设置网络基本结构
```python
from torch import nn
import torch.nn.functional as F
class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 256)
        self.out  = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x
```
更新参数三件套
```python
loss.backward()
opt.step()
opt.zero_grad()
```
```python
#优化器设置
from torch import optim
opt = optim.Adam(model.parameters(), lr=0.001)
```
---
# 02_神经网络_气温回归预测
## 数据
| year | month | day | week | temp_2 | temp_1 | average | actual | friend |     |
|------|-------|-----|------|--------|--------|---------|--------|--------|-----|
| 0    | 2016  | 1   | 1    | Fri    | 45     | 45      | 45.6   | 45     | 29  |
| 1    | 2016  | 1   | 2    | Sat    | 44     | 45      | 45.7   | 44     | 61  |
| 2    | 2016  | 1   | 3    | Sun    | 45     | 44      | 45.8   | 41     | 56  |
| 3    | 2016  | 1   | 4    | Mon    | 44     | 41      | 45.9   | 40     | 53  |
| 4    | 2016  | 1   | 5    | Tues   | 41     | 40      | 46.0   | 44     | 41  |
* year,moth,day,week分别表示的具体的时间
* temp_2：前天的最高温度值
* temp_1：昨天的最高温度值
* average：在历史中，每年这一天的平均最高温度值
* actual：这就是我们的标签值了，当天的真实最高温度
* friend：这一列可能是凑热闹的，你的朋友猜测的可能值，咱们不管它就好了
## 模型
输入层-隐层-输出层

- 输入层[batch,14]
- 隐层为[batch,14,128]
- 输出层[batch,128,1]

## 代码
独热编码
```python
import pandas as pd 
features = pd.read_csv('temps.csv')
features = pd.get_dummies(features)
features.head(5)
#自动检测数据，将文本信息转化为数据，有多少个不同的字符串，数据维度将增加多少个维度
```

数据标准化操作
```python
from sklearn import preprocessing
import pandas as pd 
features = pd.read_csv('temps.csv')
input_features = preprocessing.StandardScaler().fit_transform(features)
#完成（x-miu）/sigma
```

---
# 03_卷积神经网络_测试
## 数据
数据与01_神经网络_手写字体任务分类相同，为6000张[28，28，1]的灰度图。内容为手写数字，共有0-9共10个标签。
## 模型
模型采用卷积神经网络，总体结构为:

### 输入层
- 尺寸[batch, 1, 28, 28]
### 卷积层1

- 输入尺寸[batch,1, 28, 28]
- 卷积核数量：16
- 卷积核大小：[5,5]
- 步长：1
- padding：2
- 激活函数：relu（）
- 池化[2,2]
- 输出：[batch, 16, 14, 14]
### 卷积层2

- 输入尺寸：[batch, 16, 14, 14]
- 卷积核数量：32
- 输出：[batch, 32, 7, 7]
- 其余与卷积层1相同

### 卷积层3

- 输入尺寸：[batch, 32, 7, 7]
- 卷积核数量：64
- 输出：[batch, 64, 7, 7]
- 无池化，其余与卷积层1相同

### 输出层

- 全连接层nn.Linear(64 * 7 * 7, 10)
- 输出为[batch, 10]

## 代码

一个基本的卷积层结构代码
```python
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( 
            nn.Conv2d(16, 32, 5, 1, 2),     # 输出 (32, 14, 14)
            nn.ReLU(),                      # relu层
            nn.MaxPool2d(2),                # 输出 (16, 7, 7)
        )
        #nn.Conv2d()参数说明
        # in_channels=16,          #图片输入通道数
        # out_channels=32,        #输出的通道数，卷积核的数量
        # kernel_size=5,          #卷积核大小
        # stride=1,               #步长大小
        # padding=2               #周围补像素数量，为了保证输入输出的大小相同，输出为（16*28*28）
```
