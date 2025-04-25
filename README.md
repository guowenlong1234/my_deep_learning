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

# 04_图像识别模型与训练策略
## 数据与任务
数据存放在路径'./flower_data/'，其中有两个文件夹，分别为训练数据集与测试集。每个数据集中图片大小不同，以对应的子文件夹名称序号作为标签

通过训练神经网络，做出102分类任务，分类出每种花的id，查询cat_to_name.json文件得到对应的花的名字
## 训练策略
数据增强策略
```python
from torchvision import transforms, models, datasets
data_transforms = {
    'train': 
        transforms.Compose([
        transforms.Resize([96, 96]),
        transforms.RandomRotation(45),#随机旋转，-45到45度之间随机选
        transforms.CenterCrop(64),#从中心开始裁剪
        transforms.RandomHorizontalFlip(p=0.5),#随机水平翻转 选择一个概率概率
        transforms.RandomVerticalFlip(p=0.5),#随机垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),#参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
        transforms.RandomGrayscale(p=0.025),#概率转换成灰度率，3通道就是R=G=B
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#均值，标准差
    ]),
    'valid': 
        transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
```
加载预训练模型模块
```python
from torchvision import transforms, models, datasets
model_name = 'resnet'  #可选的比较多 ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
#是否用人家训练好的特征来做
feature_extract = True #都用人家特征，咱先不更新
model_ft = models.resnet18()#18层的能快点，条件好点的也可以选152
```
先使用预训练模型训练修改之后的输出层，再解冻其他层，训练所有参数。


## 代码

加载通过文件夹结构组织数据方法
```python
import os
from torchvision import transforms, models, datasets
data_dir = './flower_data/'
#数据加载，datasets.ImageFolder通过文件夹结构加载数据，加载为字典结构。ImageFolder类需要两个参数：root 和 transform。root是数据集根目录；transform指定对每个图像应该执行的预处理操作，例如调整大小、裁剪、翻转等。
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
```
是否使用Gpu训练
```python
import  torch
# 是否用GPU训练
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)
```

学习率调度器
```python
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
```
- 每隔 step_size 个 epoch，将优化器的学习率乘以 gamma（即按一定比例衰减）。
- optimizer_ft:已定义的优化器（如 torch.optim.SGD 或 torch.optim.Adam）。调度器会基于此优化器的当前学习率进行调整。

保存训练结果
```
if phase == 'valid' and epoch_acc > best_acc:
    best_acc = epoch_acc
    best_model_wts = copy.deepcopy(model.state_dict())
    state = {
        'state_dict': model.state_dict(),  # 字典里key就是各层的名字，值就是训练好的权重
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filename)
```
## 未解决的问题
在训练过程中，显存占用率正常，CPU占用率也没有达到瓶颈，但是GPU占用率很低，而且会突然高一下，感觉应该是给GPU送数据速度跟不上，后来更新了dataloder加载方式也没什么改善，可能能通过将数据全部封装处理好一起放进内存再训练改善。
## debug
### 首次执行代码下载预训练模型会遇到网络问题

报错信息：URLError: urlopen error Remote end closed connection without response

解决办法，找到报错信息中的下载路径与模型存储Downloading:
"https://download.pytorch.org/models/resnet18-f37072fd.pth" to C:\Users\98712/.cache\torch\hub\checkpoints\resnet18-f37072fd.pth
手动下载模型文件并将其保存在指定路径中，再次运行代码将会从本地加载预训练模型。

## 结果展示
| 模型名称     | 数据增强 | 加载预训练模型 | 训练输出层 | 输出层epoch | Resize  | time    | val_acc |   |   |   |   |   |   |
|----------|------|---------|-------|----------|---------|---------|---------|---|---|---|---|---|---|
| resnet18 | 是    | 是       | 是     | 10       | 96*96   | 5m35s   | 39.36%  |   |   |   |   |   |   |
|          |      |         |       | 10       | 256*256 | 13m 1s  | 87.16％  |   |   |   |   |   |   |
|          |      |         |       | 30       | 96*96   | 16m 27s | 41.19％  |   |   |   |   |   |   |
|          |      |         |       | 30       | 256*256 | 43m 13s | 90.95%  |   |   |   |   |   |   |
|          |      |         |       |          |         |         |         |   |   |   |   |   |   |
|          |      |         |       |          |         |         |         |   |   |   |   |   |   |
|          |      |         |       |          |         |         |         |   |   |   |   |   |   |
|          |      |         | 训练所有层 | 所有层epoch | Resize  | time    | val_acc |   |   |   |   |   |   |
|          |      |         | 是     | 10       | 96*96   | 5m 38s  | 76.77%  |   |   |   |   |   |   |
|          |      |         |       | 10       | 256*256 | 16m 27s | 95.35%  |   |   |   |   |   |   |
|          |      |         |       | 30       | 96*96   | 16m 50s | 78.24％  |   |   |   |   |   |   |
|          |      |         |       | 30       | 256*256 | 46m 20s | 95.11%  |   |   |   |   |   |   |
|          |      |         |       |          |         |         |         |   |   |   |   |   |   |
|          |      |         |       |          |         |         |         |   |   |   |   |   |   |
|          |      |         |       |          |         |         |         |   |   |   |   |   |   |
|          |      |         |       |          |         |         |         |   |   |   |   |   |   |

# 05_dataloader_自定义数据集

## dataloader总体功能
提供两个list，其中一个存储所有图像地址，另外一个存储所有的标签数据，要根据标号一一对应。


- 1.注意要使用from torch.utils.data import Dataset, DataLoader
- 2.类名定义class FlowerDataset(Dataset)，其中FlowerDataset可以改成自己的名字
- 3.def __init__(self, root_dir, ann_file, transform=None):咱们要根据自己任务重写
- 4.def __getitem__(self, idx):根据自己任务，返回图像数据和标签数据

## debug（libiomp5md.dll文件报错看过来）
### 报错信息：
```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
```
错误原因：
问题出现主要是因为torch包中包含了名为libiomp5md.dll的文件，与Anaconda环境中的同一个文件出现了某种冲突，所以需要删除一个。

### 临时解决方法：
```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
```
这个方法会告诉 OpenMP 允许多个运行时库并存，但是它有潜在的性能损失，并且可能导致崩溃或结果不准确。

### 根本解决方法
我使用的是Anaconda环境，删除base环境中的文件，删除了C:\Users\98712\anaconda3\Library\bin\libiomp5md.dll
将其放在C:\Users\98712\anaconda3\Library\bin\libiomp5md\libiomp5md.dll路径下。**之后使用base环境如果这个文件报错，将这个文件重新放回
原来的路径中并且删除另外一个文件**

解决方法网址:[网址1](https://blog.csdn.net/Victor_X/article/details/110082033?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control)
[网址2](https://zhuanlan.zhihu.com/p/371649016)

# 06_vit源码
















