import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
import imageio
import time
import random
import sys
import copy
import json
from PIL import Image

from torch.utils.data import Dataset, DataLoader

data_dir = './flower_data/'
train_dir = data_dir + '/train_filelist'
valid_dir = data_dir + '/val_filelist'

num_epochs = 10
num_epochs_all = 10
#训练参数
batch_size = 64

#数据增强策略参数
DA_resize = 96
DA_rotation = 45
DA_centercrop = 64

class FlowerDataset(Dataset):       #定义类名可以任意取
    def __init__(self, root_dir, ann_file, transform=None):
        self.ann_file = ann_file            #ann_file表示标签文件路径与文件名
        self.root_dir = root_dir            #root_dir表示图片数据路径
        self.img_label = self.load_annotations()    #构造字典，以图片文件路径为Key，图片标签为值
        self.img = [os.path.join(self.root_dir, img) for img in list(self.img_label.keys())]        #取出Key值，加上根目录路径，构成图片路径
        self.label = [label for label in list(self.img_label.values())]     #取出值，转化成标签值
        self.transform = transform          #完成图像预处理

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        image = Image.open(self.img[idx])           #每次调用随机生成一个idx索引，打开img文件，将文件数据存储在image中
        label = self.label[idx]                     #取出对应的标签值
        if self.transform:
            image = self.transform(image)           #如果有预处理操作，完成图像的预处理
        label = torch.from_numpy(np.array(label))   #标签值转化为tensor格式
        return image, label

    def load_annotations(self):
        '''
        完成图片的路径名称获取，空格分割，
        :return: 返回字典，键为图片名，值为标签数据。
        '''
        data_infos = {}
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                data_infos[filename] = np.array(gt_label, dtype=np.int64)
        return data_infos


#采用数据增强策略
data_transforms = {
    'train':
        transforms.Compose([
        transforms.Resize([DA_resize, DA_resize]),
        transforms.RandomRotation(DA_rotation),#随机旋转，-45到45度之间随机选
        transforms.CenterCrop(DA_centercrop),#从中心开始裁剪
        transforms.RandomHorizontalFlip(p=0.5),#随机水平翻转 选择一个概率概率
        transforms.RandomVerticalFlip(p=0.5),#随机垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),#参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
        transforms.RandomGrayscale(p=0.025),#概率转换成灰度率，3通道就是R=G=B
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#均值，标准差
    ]),
    'valid':
        transforms.Compose([
        transforms.Resize([DA_resize, DA_resize]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#数据加载，datasets.ImageFolder通过文件夹结构加载数据，加载为字典结构。ImageFolder类需要两个参数：root 和 transform。root是数据集根目录；transform指定对每个图像应该执行的预处理操作，例如调整大小、裁剪、翻转等。
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid']}
train_dataset = FlowerDataset(root_dir=train_dir, ann_file = './flower_data/train.txt', transform=data_transforms['train'])
val_dataset = FlowerDataset(root_dir=valid_dir, ann_file = './flower_data/val.txt', transform=data_transforms['valid'])
#将两个数据集分别加载为dataloder格式
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

image_datasets = {
    'train':train_dataset
    ,'valid':val_dataset
}
dataloaders = {
    'train':train_loader,
    'valid':val_loader
}
#到此为止，dataloader全部构造完毕，为了确保输入模型的正确，需要验证一下是否正确
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}         #dataset_sizes存储两个数据集数据个数
# class_names = train_dataset.classes                                   #存储标签值

#通过读取cat_to_name.json文件，存储ID对应的花朵名称，存储为字典结构，存储在cat_to_name
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

model_name = 'resnet'  #可选的比较多 ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
#是否用人家训练好的特征来做
feature_extract = True #都用人家特征，咱先不更新

# 是否用GPU训练
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#设置是否更新参数
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# model_ft = models.resnet18()#18层的能快点，条件好点的也可以选152

#将输出层修改为自己的输出层
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    set_parameter_requires_grad(model_ft, feature_extract)      #设置参数禁止更新

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)  # 类别数自己根据自己任务来，再将输出层替换，默认输出层为参数可更新

    input_size = 64  # 输入大小根据自己配置来

    return model_ft, input_size

model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)

#设置GPU训练
model_ft = model_ft.to(device)

#设置保存模型的名字
filename='best.pt'

#设置并且输出当前训练的那些层，将需要更新的层存储在数组params_to_update = []中
params_to_update = model_ft.parameters()        #存储模型所有的参数
print("Params to learn:")

if feature_extract:
    params_to_update = []#True表示使用预训练模型的所有参数，不进行更新
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:                                           #False表示不冻结预训练模型，所有参数都需要更新
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

#优化器设置
optimizer_ft = optim.Adam(params_to_update, lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)#学习率每7个epoch衰减成原来的1/10
criterion = nn.CrossEntropyLoss()

#训练模块
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25,filename='best.pt'):
    #记录执行时间
    since = time.time()
    best_acc = 0
    model.to(device)        #放在GPU训练

    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    LRs = [optimizer.param_groups[0]['lr']]                     #记录学习率
    best_model_wts = copy.deepcopy(model.state_dict())          #记录保存最好的模型的参数组合，最初为预训练模型的参数组合
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)                                         #输出表头

        #训练和验证写到一起
        for phase in ['train', 'valid']:        #判断当前是训练模式还是验证模式
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            #取数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()                   #梯度清零
                outputs = model(inputs)                 #获取模型输出
                loss = criterion(outputs, labels)       #计算损失函数
                _, preds = torch.max(outputs, 1)        #取出最大值对应的数组标号

                #判断是否是训练阶段，如果是，更新权重和梯度
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                #计算损失，判断模型输出是否正确
                running_loss += loss.item() * inputs.size(0)        #0表示batch那个维度
                running_corrects += torch.sum(preds == labels.data) #预测结果最大的和真实值是否一致

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            time_elapsed = time.time() - since  # 一个epoch我浪费了多少时间,返回结果为秒
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}%'.format(phase, epoch_loss, epoch_acc*100))

            #保存结果最好的模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),  # 字典里key就是各层的名字，值就是训练好的权重
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)

            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)     #保存每个epoch中的训练成绩
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)     #保存训练集每个epoch的训练成绩

        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()#学习率衰减

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}%'.format(best_acc*100))

    model.load_state_dict(best_model_wts)   #装载最好的一次训练结果
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs

model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs  = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=num_epochs)

#在训练好输出层的基础上继续训练所有层
for param in model_ft.parameters():
    param.requires_grad = True      #所有参数设置为允许梯度更新

# 再继续训练所有的参数，学习率调小一点
optimizer = optim.Adam(model_ft.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# 加载之前训练好的权重参数

checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])

model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs  = train_model(model_ft, dataloaders, criterion, optimizer, num_epochs=num_epochs_all,filename='best.pt')


























