import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
from torch.utils.data import Dataset, DataLoader
import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image

#给出文件路径
data_dir = './flower_data/'
train_dir = data_dir + '/train_filelist'
valid_dir = data_dir + '/val_filelist'

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

#图像预处理操作过程
data_transforms = {
    'train':
        transforms.Compose([
            transforms.Resize(64),
            transforms.RandomRotation(45),  # 随机旋转，-45到45度之间随机选
            transforms.CenterCrop(64),  # 从中心开始裁剪
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
            transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
            # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
            transforms.RandomGrayscale(p=0.025),  # 概率转换成灰度率，3通道就是R=G=B
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
        ]),
    'valid':
        transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
}
#实例化两个数据集，一个train，一个valid
train_dataset = FlowerDataset(root_dir=train_dir, ann_file = './flower_data/train.txt', transform=data_transforms['train'])
val_dataset = FlowerDataset(root_dir=valid_dir, ann_file = './flower_data/val.txt', transform=data_transforms['valid'])
#将两个数据集分别加载为dataloder格式
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
#到此为止，dataloader全部构造完毕，为了确保输入模型的正确，需要验证一下是否正确

data_iter = iter(train_loader)  # 获取数据迭代器
images, labels = next(data_iter)  # 获取一个 batch 数据（64*3*64*64）
sample = images[0].squeeze()                     #取出其中的一个数据，进行squeeze()操作（1*3*64*64）->（3*64*64）
sample = sample.permute((1, 2, 0)).numpy()      #为了展示，颜色通道应放到最后（3*64*64）->（64*64*3）
sample *= [0.229, 0.224, 0.225]
sample += [0.485, 0.456, 0.406]
plt.imshow(sample)
plt.show()
print('Label is: {}'.format(labels[0].numpy()))


