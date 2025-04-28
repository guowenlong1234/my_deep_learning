import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler


logger = logging.getLogger(__name__)


def get_loader(args):
    if args.local_rank not in [-1, 0]:      #如果不是主进程，则执行以下代码
        torch.distributed.barrier()         #同步操作，使所有的进程再这里等待，所有进程进行同步

    transform_train = transforms.Compose([#定义训练集的预处理流程
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),    #图像将进行随机缩放与裁剪，常见的数据增强手段，args.img_size表明最后的数据输出的尺寸，scale=(0.05, 1.0)说明随机裁剪范围在原图像的5%-100%之间
        transforms.ToTensor(),  #将数据转化为tensor格式
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),    #对图像进行标准化，mean表示每个通道的均值，std表示每个通道的标准差
    ])      #完成定义训练集的预处理流程
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])      #完成定义验证集集的预处理流程

    if args.dataset == "cifar10":       #如果采用cifar10数据集，加载ciffar10数据集
        #加载训练集
        trainset = datasets.CIFAR10(root="./data",                  #将数据集保存到路径
                                    train=True,                     #加载训练集
                                    download=True,                  #如果本地没有数据集，会下载
                                    transform=transform_train)      #预处理操作流程为之前定义的预处理操作
        #加载验证集
        testset = datasets.CIFAR10(root="./data",
                                   train=False,                     #加载验证集
                                   download=True,                   #允许通过网络下载数据集
                                   transform=transform_test) if args.local_rank in [-1, 0] else None    #只在主进程中加载测试集，

    else:   #同样的操作流程，下载cifar100数据集
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None
    if args.local_rank == 0:        #如果是主进程，再此对其节点，与上面的非主进程相对应
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)#如果是单机训练，则随机加载数据集，如果分布式训练，则采用切分数据集进行训练的方法
    test_sampler = SequentialSampler(testset)#对于验证集数据集，不需要打乱顺序，只需逐个的提供数据
    #生成训练集dataloader
    train_loader = DataLoader(trainset,     #获取数据
                              sampler=train_sampler,    #设置提供数据策略
                              batch_size=args.train_batch_size,     #设置batch_size
                              num_workers=0,        #使用的进程数量，通常这个是指的cpu进程加载数据
                              pin_memory=True)      #是否将数据都加载进内存，使用的话，将会加快从硬盘到cpu的速度，同时会占用内存空间
    #生成训练集的dataloader，策略与训练集相同
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=0,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader        #返回封装好的两个dataloader
