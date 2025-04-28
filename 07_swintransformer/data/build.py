# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from timm.data.transforms import str_to_pil_interp

from .cached_image_folder import CachedImageFolder
from .samplers import SubsetRandomSampler


def build_loader(config):
    config.defrost()        #解冻配置文件
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)   #build_dataset返回训练数据的dataset，和训练集分类类别数量
    config.freeze()         #冻结配置文件
    #print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)       #设置验证集dataset共1000个
    #print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    #num_tasks = dist.get_world_size()
    num_tasks = 1       #使用单进程训练
    global_rank = 0     #当前的进程号0
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':   #如果采用了压缩文件形式并且采用部分缓存策略
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())#间隔进行选择样本进行分布式训练
        sampler_train = SubsetRandomSampler(indices)    #每个进程根据索引进行采样抽取数据
    else:
        # 为分布式训练设置采样器，dataset_train数据集，num_replicas训练的总进程数量，rank=global_rank表示当前进程的标号，shuffle随机打乱顺序
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    indices = np.arange(0, len(dataset_val), 1)     #为验证数据生成一个索引
    sampler_val = SubsetRandomSampler(indices)      #设置验证数据的采样器
    #生成训练集的dataloader。dataset_train数据集，sampler采样器如何从数据集中加载数据，num_workers进程数，pin_memory是否加载到固定内存中，drop_last如果不能整除，是否舍弃最后一个小批次
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=0,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )
    #生成验证数据dataloader，不打乱，不舍弃最后一个小batch
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None#根据一些参数决定是否采用Mixup数据增强策略，
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)
    '''
    Mixup：是一种数据增强技术，它通过将两张图片线性混合（加权和）来生成新样本。这种方法有助于模型更加鲁棒地学习，尤其是在数据较少的情况下。
    CutMix：是一种扩展了 Mixup 的技术，它将两张图片的矩形区域进行切割和交换。这不仅能保持图像的一致性，还可以通过改变图片的局部区域来提高模型的泛化能力。
    '''
    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)   #build_transform，生成数据预处理流程函数，生成训练集数据则is_train = true，验证集则为false
    if config.DATA.DATASET == 'imagenet':   #数据集是imagenet
        prefix = 'train' if is_train else 'val'     #判断是验证阶段韩式训练阶段，对prefix赋值
        if config.DATA.ZIP_MODE:    #如果使用的是压缩格式，执行解压操作
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"  #构造压缩文件访问路径
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')  #从压缩文件中获取数据
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)      #构造数据访问路径
            dataset = datasets.ImageFolder(root, transform=transform)   #通过文件夹形式获取数据
        nb_classes = 1000   #类别1000个
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    #生成训练集数据则is_train = true，验证集则为false
    resize_im = config.DATA.IMG_SIZE > 32 #判断图片输入尺寸是否大于32
    if is_train:    #训练模式则执行
        # this should always dispatch to transforms_imagenet_train
        #这是针对训练集数据的预处理流程，调用内置的create_transform函数进行数据增强
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,        #图像尺寸
            is_training=True,                       #训练模式验证
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,  #颜色扰动策略
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,    #自动强策略
            re_prob=config.AUG.REPROB,      #重新概率
            re_mode=config.AUG.REMODE,      #增强的策略
            re_count=config.AUG.RECOUNT,    #多少次重采样
            interpolation=config.DATA.INTERPOLATION,    #图像插值策略
        )
        if not resize_im:#如果图片很小，将会修改transform中的第一个操作，将其替换为RandomCrop(config.DATA.IMG_SIZE, padding=4)，否则裁剪后数据过小
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=str_to_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=str_to_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
