# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from torch import optim as optim


def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):#检查模型是否有no_weight_decay这个属性，如果有这个属性，说明有些层不需要权重衰减
        skip = model.no_weight_decay()  #取出这些不需要衰减的层
    if hasattr(model, 'no_weight_decay_keywords'):  #检查模型是否有no_weight_decay_keywords这个属性
        skip_keywords = model.no_weight_decay_keywords()    #取出对应的模型
    parameters = set_weight_decay(model, skip, skip_keywords)   #设置模型权重衰减策略，#返回一个包含两个字典的数组，第一个是需要权重衰减的参数，第二个是不需要权重衰减的参数

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()     #获取OPTIMIZER的名称，并将其转化为小写字母
    optimizer = None
    if opt_lower == 'sgd':  #通过optimizer的值确定采用的优化器
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
        #parameters，要更新的参数
        # momentum=config.TRAIN.OPTIMIZER.MOMENTUM控制动量的超参数
        #nesterov=True启用动量
        #lr=config.TRAIN.BASE_LR基础学习率
        #weight_decay=config.TRAIN.WEIGHT_DECAY权重衰减参数
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):  #skip_list：一个包含不需要权重衰减的参数名称列表。skip_keywords：一个包含不需要权重衰减的参数名称的关键字列表
    has_decay = []  #用于存储需要进行衰减的参数
    no_decay = []   #用于存储不需要衰减的参数

    for name, param in model.named_parameters():    #遍历模型所有的参数，返回对应的张量以及对应的名称
        if not param.requires_grad:     #检查是否需要梯度更行，如果不需要梯度更新，说明参数被冻结，跳过这个参数
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):#参数为1，名称为".bias"结尾，或者名称在skip_list，或者那么包含在skip_keywords中，就将他加入到不需要参数衰减的数组中，如果不是，加入到需要衰减的数组中
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},#返回一个包含两个字典的数组，第一个是需要权重衰减的参数，第二个是不需要权重衰减的参数
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    #检查name是否在keywords中的任何一个关键字，如果包含，则返回true，说明这个参数不需要进行梯度衰减
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
